import logging
import traceback
from collections import defaultdict
from pathlib import Path

import av
import mergedeep as merge
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tqdm
import yaml
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import and_, or_

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    DatasetDB,
    EpisodeQcDB,
    TaskStatus,
)
from robocoin_dataset.distribution_computation.constant import (
    DATASET_UUID,
    DEVICE_MODEL,
    DEVICE_MODEL_VERSION,
    ERR_MSG,
    TASK_RESULT_CONTENT,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
)
from robocoin_dataset.distribution_computation.task_client import TaskClient
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.format_converter.tolerobot.constant import (
    LEFORMAT_PATH,
)
from robocoin_dataset.quality_check.checker_registry import (
    DATASET_DATA_CHECKERS,
    EPISODE_DATA_CHECKERS,
    EPISODE_VIDEO_CHECKERS,
)
from robocoin_dataset.utils.le_path import get_episodes_frames, get_parquet_files, get_video_files
# 替换为实际的checkers模块路径
from robocoin_dataset.quality_check.checkers import clear_video_decode_cache


QC_CONFIG = "qc_config"
QC_RESULT = "qc_result"
# MERGED_DATA_FEATURE = "merged"
MERGED_DATA_FEATURE = None


def get_checker_config(
    device_model: str, device_model_version: str, device_verison_config_file: str | Path
) -> dict:
    device_verison_config_file = Path(device_verison_config_file)
    checker_root_path = device_verison_config_file.parent
    device_verison_checker_config = yaml.safe_load(device_verison_config_file.open())
    default_config_path = checker_root_path / "default_checker_config.yaml"
    default_config = yaml.safe_load(default_config_path.open())
    if device_model not in device_verison_checker_config:
        return default_config

    if device_model_version not in device_verison_checker_config[device_model]:
        device_model_version = "default_version"

    version_idx = 0
    for i, v in enumerate(device_verison_checker_config[device_model]):
        if v == device_model_version:
            version_idx = i
            break

    base_config_path = (
        checker_root_path
        / device_verison_checker_config[device_model][version_idx]["base_config_path"]
    )
    specific_config_path = (
        checker_root_path
        / device_verison_checker_config[device_model][version_idx]["specific_config_path"]
    )

    base_config = yaml.safe_load(base_config_path.open())
    specific_config = yaml.safe_load(specific_config_path.open())

    return merge.merge(base_config, specific_config)


def check_length_consistency(
    repo_path: str | Path, data_feature: str | None, video_feature: str | None = None
) -> set[int]:
    bad_episodes = set()
    parquet_files = get_parquet_files(repo_path, data_feature)
    video_files = get_video_files(repo_path, video_feature)
    episode_nums = get_episodes_frames(repo_path)
    if len(parquet_files) != len(video_files):
        raise ValueError(
            f"The number of parquet files ({len(parquet_files)}) does not match the number of video files ({len(video_files)})."
        )

    for ep_idx in tqdm.tqdm(
        range(len(parquet_files)), desc="Checking Length Consistency", unit="episode"
    ):
        metadata = pq.read_metadata(parquet_files[ep_idx])
        row_num = metadata.num_rows
        if row_num != episode_nums[ep_idx]:
            bad_episodes.add(ep_idx)
            break
        for video_file in video_files[ep_idx]:
            container = av.open(video_file)
            stream = container.streams.video[0]  # 假设第一个视频流
            frame_count = stream.frames  # 可能为 0 或 None（如果未知）
            if frame_count != episode_nums[ep_idx]:
                bad_episodes.add(ep_idx)
                break
            container.close()

    return bad_episodes


def quality_check_pipeline(
    repo_path: str | Path, configs: dict, data_feature: str
) -> tuple[dict, dict]:
    dataset_data_checkers_config = configs.get("dataset_data_checkers", None)
    if not dataset_data_checkers_config:
        raise ValueError("No dataset_data_checkers config found.")

    episode_data_checkers_config = configs.get("episode_data_checkers", None)
    if not episode_data_checkers_config:
        raise ValueError("No episode_data_checkers config found.")

    episode_video_checkers_config = configs.get("episode_video_checkers", None)

    try:
        episode_nums = get_episodes_frames(repo_path)
        bad_data_episodes = check_length_consistency(repo_path, data_feature)
        for checker_cfg in dataset_data_checkers_config:
            name = checker_cfg["name"]
            func = DATASET_DATA_CHECKERS[name]
            params = checker_cfg.get("params", {})
            bad_data_episodes = func(episode_nums, bad_data_episodes, **params)

        state_data_scores = {}
        action_data_scores = {}
        parquet_files = get_parquet_files(repo_path, data_feature)
        state_data_scores_perchecker = defaultdict(dict)
        action_data_scores_perchecker = defaultdict(dict)

        # 定义不参与分数计算的data算子名单
        data_checkers_skip_score = {"motion_data_valid_frame_range"}

        for idx, parquet_path in tqdm.tqdm(
            enumerate(parquet_files),
            desc="Checking episodes",
            total=len(parquet_files),
            unit="episode",
        ):
            if idx in bad_data_episodes:
                continue

            df = pd.read_parquet(parquet_path)
            state_data = np.array(df["observation.state"].tolist())
            action_data = np.array(df["action"].tolist())
            state_score = 0
            action_score = 0
            total_weight = 0
            checked = False
            for checker_cfg in episode_data_checkers_config:
                name = checker_cfg.get("name")
                weight = checker_cfg.get("score_weight")
                checker_flag = checker_cfg.get("should_check", False)
                if not checker_flag:
                    continue
                
                # 校验基础配置
                if not name:
                    raise ValueError("Each episode_data_checker config must have 'name'.")
                if name not in data_checkers_skip_score and (not weight or weight <= 0.001):
                    raise ValueError(f"Episode data checker {name} must have valid 'weight'.")

                func = EPISODE_DATA_CHECKERS[name]
                params = checker_cfg.get("params", {})
                
                # 执行算子（无论是否参与分数计算，都执行）
                state_checker_result = func(state_data, **params)
                action_checker_result = func(action_data, **params)

                # 仅非跳过算子参与分数计算
                if name not in data_checkers_skip_score:
                    total_weight += weight
                    state_checker_score = 1 - state_checker_result
                    action_checker_score = 1 - action_checker_result
                    state_score += state_checker_score * weight
                    action_score += action_checker_score * weight
                    checked = True
                else:
                    # 跳过分数计算，但记录原始结果（有效帧区间）
                    state_checker_score = state_checker_result
                    action_checker_score = action_checker_result

                # 记录每个checker的结果（无论是否参与分数）
                state_data_scores_perchecker[idx][name] = state_checker_score
                action_data_scores_perchecker[idx][name] = action_checker_score

            # 只有有有效分数时才计算平均值
            if checked and total_weight > 0:
                state_data_scores[idx] = state_score / total_weight
                action_data_scores[idx] = action_score / total_weight

        if not episode_video_checkers_config:
            clear_video_decode_cache()
            return (
                {
                    "bad_data_episodes": list(bad_data_episodes),
                    "state_data_scores": state_data_scores,
                    "action_data_scores": action_data_scores,
                },
                {
                    "state_data_scores_perchecker": state_data_scores_perchecker,
                    "action_data_scores_perchecker": action_data_scores_perchecker,
                },
            )
        
        video_scores = {}
        video_scores_perchecker = defaultdict(dict)
        video_paths = get_video_files(repo_path)
        # 定义不参与分数计算的video算子名单
        video_checkers_skip_score = {"video_valid_frame_range"}

        for idx, paths in tqdm.tqdm(
            enumerate(video_paths), desc="Checking videos", unit="video", total=len(video_paths)
        ):
            if idx in bad_data_episodes:
                continue
            total_weight = 0
            video_score = 0
            checked_video = False
            for checker_cfg in episode_video_checkers_config:
                name = checker_cfg.get("name")
                func = EPISODE_VIDEO_CHECKERS[name]
                params = checker_cfg.get("params", {})
                weight = checker_cfg.get("score_weight")
                checker_flag = checker_cfg.get("should_check", False)
                if not checker_flag:
                    continue
                
                # 校验基础配置
                if not name:
                    raise ValueError("Each episode_video_checker config must have 'name'.")
                if name not in video_checkers_skip_score and (not weight or weight <= 0.001):
                    raise ValueError(f"Episode video checker {name} must have valid 'weight'.")

                # 执行算子（无论是否参与分数计算，都执行）
                video_checker_result = func(paths, episode_idx=idx, **params)

                # 仅非跳过算子参与分数计算
                if name not in video_checkers_skip_score:
                    total_weight += weight
                    video_checker_score = 1 - video_checker_result
                    # 分辨率不一致直接标记为bad
                    if name == "camera_resolution_consistency" and video_checker_score == 0.0:
                        bad_data_episodes.add(idx)
                        break
                    video_score += video_checker_score * weight
                    checked_video = True
                else:
                    # 跳过分数计算，记录原始结果（有效帧区间）
                    video_checker_score = video_checker_result

                # 记录每个checker的结果（无论是否参与分数）
                video_scores_perchecker[idx][name] = video_checker_score

            # 只有有有效分数时才计算平均值
            if checked_video and total_weight > 0:
                video_score /= total_weight
                video_scores[idx] = video_score

        clear_video_decode_cache()
        return (
            {
                "bad_data_episodes": list(bad_data_episodes),
                "state_data_scores": state_data_scores,
                "action_data_scores": action_data_scores,
                "video_scores": video_scores,
            },
            {
                "state_data_scores_perchecker": state_data_scores_perchecker,
                "action_data_scores_perchecker": action_data_scores_perchecker,
                "video_scores_perchecker": video_scores_perchecker,
            },
        )

    except Exception:
        raise



def _gen_one_dataset_quality_check_task(
    session: Session,
) -> tuple[str | None, str | None, str | None, str | None]:
    query = session.query(DatasetDB).filter(
        and_(
            DatasetDB.convert_status == TaskStatus.COMPLETED,
            or_(
                # 分支1: 正在排队
                DatasetDB.qc_status == TaskStatus.PENDING,
                # 分支2: 已完成但版本过期
                and_(
                    DatasetDB.qc_status == TaskStatus.COMPLETED,
                    DatasetDB.qc_version_ps != DatasetDB.data_merge_status,
                ),
            ),
        )
    )
    item = query.first()
    if not item:
        return None, None, None, None
    item.motion_annotation_status = TaskStatus.PROCESSING
    session.commit()
    return item.dataset_uuid, item.convert_path, item.device_model, item.device_model_version

def get_max_coverage_range(*ranges) -> tuple[int, int]:
    """
    从多个帧区间元组中提取最大覆盖范围（最左start + 最右end）
    自动过滤无效区间（start>end/负数），兜底返回(0,0)
    """
    valid_starts = []
    valid_ends = []
    
    # 遍历所有传入的区间，筛选有效值
    for start, end in ranges:
        if start >= 0 and end >= start:  # 仅保留有效区间
            valid_starts.append(start)
            valid_ends.append(end)
    
    # 无有效区间返回(0,0)，否则取最小start和最大end
    if not valid_starts or not valid_ends:
        return (0, 0)
    return (min(valid_starts), max(valid_ends))


def _build_episode_qc_summary(
    bad_data_episodes: list[int] | None = None,
    state_data_scores: dict[int, float] | None = None,
    action_data_scores: dict[int, float] | None = None,
    video_scores: dict[int, float] | None = None,
    state_data_scores_perchecker: dict[int, dict] | None = None,  # episode_data算子单独得分
    action_data_scores_perchecker: dict[int, dict] | None = None,  # episode_data算子单独得分
    video_scores_perchecker: dict[int, dict] | None = None,       # episode_video算子单独得分
) -> dict[int, dict]:
    # 转为 set 加速查找
    if bad_data_episodes is None:
        bad_data_episodes = []
    bad_set = set(bad_data_episodes)
    if state_data_scores is None:
        state_data_scores = {}
    if action_data_scores is None:
        action_data_scores = {}
    if video_scores is None:
        video_scores = {}

    state_data_scores_perchecker = state_data_scores_perchecker or defaultdict(dict)
    action_data_scores_perchecker = action_data_scores_perchecker or defaultdict(dict)
    video_scores_perchecker = video_scores_perchecker or defaultdict(dict)
    # 假设所有 score 列表长度一致，取其一作为总 episode 数

    episode_summary = defaultdict(dict)
    # 处理异常episode
    for episode_idx in bad_data_episodes:
        # 提取episode_data算子单独得分
        state_static_frame_rate_score = state_data_scores_perchecker.get(episode_idx, {}).get("static_frame_rate", 0.0)
        state_static_joint_score = state_data_scores_perchecker.get(episode_idx, {}).get("static_joint", 0.0)
        
        action_static_frame_rate_score = action_data_scores_perchecker.get(episode_idx, {}).get("static_frame_rate", 0.0)
        action_static_joint_score = action_data_scores_perchecker.get(episode_idx, {}).get("static_joint", 0.0)
        
        # 提取episode_video算子单独得分
        max_frame_stable_then_jump_rate_score = video_scores_perchecker.get(episode_idx, {}).get("max_frame_stable_then_jump_rate", 0.0)
        max_frame_jump_dist_score = video_scores_perchecker.get(episode_idx, {}).get("max_frame_jump_dist", 0.0)
        color_shift_detection_score = video_scores_perchecker.get(episode_idx, {}).get("video_color_shift_detection", 0.0)
        consecutive_static_frames_score = video_scores_perchecker.get(episode_idx, {}).get("consecutive_static_frames", 0.0)
        camera_resolution_score = video_scores_perchecker.get(episode_idx, {}).get("camera_resolution_consistency", 0.0)
        static_frame_diff = abs(state_static_frame_rate_score - action_static_frame_rate_score)
        is_diff = 1 if static_frame_diff > 0.1 else 0

        state_motion_data_range = state_data_scores_perchecker.get(episode_idx, {}).get("motion_data_valid_frame_range", (0, 0))
        action_motion_data_range = action_data_scores_perchecker.get(episode_idx, {}).get("motion_data_valid_frame_range", (0, 0))
        video_data_range = video_scores_perchecker.get(episode_idx, {}).get("video_valid_frame_range", (0, 0))
        max_coverage_range = get_max_coverage_range(
            state_motion_data_range,
            action_motion_data_range,
            video_data_range
        )
        max_start, max_end = max_coverage_range
        
        episode_summary[episode_idx].update(
            {
                "is_bad": 1,
                "is_diff": is_diff,
                # 原有分组汇总得分
                "state_data_score": state_data_scores.get(episode_idx, 0),
                "action_data_score": action_data_scores.get(episode_idx, 0),
                "video_score": video_scores.get(episode_idx, 1),
                # Episode Data 算子单独得分
                "episode_state_static_frame_rate_score": state_static_frame_rate_score,
                "episode_state_static_joint_score": state_static_joint_score,
                "episode_action_static_frame_rate_score": action_static_frame_rate_score,
                "episode_action_static_joint_score": action_static_joint_score,
                # Episode Video 算子单独得分
                "episode_video_max_frame_stable_then_jump_rate_score": max_frame_stable_then_jump_rate_score,
                "episode_video_max_frame_jump_dist_score": max_frame_jump_dist_score,
                "episode_video_color_shift_detection_score": color_shift_detection_score,
                "episode_video_consecutive_static_frames_score": consecutive_static_frames_score,
                "episode_video_camera_resolution_consistency_score": camera_resolution_score,
                "max_start":max_start,
                "max_end":max_end,
            }
        )
    # 处理正常episode
    for episode_idx in (
        set(state_data_scores.keys()) | set(action_data_scores.keys()) | set(video_scores.keys())
    ):
        # 提取episode_data算子单独得分
        state_static_frame_rate_score = state_data_scores_perchecker.get(episode_idx, {}).get("static_frame_rate", 0.0)
        state_static_joint_score = state_data_scores_perchecker.get(episode_idx, {}).get("static_joint", 0.0)

        action_static_frame_rate_score = action_data_scores_perchecker.get(episode_idx, {}).get("static_frame_rate", 0.0)
        action_static_joint_score = action_data_scores_perchecker.get(episode_idx, {}).get("static_joint", 0.0)
        
        # 提取episode_video算子单独得分
        max_frame_stable_then_jump_rate_score = video_scores_perchecker.get(episode_idx, {}).get("max_frame_stable_then_jump_rate", 0.0)
        max_frame_jump_dist_score = video_scores_perchecker.get(episode_idx, {}).get("max_frame_jump_dist", 0.0)
        color_shift_detection_score = video_scores_perchecker.get(episode_idx, {}).get("video_color_shift_detection", 0.0)
        consecutive_static_frames_score = video_scores_perchecker.get(episode_idx, {}).get("consecutive_static_frames", 0.0)
        camera_resolution_score = video_scores_perchecker.get(episode_idx, {}).get("camera_resolution_consistency", 0.0)
        static_frame_diff = abs(state_static_frame_rate_score - action_static_frame_rate_score)
        is_diff = 1 if static_frame_diff > 0.1 else 0

        state_motion_data_range = state_data_scores_perchecker.get(episode_idx, {}).get("motion_data_valid_frame_range", (0, 0))
        action_motion_data_range = action_data_scores_perchecker.get(episode_idx, {}).get("motion_data_valid_frame_range", (0, 0))
        video_data_range = video_scores_perchecker.get(episode_idx, {}).get("video_valid_frame_range", (0, 0))
        max_coverage_range = get_max_coverage_range(
            state_motion_data_range,
            action_motion_data_range,
            video_data_range
        )
        max_start, max_end = max_coverage_range
        
        episode_summary[episode_idx].update(
            {
                "is_bad": episode_idx in bad_set,
                "is_diff": is_diff,
                # 原有分组汇总得分
                "state_data_score": state_data_scores.get(episode_idx, 1),
                "action_data_score": action_data_scores.get(episode_idx, 1),
                "video_score": video_scores.get(episode_idx, 1),
                # Episode Data 算子单独得分
                "episode_state_static_frame_rate_score": state_static_frame_rate_score,
                "episode_state_static_joint_score": state_static_joint_score,
                "episode_action_static_frame_rate_score": action_static_frame_rate_score,
                "episode_action_static_joint_score": action_static_joint_score,
                # Episode Video 算子单独得分
                "episode_video_max_frame_stable_then_jump_rate_score": max_frame_stable_then_jump_rate_score,
                "episode_video_max_frame_jump_dist_score": max_frame_jump_dist_score,
                "episode_video_color_shift_detection_score": color_shift_detection_score,
                "episode_video_consecutive_static_frames_score": consecutive_static_frames_score,
                "episode_video_camera_resolution_consistency_score": camera_resolution_score,
                "max_start":max_start,
                "max_end":max_end,
            }
        )
    return episode_summary


def _sync_quality_check_tasks(
    session: Session, device_model: str | None = None, device_model_version: str | None = None
) -> None:
    query = session.query(DatasetDB).filter(
        and_(
            # 必要前提：convert必须成功
            DatasetDB.convert_status == TaskStatus.COMPLETED,
            # 两个触发分支
            or_(
                # 分支1: 正在排队
                DatasetDB.qc_status == TaskStatus.PENDING,
                # 分支2: 已完成但版本过期
                and_(
                    DatasetDB.qc_status == TaskStatus.COMPLETED,
                    DatasetDB.qc_version_ps < DatasetDB.data_merge_version,
                ),
            ),
        )
    )
    items = query.all()

    if not items:
        return

    for item in items:
        item.qc_status = TaskStatus.PENDING
        item.qc_version_ps = item.data_merge_version

    session.commit()


def _gen_one_dataset_quality_check_task_without_sync(
    session: Session,
) -> tuple[str | None, str | None, str | None, str | None]:
    query = session.query(DatasetDB).filter(
        and_(
            # 必要前提：convert必须成功
            DatasetDB.convert_status == TaskStatus.COMPLETED,
            DatasetDB.qc_status == TaskStatus.PENDING,
            DatasetDB.is_ignore == False,
        )
    )
    item = query.first()

    if not item:
        return None, None, None, None

    item.qc_status = TaskStatus.PROCESSING

    item.qc_version = item.qc_version + 1
    session.commit()

    # 获取 sim_replay 配置
    device_model = item.device_model
    device_model_version = item.device_model_version

    return item.dataset_uuid, item.convert_path, device_model, device_model_version


def _check_repo(repo_path: str | Path, checker_config: dict, data_feature: str) -> dict[int, dict]:
    # 注意：这里要保留第二个返回值（包含所有算子单独得分）
    qc_results, qc_detailed_results = quality_check_pipeline(repo_path, checker_config, data_feature=data_feature)
    bad_episodes, state_data_scores, action_data_scores, video_scores = (
        qc_results.get("bad_data_episodes", []),
        qc_results.get("state_data_scores", {}),
        qc_results.get("action_data_scores", {}),
        qc_results.get("video_scores", {}),
    )
    # 提取算子单独得分
    state_data_scores_perchecker = qc_detailed_results.get("state_data_scores_perchecker", {})
    action_data_scores_perchecker = qc_detailed_results.get("action_data_scores_perchecker", {})
    video_scores_perchecker = qc_detailed_results.get("video_scores_perchecker", {})

    return _build_episode_qc_summary(
        bad_episodes,
        state_data_scores=state_data_scores,
        action_data_scores=action_data_scores,
        video_scores=video_scores,
        state_data_scores_perchecker=state_data_scores_perchecker,  # 传递episode_data算子单独得分
        action_data_scores_perchecker=action_data_scores_perchecker,  # 传递episode_data算子单独得分
        video_scores_perchecker=video_scores_perchecker,            # 传递episode_video算子单独得分
    )



class DatasetQualityCheck:
    def __init__(
        self,
        db_file_path: str | Path,
        qc_config_path: str | Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.logger = logger or logging.getLogger(__name__)
        self.qc_config_path: Path = Path(qc_config_path).expanduser().absolute()

    def check_one_repo(self) -> None:
        with self.db.with_session() as session:
            _sync_quality_check_tasks(session=session)
            dataset_uuid, repo_path, device_model, device_model_version = (
                _gen_one_dataset_quality_check_task_without_sync(session=session)
            )

        if not dataset_uuid:
            return

        checker_config = get_checker_config(
            device_model,
            device_model_version,
            self.qc_config_path,
        )

        try:
            qc_results = _check_repo(repo_path, checker_config, MERGED_DATA_FEATURE)

            with self.db.with_session() as session:
                item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                if item:
                    item.qc_status = TaskStatus.COMPLETED
                else:
                    return
                session.query(EpisodeQcDB).filter(EpisodeQcDB.dataset_uuid == dataset_uuid).delete()
                for episode_idx, summary in qc_results.items():
                    episode_qc_item = EpisodeQcDB(
                        dataset_uuid=dataset_uuid,
                        episode_idx=episode_idx,
                        is_bad_episode=summary["is_bad"],
                        is_state_frame_diff=summary["is_diff"],
                        # 原有分组汇总得分
                        state_data_score=summary["state_data_score"],
                        action_data_score=summary["action_data_score"],
                        video_score=summary["video_score"],
                        # Episode Data 算子单独得分
                        episode_state_static_frame_rate_score=summary["episode_state_static_frame_rate_score"],
                        episode_state_static_joint_score=summary["episode_state_static_joint_score"],
                        episode_action_static_frame_rate_score=summary["episode_action_static_frame_rate_score"],
                        episode_action_static_joint_score=summary["episode_action_static_joint_score"],
                        # Episode Video 算子单独得分
                        episode_video_max_frame_stable_then_jump_rate_score=summary["episode_video_max_frame_stable_then_jump_rate_score"],
                        episode_video_max_frame_jump_dist_score=summary["episode_video_max_frame_jump_dist_score"],
                        episode_video_color_shift_detection_score=summary["episode_video_color_shift_detection_score"],
                        start_frame = summary.get("max_start", 0),
                        end_frame =  summary.get("max_end", 0),
                    )
                    session.add(episode_qc_item)
                session.commit()
        # 异常处理逻辑不变
        except Exception:
            with self.db.with_session() as session:
                item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                if item:
                    item.qc_status = TaskStatus.FAILED
                    item.qc_err_msg = traceback.format_exc()
                else:
                    return
                session.commit()
            self.logger.info(traceback.format_exc())


class DatasetQualityCheckServer(TaskServer):
    def __init__(
        self,
        db_file_path: str | Path,
        qc_config_path: str | Path,
        host: str = "0.0.0.0",
        port: int = 2010,
        heartbeat_interval: float = 30.0,  # 服务端每30秒发一次 ping
        timeout: float = 15.0,  # 等待 pong 超过15秒则断开
        logger: logging.Logger | None = None,
        target_dataset_uuid: str | None = None,  # 新增：接收指定的数据集UUID
    ) -> None:
        super().__init__(
            logger=logger,
            host=host,
            port=port,
            heartbeat_interval=heartbeat_interval,
            timeout=timeout,
        )
        db_file_path = Path(db_file_path).expanduser().absolute()

        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.logger = logger or logging.getLogger(__name__)
        # 新增：保存指定的目标数据集UUID
        self.target_dataset_uuid = target_dataset_uuid

        self.qc_config_path = Path(qc_config_path).expanduser().absolute()
        if not self.qc_config_path.exists():
            raise FileNotFoundError(f"QC config file {self.qc_config_path} not found.")

    def get_task_category(self) -> str:
        return "dataset quality check"

    def generate_task_content(self) -> dict | None:
        with self.db.with_session() as session:
            # 若指定了目标UUID，直接查询该数据集，跳过同步和自动筛选
            if self.target_dataset_uuid:
                item = session.query(DatasetDB).filter(
                    DatasetDB.dataset_uuid == self.target_dataset_uuid
                ).first()
                if not item:
                    self.logger.info(f"Dataset with UUID {self.target_dataset_uuid} not found.")
                    return None
                 # 新增：判断是否已完成，且不允许重复处理
                if item.qc_status == TaskStatus.COMPLETED:
                    self.logger.info(f"Dataset {self.target_dataset_uuid} has already been processed successfully, skip duplicate assignment.")
                    return None
                if item.is_ignore == True:
                    self.logger.info(f"Dataset {self.target_dataset_uuid} has already been ignored, skip duplicate assignment.")
                    return None
                # 强制校验前置条件：convert_status完成
                if item.convert_status != TaskStatus.COMPLETED:
                    self.logger.info(f"Dataset {self.target_dataset_uuid} convert status is not completed.")
                    return None
                # 新增：关键拦截逻辑——如果当前正在处理，直接返回None，不重复分配
                if item.qc_status == TaskStatus.PROCESSING:
                    self.logger.info(f"Dataset {self.target_dataset_uuid} is already being processed, skip duplicate assignment.")
                    return None
                # 强制更新任务状态为PROCESSING
                item.qc_status = TaskStatus.PROCESSING
                item.qc_version = item.qc_version + 1
                session.commit()
                dataset_uuid = item.dataset_uuid
                repo_path = item.convert_path
                device_model = item.device_model
                device_model_version = item.device_model_version
            else:
                # 原有逻辑：自动筛选（保留，方便后续恢复）
                _sync_quality_check_tasks(session=session)
                dataset_uuid, repo_path, device_model, device_model_version = (
                    _gen_one_dataset_quality_check_task_without_sync(session=session)
                )

        if not dataset_uuid:
            self.logger.info("No dataset quality check task available.")
            return None

        checker_config = get_checker_config(
            device_model,
            device_model_version,
            self.qc_config_path,
        )
        return {
            DATASET_UUID: dataset_uuid,
            LEFORMAT_PATH: repo_path,
            DEVICE_MODEL: device_model,
            DEVICE_MODEL_VERSION: device_model_version,
            QC_CONFIG: checker_config,
        }

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        ds_uuid = task_content.get(DATASET_UUID)
        if not ds_uuid:
            self.logger.error("Handle task result failed: missing DATASET_UUID in task content.")
            return

        try:
            task_status = task_result_content.get(TASK_RESULT_STATUS)
            err_msg = task_result_content.get(ERR_MSG, "")
            task_content_dict = task_result_content.get(TASK_RESULT_CONTENT, {})
            qc_results = task_content_dict.get(QC_RESULT, {})

            with self.db.with_session() as session:
                item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
                if not item:
                    self.logger.error(f"Handle task result failed: dataset {ds_uuid} not found in DB.")
                    return

                if task_status == TASK_SUCCESS:
                    item.qc_status = TaskStatus.COMPLETED
                    item.qc_err_msg = ""
                    session.query(EpisodeQcDB).filter(EpisodeQcDB.dataset_uuid == ds_uuid).delete()

                    for episode_idx_str, summary in qc_results.items():
                        try:
                            episode_idx = int(episode_idx_str)
                        except (ValueError, TypeError):
                            self.logger.warning(f"Skip invalid episode index: {episode_idx_str}")
                            continue

                        # 提取所有字段（汇总+单独算子）
                        is_bad = summary.get("is_bad", False)
                        is_diff = summary.get("is_diff", False)
                        state_score = summary.get("state_data_score", 0.0)
                        action_score = summary.get("action_data_score", 0.0)
                        video_score = summary.get("video_score", 0.0)
                        # Episode Data 算子单独得分
                        state_static_frame_rate_score = summary.get("episode_state_static_frame_rate_score", 0.0)
                        state_static_joint_score = summary.get("episode_state_static_joint_score", 0.0)
                        action_static_frame_rate_score = summary.get("episode_action_static_frame_rate_score", 0.0)
                        action_static_joint_score = summary.get("episode_action_static_joint_score", 0.0)
                        # Episode Video 算子单独得分
                        max_frame_stable_then_jump_rate_score = summary.get("episode_video_max_frame_stable_then_jump_rate_score", 0.0)
                        max_frame_jump_dist_score = summary.get("episode_video_max_frame_jump_dist_score", 0.0)
                        episode_video_color_shift_detection_score = summary.get("episode_video_color_shift_detection_score", 0.0)
                        consecutive_static_frames_score = summary.get("episode_video_consecutive_static_frames_score", 0.0)
                        camera_resolution_consistency_score = summary.get("episode_video_camera_resolution_consistency_score", 0.0)
                        max_start = summary.get("max_start", 0)
                        max_end =  summary.get("max_end", 0)

                        episode_qc_item = EpisodeQcDB(
                            dataset_uuid=ds_uuid,
                            episode_idx=episode_idx,
                            is_bad_episode=is_bad,
                            is_state_frame_diff=is_diff,
                            # 原有分组汇总得分
                            state_data_score=state_score,
                            action_data_score=action_score,
                            video_score=video_score,
                            # Episode Data 算子单独得分
                            episode_state_static_frame_rate_score=state_static_frame_rate_score,
                            episode_state_static_joint_score=state_static_joint_score,
                            episode_action_static_frame_rate_score=action_static_frame_rate_score,
                            episode_action_static_joint_score=action_static_joint_score,
                            # Episode Video 算子单独得分
                            episode_video_max_frame_stable_then_jump_rate_score=max_frame_stable_then_jump_rate_score,
                            episode_video_max_frame_jump_dist_score=max_frame_jump_dist_score,
                            episode_video_color_shift_detection_score=episode_video_color_shift_detection_score,
                            episode_video_consecutive_static_frames_score=consecutive_static_frames_score,
                            episode_video_camera_resolution_consistency_score=camera_resolution_consistency_score,
                            start_frame = max_start,
                            end_frame = max_end,
                        )
                        session.add(episode_qc_item)

                    self.logger.info(f"Dataset {ds_uuid} quality check completed successfully.")
                else:
                    item.qc_status = TaskStatus.FAILED
                    item.qc_err_msg = err_msg[:1000]
                    self.logger.error(f"Dataset {ds_uuid} quality check failed: {err_msg}")

                session.commit()

        except Exception as e:
            self.logger.error(f"Handle task result for {ds_uuid} failed with exception: {str(e)}\n{traceback.format_exc()}")


class DatasetQualityCheckClient(TaskClient):
    def __init__(
        self,
        server_uri: str = "ws://localhost:2010",
        heartbeat_interval: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            logger=logger,
        )

    def get_task_category(self) -> str:
        return "dataset quality check"

    def generate_task_request_desc(self) -> dict:
        """客户端可自定义任务请求参数"""
        return {}

    def _sync_process_task(self, task_content: dict) -> dict:
        try:
            repo_path = task_content.get(LEFORMAT_PATH)
            if not repo_path:
                raise ValueError("Missing LEFORMAT_PATH in task content")

            results = _check_repo(
                repo_path,
                task_content.get(QC_CONFIG),
                data_feature=MERGED_DATA_FEATURE,
            )

            # 新增：打印关键日志，确认results是否为空（核心验证）
            print(f"[Client] _check_repo返回结果长度: {len(results)}")
            print(f"[Client] _check_repo返回结果前1条: {list(results.items())[:1] if results else '空'}")

            # 转换为字符串键（避免服务端解析问题），并转为纯原生字典（避免defaultdict序列化问题）
            results_send = {str(episode_idx): dict(v) for episode_idx, v in results.items()}

            # 新增：打印转换后的results_send
            print(f"[Client] 转换后results_send长度: {len(results_send)}")

            # 修复：确保这是唯一的返回语句，无多余缩进和不可达代码
            # return {
            #     TASK_RESULT_STATUS: TASK_SUCCESS,  # 标记任务成功
            #     TASK_RESULT_CONTENT: {QC_RESULT: results_send},  # 包装QC结果
            #     ERR_MSG: ""  # 清空错误信息
            # }
            return {QC_RESULT: results_send}

        except Exception as e:
            # 优化：捕获异常并返回标准错误格式，方便服务端记录
            error_msg = f"Dataset quality check for {repo_path} failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                TASK_RESULT_STATUS: "FAILED",
                TASK_RESULT_CONTENT: {},
                ERR_MSG: error_msg[:1000]
            }