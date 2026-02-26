import json
import logging
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tqdm
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import and_, or_

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    DatasetDB,
    DatasetHardLinkDB,
    EpisodeQcDB,
    TaskStatus,
)
from robocoin_dataset.distribution_computation.constant import (
    DATASET_UUID,
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
from robocoin_dataset.quality_check.hardlink.make_hardlink import (
    RepoHardLinkCorresp,
    create_hardlinks_from_correspondence,
)
from robocoin_dataset.utils.le_path import (
    get_episode_num,
    get_episodes_jsonl_file,
    get_episodes_stats_jsonl_file,
    get_meta_info_file,
    get_parquet_files,
    get_tasks_jsonl_file,
    get_video_files,
)

# from robocoin_dataset.utils.parquet_paths import get_parquet_paths

BAD_EPISODES = "bad_episodes"
HARD_LINK_PATH = "hard_link_path"
MIN_EPISODES_NUM = "min_episodes_num"

MERGED_FEATURE = None
QCED_FEATURE = "quality_checked"
HL_SUFFIX = "qced_hardlink"
DS_API_KEY = "ds_api_key"


def gen_qced_repo_files(
    repo_path: str | Path,
    bad_episodes: set[int],
    input_feature: str,
    output_feature: str,
) -> dict[str | Path, str | Path]:
    """Remove bad episodes from the dataset repository.

    Args:
        repo_path (str|Path): Path to the dataset repository.
        bad_episodes (set[int]): Set of episode indices to be removed.
        feature_type (str): Type of feature (e.g., 'state', 'action', 'video').
    """
    repo_path = Path(repo_path)
    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")

    if repo_path.is_file():
        raise ValueError(f"Repository path is a file: {repo_path}")

    if input_feature == output_feature:
        raise ValueError("Input feature cannot be the same as output feature.")

    input_info_file_path = get_meta_info_file(repo_path, input_feature)
    out_info_file_path = get_meta_info_file(repo_path, output_feature)

    input_episodes_jsonl_path = get_episodes_jsonl_file(repo_path)
    out_episodes_jsonl_path = get_episodes_jsonl_file(repo_path, output_feature)

    input_episodes_stats_jsonl_path = get_episodes_stats_jsonl_file(repo_path, input_feature)
    out_episodes_stats_jsonl_path = get_episodes_stats_jsonl_file(repo_path, output_feature)

    input_parquet_paths = get_parquet_files(
        root_dir=repo_path, feature=input_feature, meta_feature=input_feature
    )

    episodes_frame_nums = {}
    with open(input_episodes_jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            episode_id = data["episode_index"]
            frame_count = data["length"]
            episodes_frame_nums[episode_id] = frame_count

    sorted_episodes_frame_nums = sorted(episodes_frame_nums.items(), key=lambda x: x[0])
    episodes_frame_nums_list = [frame_num for _, frame_num in sorted_episodes_frame_nums]

    qc_episodes_start_frame_indices = [0]
    for episode_id in range(len(episodes_frame_nums_list)):
        if episode_id in bad_episodes:
            continue
        qc_episodes_start_frame_indices.append(
            qc_episodes_start_frame_indices[-1] + episodes_frame_nums_list[episode_id]
        )

    out_total_frames = sum(
        frame_num
        for ep_idx, frame_num in enumerate(episodes_frame_nums_list)
        if ep_idx not in bad_episodes
    )
    out_total_episodes = len(episodes_frame_nums_list) - len(bad_episodes)

    _gen_output_meta_info_file(
        input_info_file_path, out_info_file_path, out_total_frames, out_total_episodes
    )

    ep_num = get_episode_num(repo_path)
    _gen_output_episodes_jsonl_file(
        input_episodes_jsonl_path, out_episodes_jsonl_path, bad_episodes, ep_num=ep_num
    )
    _gen_output_episodes_stats_jsonl_file(
        input_episodes_stats_jsonl_path, out_episodes_stats_jsonl_path, bad_episodes, ep_num=ep_num
    )

    with open(input_info_file_path) as f:
        data = json.load(f)
        chunks_size = data.get("chunks_size")

    _gen_output_parquet_files(
        repo_path=repo_path,
        input_parquet_paths=input_parquet_paths,
        bad_episodes=bad_episodes,
        qc_episodes_start_frame_indices=qc_episodes_start_frame_indices,
        chunk_size=chunks_size,
        output_feature=output_feature,
    )
    input_video_files = get_video_files(repo_path)
    return _gen_video_path_matching_dict(
        input_video_files=input_video_files,
        repo_path=repo_path,
        bad_episodes=bad_episodes,
        chunk_size=chunks_size,
    )


def _gen_output_meta_info_file(
    input_info_file_path: Path,
    output_info_file_path: Path,
    total_frames: int,
    total_episodes: int,
) -> None:
    with open(input_info_file_path) as f:
        with open(output_info_file_path, "w") as out_f:
            data = json.load(f)
            input_total_videos = data.get("total_videos")
            input_total_episodes = data.get("total_episodes")
            chunks_size = data.get("chunks_size")
            videos_per_episode = input_total_videos // input_total_episodes
            output_train_split = f"0:{total_episodes - 1}"

            data["total_frames"] = total_frames
            data["total_episodes"] = total_episodes
            data["total_videos"] = total_episodes * videos_per_episode
            data["total_chunks"] = (total_episodes + chunks_size - 1) // chunks_size
            data["splits"]["train"] = output_train_split
            json.dump(data, out_f, indent=4, ensure_ascii=False)



def _gen_output_episodes_jsonl_file(
    input_episodes_jsonl_path: Path,
    output_episodes_jsonl_path: Path,
    bad_episodes: set[int],
    ep_num: int,
) -> None:
    out_ep_idx = 0
    with open(
        input_episodes_jsonl_path,
    ) as f:
        with open(output_episodes_jsonl_path, "w") as out_f:
            for ep_idx, line in enumerate(f):
                if ep_idx >= ep_num:
                    break
                data = json.loads(line)
                episode_id = data["episode_index"]
                if episode_id in bad_episodes:
                    continue
                data["episode_index"] = out_ep_idx
                out_f.write(json.dumps(data) + "\n")
                out_ep_idx += 1


def _gen_output_episodes_stats_jsonl_file(
    input_episodes_stats_jsonl_path: Path,
    output_episodes_stats_jsonl_path: Path,
    bad_episodes: set[int],
    ep_num: int,
) -> None:
    out_ep_idx = 0
    with open(
        input_episodes_stats_jsonl_path,
    ) as f:
        with open(output_episodes_stats_jsonl_path, "w") as out_f:
            for ep_idx, line in enumerate(f):
                if ep_idx >= ep_num:
                    break
                data = json.loads(line)
                episode_id = data["episode_index"]
                if episode_id in bad_episodes:
                    continue
                data["episode_index"] = out_ep_idx
                out_f.write(json.dumps(data) + "\n")
                out_ep_idx += 1


def _gen_output_parquet_files(
    repo_path: str | Path,
    input_parquet_paths: list[Path],
    bad_episodes: set[int],
    qc_episodes_start_frame_indices: list[int],
    chunk_size: int,
    output_feature: str,
) -> None:
    out_episode_idx = 0
    repo_path = Path(repo_path).expanduser().absolute()

    def get_output_parquet_path(out_ep_idx: int) -> Path:
        chunk_idx = out_ep_idx // chunk_size
        output_parquet_path = (
            repo_path
            / f"{output_feature}_data"
            / f"chunk-{chunk_idx:03d}"
            / f"episode_{out_ep_idx:06d}.parquet"
        )
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        return output_parquet_path

    for ep_idx, input_parquet_path in tqdm.tqdm(
        enumerate(input_parquet_paths),
        total=len(input_parquet_paths),
        desc="Generating output parquet files",
        unit="episode",
    ):
        if ep_idx in bad_episodes:
            continue
        df = pd.read_parquet(input_parquet_path)
        indices_data = np.array(df["index"].to_list(), dtype=int)
        indices_data = (
            indices_data - indices_data[0] + qc_episodes_start_frame_indices[out_episode_idx]
        )
        df["index"] = indices_data.tolist()
        df["episode_index"] = out_episode_idx
        output_parquet_path = get_output_parquet_path(out_episode_idx)
        df.to_parquet(output_parquet_path, engine="pyarrow")
        out_episode_idx += 1


def _gen_video_path_matching_dict(
    input_video_files: list[list[Path]],
    repo_path: str | Path,
    bad_episodes: set[int],
    chunk_size: int,
) -> dict[int, list[Path]]:
    repo_path = Path(repo_path).expanduser().absolute()

    def get_video_new_path(ep_video_paths: list[Path], output_ep_idx: int) -> dict[str, str]:
        chunk_idx = output_ep_idx // chunk_size
        results = {}
        for video_path in ep_video_paths:
            new_video_path = (
                repo_path
                / "videos"
                / f"chunk-{chunk_idx:03d}"
                / video_path.parent.name
                / f"episode_{output_ep_idx:06d}.mp4"
            )
            results[str(video_path)] = str(new_video_path)
        return results

    matching_dict = {}

    out_episode_idx = 0
    for ep_idx, video_paths in enumerate(input_video_files):
        if ep_idx in bad_episodes:
            continue

        matching_dict.update(get_video_new_path(video_paths, out_episode_idx))
        out_episode_idx += 1

    return matching_dict


def _needs_hand_normalization(text: str) -> bool:
    """检测是否包含左右手的限定"""
    lowered = text.lower()
    hand_keywords = [
        "left hand",
        "right hand",
        "left arm",
        "right arm",
        "left side",
        "right side",
        "left glove",
        "right glove",
    ]
    if any(kw in lowered for kw in hand_keywords):
        return True
    # 简单处理“左/右”中文描述
    if re.search(r"[左右]手|[左右]臂|[左右]边", text):
        return True
    return False


def normalize_task_hand_instructions(
    src_jsonl_path: str | Path, target_jsonl_path: str | Path, ds_api_key: str
) -> None:
    """调用LLM将任务指令中的左右手限定改成无左右区分的描述，并写回JSONL"""
    src_jsonl_path = Path(src_jsonl_path).expanduser().absolute()
    if not src_jsonl_path.exists():
        raise FileNotFoundError(f"任务文件不存在：{src_jsonl_path}")

    lines = src_jsonl_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return

    records: list[dict] = []
    candidate_tasks: list[str] = []
    candidate_indices: list[int] = []

    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {idx + 1} 行不是合法 JSON：{exc}") from exc
        records.append(record)
        task_text = record.get("task", "")
        if isinstance(task_text, str) and _needs_hand_normalization(task_text):
            candidate_tasks.append(task_text)
            candidate_indices.append(idx)

    if not candidate_tasks:
        with open(target_jsonl_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        return

    prompt_parts = [
        "Please read the task descriptions below and rewrite EACH one into a single natural English sentence "
        "that does NOT distinguish between left/right hand/arm/side while preserving the original meaning."
        "Do NOT include any left/right wording in the rewritten sentence. The output MUST be in English."
    ]
    task_block = "\n".join(f"{i + 1}. {task}" for i, task in enumerate(candidate_tasks))
    prompt_parts.append(task_block)
    prompt_parts.append(
        "Output the rewritten sentences in order, one per line, without numbering, and in English only."
    )
    prompt = "\n\n".join(prompt_parts)

    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ds_api_key}"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": 1024,
        "stream": False,
    }

    try:
        print("Trying to call DeepSeek API to normalize task hand instructions")
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=100)
        if response.status_code != 200:
            raise RuntimeError(f"API请求失败: {response.status_code} - {response.text}")

        response_data = response.json()
        processed_text = response_data["choices"][0]["message"]["content"].strip()
        result_lines = [line.strip() for line in processed_text.splitlines() if line.strip()]
        cleaned_results: list[str] = []
        for line in result_lines:
            if ". " in line and line.split(". ")[0].isdigit():
                cleaned_results.append(line.split(". ", 1)[-1])
            else:
                cleaned_results.append(line)

        if len(cleaned_results) != len(candidate_tasks):
            raise ValueError(
                f"改写结果数量({len(cleaned_results)})与待处理任务数量({len(candidate_tasks)})不匹配"
            )

        for idx, new_task in zip(candidate_indices, cleaned_results):
            record = records[idx]
            record["task"] = new_task

        with target_jsonl_path.open("w", encoding="utf-8") as fp:
            for record in records:
                fp.write(json.dumps(record, ensure_ascii=False))
                fp.write("\n")

    except Exception as exc:
        raise RuntimeError(f"调用AI接口改写任务描述失败: {exc}") from exc


def _gen_optimized_tasks_jsonl(
    src_path: str | Path, target_path: str | Path, ds_api_key: str | None = None
) -> None:
    normalize_task_hand_instructions(
        src_jsonl_path=src_path, target_jsonl_path=target_path, ds_api_key=ds_api_key
    )
    return


def gen_qced_repo(
    repo_path: str | Path,
    bad_episodes: set[int],
    input_feature: str = MERGED_FEATURE,
    qced_feature: str = QCED_FEATURE,
    hl_suffix: str = HL_SUFFIX,
    min_episodes_num: int = 10,
    ds_api_key: str | None = None,
) -> str:
    repo_path = Path(repo_path).expanduser().absolute()

    episodes_num = get_episode_num(repo_path)
    if (episodes_num - len(bad_episodes)) < min_episodes_num:
        raise ValueError(
            f"Dataset {repo_path}: The number of episodes after removing bad episodes is less than {min_episodes_num}, original episodes num: {episodes_num}"
        )

    src_tasks_jsonl_path = get_tasks_jsonl_file(repo_path)
    target_tasks_jsonl_path = get_tasks_jsonl_file(repo_path, feature=qced_feature)
    _gen_optimized_tasks_jsonl(
        src_path=src_tasks_jsonl_path, target_path=target_tasks_jsonl_path, ds_api_key=ds_api_key
    )

    video_path_corresp = gen_qced_repo_files(
        repo_path=repo_path,
        bad_episodes=bad_episodes,
        input_feature=input_feature,
        output_feature=qced_feature,
    )
    hard_link_repo_path = repo_path.parent / f"{str(repo_path.name)}_{hl_suffix}"
    file_corresp, dir_corresp = RepoHardLinkCorresp(
        input_feature=qced_feature,
        source_repo_path=repo_path,
        hard_link_repo_path=hard_link_repo_path,
        video_path_corresp=video_path_corresp,
    ).get_hard_link_corresp()

    dir_corresp_filtered = {}
    for src_dir, dst_dir in dir_corresp.items():
        if "annotations" not in str(src_dir).lower() and "annotations" not in str(dst_dir).lower():
            dir_corresp_filtered[src_dir] = dst_dir
    dir_corresp = dir_corresp_filtered

    # print("=" * 80)
    # print("即将创建硬链接的文件清单（共 {} 个文件）：".format(len(file_corresp)))
    # print("-" * 80)
    # for idx, (src_file, dst_file) in enumerate(file_corresp.items(), 1):
    #     print(f"{idx}. 源文件：{src_file}")
    #     print(f"   目标硬链接：{dst_file}")
    #     print("-" * 40)

    # print("\n" + "=" * 80)
    # print("即将创建硬链接的目录清单（共 {} 个目录）：".format(len(dir_corresp)))
    # print("-" * 80)
    # for idx, (src_dir, dst_dir) in enumerate(dir_corresp.items(), 1):
    #     print(f"{idx}. 源目录：{src_dir}")
    #     print(f"   目标硬链接目录：{dst_dir}")
    #     print("-" * 40)
    # print("=" * 80)

    create_hardlinks_from_correspondence(file_corresp=file_corresp, dir_corresp=dir_corresp)
    return str(hard_link_repo_path)


def _sync_qced_repo_gen_tasks(session: Session) -> None:
    query = session.query(DatasetDB).filter(
        and_(
            # 必要前提：convert必须成功
            DatasetDB.qc_status == TaskStatus.COMPLETED,
            # 两个触发分支
            or_(
                # 分支1: 正在排队
                DatasetDB.qced_repo_gen_status == TaskStatus.PENDING,
                # 分支2: 已完成但版本过期
                and_(
                    DatasetDB.qced_repo_gen_status == TaskStatus.COMPLETED,
                    DatasetDB.qced_repo_gen_version_ps < DatasetDB.qc_version,
                ),
            ),
        )
    )
    items = query.all()

    if not items:
        return

    for item in items:
        item.qced_repo_gen_status = TaskStatus.PENDING
        item.qced_repo_gen_version_ps = item.qc_version

    session.commit()


def _gen_one_qced_repo_gen_task(
    session: Session,
) -> tuple[str | None, str | None, str | None, str | None]:
    query = session.query(DatasetDB).filter(
        and_(
            # 必要前提：convert必须成功
            DatasetDB.qc_status == TaskStatus.COMPLETED,
            DatasetDB.qced_repo_gen_status == TaskStatus.PENDING,
        )
    )
    item = query.first()

    if not item:
        return None, None

    item.qced_repo_gen_status = TaskStatus.PROCESSING

    item.qced_repo_gen_version = item.qced_repo_gen_version + 1
    session.commit()

    return item.dataset_uuid, item.convert_path


def _get_bad_episodes(
    session: Session,
    dataset_uuid: str,
    state_data_score_threshold: float = 0.75,
    action_data_score_threshold: float = 0.75,
    video_score: float = 0.87,
    consecutive_static_frames_threshold: float = 0.6,
) -> set[int]:
    items = (
        session.query(EpisodeQcDB)
        .filter(
            EpisodeQcDB.dataset_uuid == dataset_uuid,
        )
        .all()
    )
    if not items:
        return set()

    bad_episodes = set()
    for item in items:
        if item.is_bad_episode:
            bad_episodes.add(item.episode_idx)
            continue
        if item.state_data_score < state_data_score_threshold:
            bad_episodes.add(item.episode_idx)
            continue
        if item.action_data_score < action_data_score_threshold:
            bad_episodes.add(item.episode_idx)
            continue
        if item.video_score < video_score:
            bad_episodes.add(item.episode_idx)
            continue
        if item.episode_video_consecutive_static_frames_score < consecutive_static_frames_threshold:
            bad_episodes.add(item.episode_idx)
            continue
        if item.is_state_frame_diff:
            bad_episodes.add(item.episode_idx)

    return bad_episodes


class QualityCheckedRepoGenerator:
    def __init__(
        self,
        db_file_path: str | Path,
        state_data_score_threshold: float = 0.85,
        action_data_score_threshold: float = 0.85,
        video_score_threshold: float = 0.9,
        min_episodes_num: int = 10,
        ds_api_key: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.logger = logger or logging.getLogger(__name__)
        self.state_data_score_threshold = state_data_score_threshold
        self.action_data_score_threshold = action_data_score_threshold
        self.video_score_threshold = video_score_threshold
        self.min_episodes_num = min_episodes_num
        self.ds_api_key = ds_api_key

    def gen_one_qced_repo(self) -> None:
        with self.db.with_session() as session:
            _sync_qced_repo_gen_tasks(session=session)
            dataset_uuid, repo_path = _gen_one_qced_repo_gen_task(session=session)

            bad_episodes = _get_bad_episodes(
                session=session,
                dataset_uuid=dataset_uuid,
                state_data_score_threshold=self.state_data_score_threshold,
                action_data_score_threshold=self.action_data_score_threshold,
                video_score=self.video_score_threshold,
            )

        if not dataset_uuid:
            return

        try:
            hardlink_repo_path = gen_qced_repo(
                repo_path=repo_path,
                bad_episodes=bad_episodes,
                min_episodes_num=self.min_episodes_num,
                ds_api_key=self.ds_api_key,
            )
            with self.db.with_session() as session:
                ds_item = (
                    session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                )
                if not ds_item:
                    return
                ds_item.qced_repo_gen_status = TaskStatus.COMPLETED
                ds_item.qced_repo_gen_path = hardlink_repo_path  # 新增这一行，硬链接路径直接赋值
                hardlink_item = (
                    session.query(DatasetHardLinkDB)
                    .filter(DatasetHardLinkDB.dataset_uuid == dataset_uuid)
                    .first()
                )
                if hardlink_item:
                    hardlink_item.hard_link_path = hardlink_repo_path
                else:
                    session.add(
                        DatasetHardLinkDB(
                            dataset_uuid=dataset_uuid,
                            hard_link_path=hardlink_repo_path,
                        )
                    )
                session.commit()
        except Exception:
            with self.db.with_session() as session:
                ds_item = (
                    session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                )
                if ds_item:
                    ds_item.qced_repo_gen_status = TaskStatus.FAILED
                    ds_item.qced_repo_gen_err_msg = traceback.format_exc()
                else:
                    return
                session.commit()
            self.logger.info(traceback.format_exc())


class QualityCheckedRepoGeneratorServer(TaskServer):
    def __init__(
        self,
        db_file_path: str | Path,
        host: str = "0.0.0.0",
        port: int = 2010,
        heartbeat_interval: float = 30.0,  # 服务端每30秒发一次 ping
        timeout: float = 15.0,  # 等待 pong 超过15秒则断开
        logger: logging.Logger | None = None,
        qc_config: dict | None = None,
        ds_api_key: str | None = None,
        target_dataset_uuid: str | None = None,  # 新增：支持指定UUID
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

        self.state_data_score_threshold = qc_config.get("state_data_score_threshold", 0.75)
        self.action_data_score_threshold = qc_config.get("action_data_score_threshold", 0.75)
        self.video_score_threshold = qc_config.get("video_score_threshold", 0.87)
        self.min_episodes_num = qc_config.get("min_episodes_num", 30)
        self.consecutive_static_frames_threshold = qc_config.get("consecutive_static_frames_threshold", 0.6)
        self.ds_api_key = ds_api_key
        self.target_dataset_uuid = target_dataset_uuid  # 新增：保存指定UUID

    def get_task_category(self) -> str:
        return "dataset quality checked repo generation"

    def generate_task_content(self) -> dict | None:
        with self.db.with_session() as session:
            dataset_uuid = None
            repo_path = None
            bad_episodes = set()

            # 新增：优先处理指定UUID的数据集
            if self.target_dataset_uuid:
                item = session.query(DatasetDB).filter(
                    DatasetDB.dataset_uuid == self.target_dataset_uuid
                ).first()
                if not item:
                    self.logger.info(f"Dataset with UUID {self.target_dataset_uuid} not found.")
                    return None
                # 校验前置状态：QC必须已完成
                if item.qc_status != TaskStatus.COMPLETED:
                    self.logger.info(f"Dataset {self.target_dataset_uuid} QC status is not COMPLETED, skip.")
                    return None
                # 校验是否正在处理/已完成（避免重复）
                if item.qced_repo_gen_status == TaskStatus.PROCESSING:
                    self.logger.info(f"Dataset {self.target_dataset_uuid} is already being processed, skip.")
                    return None
                if item.qced_repo_gen_status == TaskStatus.COMPLETED:
                    self.logger.info(f"Dataset {self.target_dataset_uuid} has already been processed successfully, skip.")
                    return None
                # 更新为处理中状态
                item.qced_repo_gen_status = TaskStatus.PROCESSING
                item.qced_repo_gen_version = item.qced_repo_gen_version + 1
                session.commit()

                # 提取必要信息
                dataset_uuid = item.dataset_uuid
                repo_path = item.convert_path
                # 获取坏样本集
                bad_episodes = _get_bad_episodes(
                    session=session,
                    dataset_uuid=dataset_uuid,
                    state_data_score_threshold=self.state_data_score_threshold,
                    action_data_score_threshold=self.action_data_score_threshold,
                    video_score=self.video_score_threshold,
                    consecutive_static_frames_threshold=self.consecutive_static_frames_threshold,
                )
            else:
                # 原有逻辑：自动筛选待处理任务
                _sync_qced_repo_gen_tasks(session=session)
                dataset_uuid, repo_path = _gen_one_qced_repo_gen_task(session=session)
                if dataset_uuid:
                    bad_episodes = _get_bad_episodes(
                        session=session,
                        dataset_uuid=dataset_uuid,
                        state_data_score_threshold=self.state_data_score_threshold,
                        action_data_score_threshold=self.action_data_score_threshold,
                        video_score=self.video_score_threshold,
                        consecutive_static_frames_threshold=self.consecutive_static_frames_threshold,
                    )

            if not dataset_uuid:
                return None
            bad_episodes = list(bad_episodes)

        return {
            DATASET_UUID: dataset_uuid,
            LEFORMAT_PATH: repo_path,
            BAD_EPISODES: bad_episodes,
            MIN_EPISODES_NUM: self.min_episodes_num,
            DS_API_KEY: self.ds_api_key,
        }


    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        ds_uuid = task_content.get(DATASET_UUID)
        if not ds_uuid:  # 新增：校验UUID是否存在，避免后续报错
            self.logger.error("Task content missing DATASET_UUID, skip handling result.")
            return

        # 修复：正确提取任务状态和错误信息
        task_status = task_result_content.get(TASK_RESULT_STATUS)
        err_msg = task_result_content.get(ERR_MSG, "")  # 避免key不存在抛出异常

        try:
            # 修复：安全提取hardlink_path，避免嵌套key不存在抛出异常
            task_result_content_dict = task_result_content.get(TASK_RESULT_CONTENT, {})
            hardlink_path = task_result_content_dict.get(HARD_LINK_PATH, "")
        except Exception as e:
            hardlink_path = ""
            self.logger.warning(f"Failed to extract hardlink path from task result: {str(e)}")

        if task_status == TASK_SUCCESS and hardlink_path:  # 新增：校验hardlink_path有效性
            with self.db.with_session() as session:
                item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
                if not item:
                    self.logger.error(f"Dataset with UUID {ds_uuid} not found in DB, skip updating status.")
                    return
                item.qced_repo_gen_status = TaskStatus.COMPLETED
                item.qced_repo_gen_path = hardlink_path  # 新增这一行，覆盖原有路径（如有）
                hl_item = session.query(DatasetHardLinkDB).filter(DatasetHardLinkDB.dataset_uuid == ds_uuid).first()
                if hl_item:
                    hl_item.hard_link_path = hardlink_path
                else:
                    session.add(DatasetHardLinkDB(dataset_uuid=ds_uuid, hard_link_path=hardlink_path))
                session.commit()
            self.logger.info(f"Dataset {ds_uuid} qced repo generation completed successfully.")
        else:
            # 修复：任务失败时，正确更新状态为FAILED并保存错误信息
            with self.db.with_session() as session:
                item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
                if not item:
                    self.logger.error(f"Dataset with UUID {ds_uuid} not found in DB, skip updating status.")
                    return
                item.qced_repo_gen_status = TaskStatus.FAILED
                item.qced_repo_gen_err_msg = err_msg[:1000]  # 限制错误信息长度，避免数据库字段溢出
                session.commit()
            self.logger.error(
                f"Dataset {ds_uuid} qced repo generation failed. Error: {err_msg[:500]}..."
            )



class QualityCheckedRepoGeneratorClient(TaskClient):
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
        return "dataset quality checked repo generation"

    def generate_task_request_desc(self) -> dict:
        """客户端可自定义任务请求参数"""
        return {}

    def _sync_process_task(self, task_content: dict) -> dict:
        try:
            repo_path = task_content.get(LEFORMAT_PATH)
            bad_episodes = task_content.get(BAD_EPISODES)
            bad_episodes = set(bad_episodes)
            min_episodes_num = task_content.get(MIN_EPISODES_NUM)
            ds_api_key = task_content.get(DS_API_KEY)

            hardlink_repo_path = gen_qced_repo(
                repo_path=repo_path,
                bad_episodes=bad_episodes,
                min_episodes_num=min_episodes_num,
                ds_api_key=ds_api_key,
            )

            return {HARD_LINK_PATH: hardlink_repo_path}
        except Exception as e:
            raise RuntimeError(f"dataset quality checked repo generation {repo_path} failed") from e
