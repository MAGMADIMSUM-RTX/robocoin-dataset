import json
import logging
import traceback
import shutil  # 新增：用于删除文件夹
from pathlib import Path

import pandas as pd
import tqdm
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import and_, or_

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    DatasetDB,
    TaskStatus,
)
from robocoin_dataset.distribution_computation.constant import (
    DATASET_UUID,
    ERR_MSG,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
)
from robocoin_dataset.distribution_computation.task_client import TaskClient
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.format_converter.tolerobot.constant import (
    LEFORMAT_PATH,
)

from robocoin_dataset.utils.le_path import (
    get_episode_num,
    get_episodes_stats_jsonl_file,
    get_meta_info_file,
    get_parquet_files,
)


def _get_uuid_list_from_string(uuids_str: str) -> list[str]:
    return [uuid_str.strip() for uuid_str in uuids_str.split(",") if uuid_str]


def _get_string_from_uuid_list(uuids: list[str]) -> str:
    return ",".join(uuids)


class DataMergeConfig:
    pre_stage_set: set[tuple[str, str, str]] = {
        (
            DatasetDB.video_embed_subtask_annotation_status,
            DatasetDB.video_embed_subtask_annotation_version,
            DatasetDB.data_merge_version_ps_sta,
        ),
        (
            DatasetDB.scene_annotation_status,
            DatasetDB.scene_annotation_version,
            DatasetDB.data_merge_version_ps_sa,
        ),
        (
            DatasetDB.motion_annotation_status,
            DatasetDB.motion_annotation_version,
            DatasetDB.data_merge_version_ps_ma,
        ),
    }
    patch_features = ["subtask_annotation", "scene_annotation", "motion_annotation", "state_action"]
    merge_feature = ''


def _merge_episode_parquet_files(
    ori_path: str | Path,
    patch_paths: list[str | Path],
    output_path: str | Path,
) -> None:
    # 1. 读取原始数据
    ori_df = pd.read_parquet(ori_path)

    # 2. 逐个应用 patch
    result_df = ori_df.copy()
    updated_columns = set()

    for i, p_path in enumerate(patch_paths, 1):
        p_path = Path(p_path)
        if not p_path.exists():
            continue

        patch_df = pd.read_parquet(p_path)
        if len(patch_df) != len(result_df):
            raise ValueError(
                f"文件行数不一致: {p_path.name} ({len(patch_df)} 行) vs 原始数据 ({len(result_df)} 行)"
            )

        # 找出 patch 中存在的数据列（排除主键类列，但这里我们只更新实际数据）
        for col in patch_df.columns:
            result_df[col] = patch_df[col].values  # 直接按位置赋值
            updated_columns.add(col)

    # 3. 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)


def _merge_parquet_files(
    ori_parquet_files: list[str | Path],
    patch_parquet_files: list[list[str | Path]],
    merged_parquet_files: list[str | Path],
) -> None:
    if len(ori_parquet_files) != len(merged_parquet_files):
        raise ValueError(
            f"The number of original parquet files ({len(ori_parquet_files)}) "
            f"does not match the number of merged parquet files ({len(merged_parquet_files)})."
        )
    for parquet_files in patch_parquet_files:
        if len(parquet_files) != len(merged_parquet_files):
            raise ValueError(
                f"parquet_files length {len(parquet_files)} != merge_parquet_files length {len(merged_parquet_files)}"
            )

    for ep_idx in tqdm.tqdm(
        range(len(merged_parquet_files)), desc="Merging Parquet Files", unit="episode"
    ):
        feature_patch_parquet_files = [
            parquet_files[ep_idx] for parquet_files in patch_parquet_files
        ]
        _merge_episode_parquet_files(
            ori_parquet_files[ep_idx], feature_patch_parquet_files, merged_parquet_files[ep_idx]
        )


def merge_dataset_parquet_files(
    root_dir: str | Path, patch_features: list[str], merge_feature: str = "merged"
) -> None:
    features_parquet_files = []
    ori_parquet_files = get_parquet_files(root_dir)
    merge_parquet_files = get_parquet_files(root_dir, merge_feature)
    for feature in patch_features:
        feature_parquet_files = get_parquet_files(root_dir, feature=feature)
        if not feature_parquet_files[0].exists():
            # raise ValueError(f"{feature_parquet_files[0]} file not found")
            continue
        features_parquet_files.append(feature_parquet_files)

    _merge_parquet_files(ori_parquet_files, features_parquet_files, merge_parquet_files)


def _deep_merge_dict(ori: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if key in ori:
            if isinstance(ori[key], dict) and isinstance(value, dict):
                # 递归合并字典
                _deep_merge_dict(ori[key], value)
            elif isinstance(ori[key], list) and isinstance(value, list):
                ori[key] = value
            else:
                # 基本类型或类型不同 → 直接替换
                ori[key] = value
        else:
            # 新增键
            ori[key] = value
    return ori


def _merge_info_files(
    ori_path: str | Path,
    patch_paths: list[str | Path],
) -> dict:
    with open(ori_path) as f:
        ori_info = json.load(f)
    for patch_path in patch_paths:
        with open(patch_path) as f:
            patch_info = json.load(f)
        ori_info = _deep_merge_dict(ori_info, patch_info)

    return ori_info


def _fill_dtype_and_shape(info: dict[str, dict], parquet_file_path: str | Path) -> dict:
    parquet_file_path = Path(parquet_file_path)
    df = pd.read_parquet(str(parquet_file_path))

    for feature_name in info["features"].keys():
        if feature_name in df.columns:
            first_value = df[feature_name].iloc[0]
            import numpy as np

            arr = np.array(first_value)
            if arr.shape == ():
                shape = [1]
            else:
                shape = list(arr.shape)

            info["features"][feature_name]["dtype"] = str(first_value.dtype)
            info["features"][feature_name]["shape"] = shape

    return info


def merge_dataset_info_files(
    root_dir: str | Path, patch_features: list[str], merged_feature: str = "merged"
) -> dict:
    root_dir = Path(root_dir).expanduser().absolute()
    ori_info_path = get_meta_info_file(root_dir)
    merged_info_path = get_meta_info_file(root_dir, merged_feature)
    merged_parquet_paths = get_parquet_files(root_dir, merged_feature)
    patch_info_paths = []
    for feature in patch_features:
        path = get_meta_info_file(root_dir, feature)
        if not path.exists():
            # raise FileNotFoundError(f"{path} not found")
            continue
        patch_info_paths.append(path)

    merged_info = _merge_info_files(ori_info_path, patch_info_paths)
    merged_info = _fill_dtype_and_shape(merged_info, merged_parquet_paths[0])

    with open(merged_info_path, "w") as f:
        json.dump(merged_info, f, indent=4, ensure_ascii=False, sort_keys=False)
    return merged_info


def _merge_jsonl_files(
    ori_file: str | Path, patch_files: list[str, Path], output_file: str | Path, ep_num: int
) -> None:
    with open(ori_file) as f:
        ori_jsonl = [json.loads(line) for line in f]
    if len(ori_jsonl) < ep_num:
        raise ValueError(f"ori_jsonl 长度不足: {len(ori_jsonl)} < {ep_num}")

    patch_jsonls = []
    for patch_file in patch_files:
        with open(patch_file) as f:
            patch_jsonl = [json.loads(line) for line in f]
            if len(patch_jsonl) < ep_num:
                raise ValueError(f"patch_jsonl 长度不足: {len(patch_jsonl)} < {ep_num}")
        patch_jsonls.append(patch_jsonl)

    new_jsonl = []
    for i in range(ep_num):
        ori_json = ori_jsonl[i]
        for patch_jsonl in patch_jsonls:
            patch_json = patch_jsonl[i]
            ori_json = _deep_merge_dict(ori_json, patch_json)
        new_jsonl.append(ori_json)

    with open(output_file, "w") as f:
        for json_obj in new_jsonl:
            json.dump(json_obj, f)
            f.write("\n")


def merge_dataset_stats_jsonl_files(
    root_dir: str | Path, patch_features: list[str], ep_num: int, merge_feature: str = "merged"
) -> None:
    ori_stats_file = get_episodes_stats_jsonl_file(root_dir)
    merged_stats_file = get_episodes_stats_jsonl_file(root_dir, merge_feature)
    patch_stats_files = []
    for patch_feature in patch_features:
        patch_stats_file = get_episodes_stats_jsonl_file(root_dir, patch_feature)
        if not patch_stats_file.exists():
            raise FileNotFoundError(f"{patch_stats_file} does not exist")
        patch_stats_files.append(patch_stats_file)

    _merge_jsonl_files(ori_stats_file, patch_stats_files, merged_stats_file, ep_num=ep_num)


def check_merged_info_file(
    root_dir: str | Path, 
    merge_feature: str = "merged",
    logger: logging.Logger | None = None
) -> tuple[bool, bool, str | None, dict | None, str | None]:
    """
    同时检查文件存在性和内容合规性（merged_info.json 不存在时 fallback 到 info.json）
    :param root_dir: 数据集根目录
    :param merge_feature: 合并特征名（用于定位 merged_info.json）
    :param logger: 日志对象
    :return: (文件是否存在, 内容是否合规, 错误信息, 读取的字典, 检查的文件类型: 'merged_info'/'info'/None)
    """
    logger = logger or logging.getLogger(__name__)
    root_dir = Path(root_dir).expanduser().absolute()
    
    # ========== 1. 先检查 merged_info.json ==========
    merged_info_path = get_meta_info_file(root_dir, "merged")

    if merged_info_path.exists():
        logger.info(f"检测到 merged_info.json 存在，执行合规性检查")
        exists, valid, err_msg, data = _check_info_file_compliance(merged_info_path, logger)
        return exists, valid, err_msg, data, "merged_info"  # 标识文件类型为 merged_info
    
    # ========== 2. merged_info.json 不存在，检查 info.json ==========
    info_path = get_meta_info_file(root_dir)  # 不带 feature，默认返回 info.json
    logger.warning(f"merged_info.json 不存在 ({merged_info_path})，fallback 检查 info.json: {info_path}")
    
    if not info_path.exists():
        err_msg = f"merged_info.json 和 info.json 均不存在（merged_info路径: {merged_info_path}, info路径: {info_path}）"
        logger.error(err_msg)
        return False, False, err_msg, None, None
    
    # 对 info.json 执行合规性检查
    logger.info(f"检测到 info.json 存在，执行合规性检查")
    exists, valid, err_msg, data = _check_info_file_compliance(info_path, logger, is_fallback=True)
    return exists, valid, err_msg, data, "info"  # 标识文件类型为 info


def _check_info_file_compliance(
    file_path: Path,
    logger: logging.Logger,
    is_fallback: bool = False
) -> tuple[bool, bool, str | None, dict | None]:
    """
    内部函数：执行单个 info 文件（merged_info.json/info.json）的合规性检查
    :param file_path: 待检查文件路径
    :param logger: 日志对象
    :param is_fallback: 是否是 fallback 到 info.json 的检查
    :return: (文件是否存在, 内容是否合规, 错误信息, 读取的字典)
    """
    file_type = "info.json" if is_fallback else "merged_info.json"
    
    # 1. 读取文件内容
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            info_data = json.load(f)
    except Exception as e:
        err_msg = f"{file_type} 存在但读取失败（路径: {file_path}）: {str(e)}"
        logger.warning(f"{file_type} 合规性不通过: {err_msg}")
        return True, False, err_msg, None

    # 2. 检查顶级字段
    required_top_fields = [
        "codebase_version", "robot_type", "total_episodes", "total_frames",
        "total_tasks", "total_videos", "total_chunks", "chunks_size", "fps",
        "splits", "data_path", "video_path", "features"
    ]
    missing_top_fields = [f for f in required_top_fields if f not in info_data]
    if missing_top_fields:
        err_msg = f"{file_type} 缺少顶级字段（路径: {file_path}）: {', '.join(missing_top_fields)}"
        logger.warning(f"{file_type} 合规性不通过: {err_msg}")
        return True, False, err_msg, info_data

    # 3. 检查 features 下字段
    required_feature_fields = [
        "observation.state", "action",
        "timestamp", "frame_index", "episode_index", "index", "task_index",
        "subtask_annotation", "scene_annotation", "eef_sim_pose_state",
        "eef_sim_pose_action", "eef_direction_state", "eef_direction_action",
        "eef_velocity_state", "eef_velocity_action", "eef_acc_mag_state",
        "eef_acc_mag_action", "gripper_open_scale_state", "gripper_open_scale_action",
        "gripper_mode_state", "gripper_mode_action", "gripper_activity_state",
        "gripper_activity_action"
    ]
    features = info_data.get("features", {})
    missing_feature_fields = [f for f in required_feature_fields if f not in features]
    if missing_feature_fields:
        err_msg = f"{file_type} 的 features 缺少字段（路径: {file_path}）: {', '.join(missing_feature_fields)}"
        logger.warning(f"{file_type} 合规性不通过: {err_msg}")
        return True, False, err_msg, info_data

    # 4. 合规性检查通过
    logger.info(f"{file_type} 存在且合规性检查通过（路径: {file_path}）")
    return True, True, None, info_data


def check_required_files_and_folders(
    root_dir: str | Path,
    logger: logging.Logger | None = None
) -> tuple[bool, str | None]:
    """
    前置检查：验证数据集根目录下核心文件夹、meta文件夹下指定文件是否存在
    （优先检查 backup 文件夹内的内容，无 backup 则检查根目录；backup 内跳过 annotations/videos 检查，转至上一级验证）
    :param root_dir: 数据集根目录
    :param logger: 日志对象（可选）
    :return: (检查是否通过, 错误信息/None)
    """
    logger = logger or logging.getLogger(__name__)
    root_dir = Path(root_dir).expanduser().absolute()
    
    # ========== 1. 优先确定检查基准目录（backup 优先） ==========
    backup_dir = root_dir / "backup"
    is_backup_base = False
    if backup_dir.exists() and backup_dir.is_dir():
        base_dir = backup_dir
        is_backup_base = True
        logger.info(f"检测到 backup 文件夹存在，优先检查 backup 内的内容: {base_dir}")
    else:
        base_dir = root_dir
        logger.info(f"未检测到 backup 文件夹，检查根目录内容: {base_dir}")
    
    # 基于基准目录定位 meta 文件夹
    meta_dir = base_dir / "meta"

    # ========== 2. 定义需要检查的文件/文件夹清单 ==========
    # meta文件夹下必须存在的文件列表（基准目录下）
    required_meta_files = [
        "episodes.jsonl",
        "scene_annotation_episodes_stats.jsonl",
        "episodes_stats.jsonl",
        "scene_annotation_info.json",
        "info.json",
        "state_action_episodes_stats.jsonl",
        "state_action_info.json",
        "subtask_annotation_episodes_stats.jsonl",
        "motion_annotation_episodes_stats.jsonl",
        "subtask_annotation_info.json",
        "motion_annotation_info.json",
        "tasks.jsonl"
    ]

    # 通用核心文件夹（所有场景都检查，基准目录下）
    core_folders = [
        "meta",
        "state_action_data",
        "motion_annotation_data",
        "subtask_annotation_data",
        "scene_annotation_data"
    ]
    # 仅根目录检查的文件夹（annotations、videos）
    root_only_folders = [
        "annotations",
        "videos"
    ]

    # ========== 3. 检查meta文件夹本身是否存在（基于基准目录） ==========
    if not meta_dir.exists() or not meta_dir.is_dir():
        err_msg = f"基准目录 {base_dir} 下 meta 文件夹不存在: {meta_dir}"
        logger.error(err_msg)
        return False, err_msg

    # ========== 4. 检查meta文件夹下的文件（基于基准目录） ==========
    missing_meta_files = []
    for file_name in required_meta_files:
        file_path = meta_dir / file_name
        if not file_path.exists() or not file_path.is_file():
            missing_meta_files.append(file_name)
    
    if missing_meta_files:
        err_msg = f"基准目录 {base_dir} 下 meta 文件夹缺失以下文件: {', '.join(missing_meta_files)}"
        logger.error(err_msg)
        return False, err_msg

    # ========== 5. 检查文件夹（分场景处理） ==========
    missing_folders = []
    
    # 5.1 检查通用核心文件夹（基准目录下）
    logger.info(f"检查基准目录 {base_dir} 下的通用核心文件夹")
    for folder_name in core_folders:
        folder_path = base_dir / folder_name
        if not folder_path.exists() or not folder_path.is_dir():
            missing_folders.append(f"{folder_name} (基准目录 {base_dir})")
    
    # 5.2 检查仅根目录存在的文件夹（annotations、videos）
    logger.info(f"检查根目录 {root_dir} 下的 annotations/videos 文件夹")
    for folder_name in root_only_folders:
        # 若基准目录是backup，跳过backup内检查，直接查根目录；否则查基准目录（即根目录）
        check_path = root_dir / folder_name
        if not check_path.exists() or not check_path.is_dir():
            missing_folders.append(f"{folder_name} (根目录 {root_dir})")
    
    if missing_folders:
        err_msg = f"缺失以下必要文件夹: {', '.join(missing_folders)}"
        # 补充说明backup场景的检查规则
        if is_backup_base:
            err_msg += "（注：backup目录下跳过annotations/videos检查，转至上一级根目录验证）"
        logger.error(err_msg)
        return False, err_msg

    # ========== 6. 所有检查通过 ==========
    logger.info(f"前置检查通过：基准目录 {base_dir} 下核心文件/文件夹均存在，根目录 {root_dir} 下 annotations/videos 文件夹也存在")
    return True, None

def reorganize_data_after_merge(
    root_dir: str | Path,
    logger: logging.Logger | None = None
) -> None:
    """
    合并数据后整理目录结构：
    1. 将指定文件/文件夹移到backup目录
    2. 新建data目录，迁移原merged_data内容
    3. 新建meta目录，复制指定文件并改名
    :param root_dir: 数据集根目录
    :param logger: 日志对象
    """
    logger = logger or logging.getLogger(__name__)
    root_dir = Path(root_dir).expanduser().absolute()
    logger.info(f"开始整理合并后的数据目录: {root_dir}")

    # ========== 1. 定义需要备份的文件/文件夹清单 ==========
    # 要备份的文件夹
    backup_folders = [
        root_dir / "meta",
        root_dir / "state_action_data",
        root_dir / "data",
        root_dir / "motion_annotation_data",
        root_dir / "subtask_annotation_data",
        root_dir / "merged_data",
        root_dir / "scene_annotation_data"
    ]
    # 要备份的文件
    backup_files = [
        root_dir / "episode_source_mapping.json",
        root_dir / "original_data_paths.json"
    ]

    # ========== 2. 创建backup目录（不存在则创建） ==========
    backup_dir = root_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"创建backup目录: {backup_dir}")

    # ========== 3. 移动文件夹到backup ==========
    for folder in backup_folders:
        if folder.exists() and folder.is_dir():
            target_path = backup_dir / folder.name
            # 避免目标已存在（如重复执行）
            if target_path.exists():
                shutil.rmtree(target_path)
                logger.warning(f"backup目录中已存在 {target_path}，已删除后重新移动")
            
            shutil.move(str(folder), str(target_path))
            logger.info(f"移动文件夹到backup: {folder} → {target_path}")
        else:
            logger.debug(f"文件夹不存在，跳过备份: {folder}")

    # ========== 4. 移动文件到backup ==========
    for file in backup_files:
        if file.exists() and file.is_file():
            target_path = backup_dir / file.name
            if target_path.exists():
                target_path.unlink()
                logger.warning(f"backup目录中已存在 {target_path}，已删除后重新移动")
            
            shutil.move(str(file), str(target_path))
            logger.info(f"移动文件到backup: {file} → {target_path}")
        else:
            logger.debug(f"文件不存在，跳过备份: {file}")

    # ========== 5. 新建data目录，迁移原merged_data内容（从backup中取出） ==========
    new_data_dir = root_dir / "data"
    new_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"创建新的data目录: {new_data_dir}")

    # 原merged_data已备份到backup，从backup中复制内容到新data
    backup_merged_data = backup_dir / "merged_data"
    if backup_merged_data.exists() and backup_merged_data.is_dir():
        # 复制merged_data下所有内容到新data
        for item in backup_merged_data.iterdir():
            target_item = new_data_dir / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(target_item), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(target_item))
        logger.info(f"将原merged_data内容迁移到新data目录: {backup_merged_data} → {new_data_dir}")
    else:
        raise FileNotFoundError(f"backup目录中未找到merged_data: {backup_merged_data}，无法重建data目录")

    # ========== 6. 新建meta目录，复制指定文件并改名 ==========
    new_meta_dir = root_dir / "meta"
    new_meta_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"创建新的meta目录: {new_meta_dir}")

    # 原meta已备份到backup，从backup中读取指定文件
    backup_meta_dir = backup_dir / "meta"
    if not backup_meta_dir.exists():
        raise FileNotFoundError(f"backup目录中未找到meta: {backup_meta_dir}，无法重建meta目录")

    # 需要复制并改名的文件映射 {原文件名: 新文件名}
    meta_file_mapping = {
        "episodes.jsonl": "episodes.jsonl",  # 直接复制
        "merged_episodes_stats.jsonl": "episodes_stats.jsonl",  # 改名
        "merged_info.json": "info.json",  # 改名
        "tasks.jsonl": "tasks.jsonl"  # 直接复制
    }

    for src_name, dst_name in meta_file_mapping.items():
        src_path = backup_meta_dir / src_name
        dst_path = new_meta_dir / dst_name

        if src_path.exists() and src_path.is_file():
            # 覆盖已存在的文件（如重复执行）
            if dst_path.exists():
                dst_path.unlink()
            shutil.copy2(str(src_path), str(dst_path))
            logger.info(f"复制并整理meta文件: {src_path} → {dst_path}")
        else:
            raise FileNotFoundError(f"backup/meta中缺失必要文件: {src_path}，无法完成meta目录重建")

    logger.info(f"数据目录整理完成: {root_dir}")

def reorganize_data_after_merge(
    root_dir: str | Path,
    file_type: str = "merged_info",  # 新增参数：merged_info / info
    logger: logging.Logger | None = None
) -> None:
    """
    合并数据后整理目录结构：
    1. merged_info模式：完整备份+重建目录（原逻辑）
    2. info模式：仅备份指定目录+直接复制原始meta文件（不改名）
    :param root_dir: 数据集根目录
    :param file_type: 文件类型，merged_info/info
    :param logger: 日志对象
    """
    logger = logger or logging.getLogger(__name__)
    root_dir = Path(root_dir).expanduser().absolute()
    logger.info(f"开始整理数据目录，模式: {file_type} | 路径: {root_dir}")

    # 统一创建backup目录
    backup_dir = root_dir / "backup"
    # 如果backup文件夹已存在，直接跳过所有整理操作
    if backup_dir.exists() and backup_dir.is_dir():
        logger.info(f"backup目录已存在: {backup_dir}，跳过数据目录整理操作")
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"确保backup目录存在: {backup_dir}")

    # ====================== 场景1：merged_info.json 模式（原有完整逻辑）======================
    if file_type == "merged_info":
        logger.info("执行 merged_info 模式目录整理")
        # 1. 定义需要备份的文件/文件夹清单
        backup_folders = [
            root_dir / "meta",
            root_dir / "state_action_data",
            root_dir / "data",
            root_dir / "motion_annotation_data",
            root_dir / "subtask_annotation_data",
            root_dir / "merged_data",
            root_dir / "scene_annotation_data"
        ]
        backup_files = [
            root_dir / "episode_source_mapping.json",
            root_dir / "original_data_paths.json"
        ]

        # 2. 移动文件夹到backup
        for folder in backup_folders:
            if folder.exists() and folder.is_dir():
                target_path = backup_dir / folder.name
                if target_path.exists():
                    shutil.rmtree(target_path)
                    logger.warning(f"backup已存在 {target_path}，删除后重新移动")
                shutil.move(str(folder), str(target_path))
                logger.info(f"移动文件夹: {folder} → {target_path}")
            else:
                logger.debug(f"文件夹不存在，跳过: {folder}")

        # 3. 移动文件到backup
        for file in backup_files:
            if file.exists() and file.is_file():
                target_path = backup_dir / file.name
                if target_path.exists():
                    target_path.unlink()
                    logger.warning(f"backup已存在 {target_path}，删除后重新移动")
                shutil.move(str(file), str(target_path))
                logger.info(f"移动文件: {file} → {target_path}")
            else:
                logger.debug(f"文件不存在，跳过: {file}")

        # 4. 新建data目录，迁移merged_data
        new_data_dir = root_dir / "data"
        new_data_dir.mkdir(parents=True, exist_ok=True)
        backup_merged_data = backup_dir / "merged_data"
        if backup_merged_data.exists() and backup_merged_data.is_dir():
            for item in backup_merged_data.iterdir():
                target_item = new_data_dir / item.name
                if item.is_dir():
                    shutil.copytree(str(item), str(target_item), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(item), str(target_item))
            logger.info(f"迁移merged_data → 新data目录: {new_data_dir}")
        else:
            raise FileNotFoundError(f"backup中未找到merged_data: {backup_merged_data}")

        # 5. 新建meta目录，复制并改名文件
        new_meta_dir = root_dir / "meta"
        new_meta_dir.mkdir(parents=True, exist_ok=True)
        backup_meta_dir = backup_dir / "meta"
        if not backup_meta_dir.exists():
            raise FileNotFoundError(f"backup中未找到meta: {backup_meta_dir}")

        meta_file_mapping = {
            "episodes.jsonl": "episodes.jsonl",
            "merged_episodes_stats.jsonl": "episodes_stats.jsonl",
            "merged_info.json": "info.json",
            "tasks.jsonl": "tasks.jsonl"
        }
        for src_name, dst_name in meta_file_mapping.items():
            src_path = backup_meta_dir / src_name
            dst_path = new_meta_dir / dst_name
            if src_path.exists():
                if dst_path.exists():
                    dst_path.unlink()
                shutil.copy2(str(src_path), str(dst_path))
                logger.info(f"复制meta文件: {src_name} → {dst_name}")
            else:
                raise FileNotFoundError(f"缺失meta文件: {src_path}")

    # ====================== 场景2：info.json 模式（你的新需求逻辑）======================
    elif file_type == "info":
        logger.info("执行 info 模式目录整理（无改名，直接复制原始文件）")
        # 1. 定义info模式需要备份的清单（移除了 data 文件夹）
        backup_folders = [
            root_dir / "meta",
            root_dir / "state_action_data",
            root_dir / "motion_annotation_data",
            root_dir / "subtask_annotation_data",
            root_dir / "merged_data",
            root_dir / "scene_annotation_data"
        ]
        backup_files = [
            root_dir / "episode_source_mapping.json",
            root_dir / "original_data_paths.json"
        ]

        # 2. 剪切文件夹到backup
        for folder in backup_folders:
            if folder.exists() and folder.is_dir():
                target_path = backup_dir / folder.name
                if target_path.exists():
                    shutil.rmtree(target_path)
                    logger.warning(f"backup已存在 {target_path}，删除后重新剪切")
                shutil.move(str(folder), str(target_path))
                logger.info(f"剪切文件夹到backup: {folder} → {target_path}")
            else:
                logger.debug(f"文件夹不存在，跳过: {folder}")

        # 3. 剪切文件到backup
        for file in backup_files:
            if file.exists() and file.is_file():
                target_path = backup_dir / file.name
                if target_path.exists():
                    target_path.unlink()
                    logger.warning(f"backup已存在 {target_path}，删除后重新剪切")
                shutil.move(str(file), str(target_path))
                logger.info(f"剪切文件到backup: {file} → {target_path}")
            else:
                logger.debug(f"文件不存在，跳过: {file}")

        # 4. 新建meta文件夹
        new_meta_dir = root_dir / "meta"
        new_meta_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建新meta目录: {new_meta_dir}")

        # 5. 从backup/meta直接复制原始文件（不改名）
        backup_meta_dir = backup_dir / "meta"
        if not backup_meta_dir.exists():
            raise FileNotFoundError(f"backup目录中未找到meta: {backup_meta_dir}，无法重建meta目录")

        # 直接复制的文件列表（不改名）
        meta_files = [
            "episodes.jsonl",
            "episodes_stats.jsonl",
            "info.json",
            "tasks.jsonl"
        ]
        for file_name in meta_files:
            src_path = backup_meta_dir / file_name
            dst_path = new_meta_dir / file_name
            if src_path.exists() and src_path.is_file():
                if dst_path.exists():
                    dst_path.unlink()
                shutil.copy2(str(src_path), str(dst_path))
                logger.info(f"直接复制meta文件: {src_path} → {dst_path}")
            else:
                raise FileNotFoundError(f"backup/meta中缺失必要文件: {src_path}")

    else:
        err_msg = f"不支持的文件类型: {file_type}，仅支持 merged_info / info"
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info(f"数据目录整理完成【{file_type}模式】: {root_dir}")

# ========== 新增：清理函数 ==========
def clean_merge_temp_files(root_dir: str | Path, logger: logging.Logger | None = None) -> None:
    """
    合并完成后整理指定的临时文件和文件夹
    :param root_dir: 数据集根目录
    :param logger: 日志对象
    """
    logger = logger or logging.getLogger(__name__)
    root_dir = Path(root_dir).expanduser().absolute()
    meta_dir = root_dir / "meta"  # meta文件夹路径
    
    # 1. 定义需要整理的文件夹
    folders_to_delete = [
        root_dir / "motion_annotation_data",
        root_dir / "state_action_data"
    ]
    
    # 2. 定义需要删除的文件（meta文件夹下）
    files_to_delete = [
        meta_dir / "motion_annotation_episodes_stats.jsonl",
        meta_dir / "motion_annotation_info.json",
        meta_dir / "state_action_episodes_stats.jsonl",
        meta_dir / "state_action_info.json"
    ]
    
    # # 3. 删除文件夹
    # for folder in folders_to_delete:
    #     if folder.exists() and folder.is_dir():
    #         try:
    #             shutil.rmtree(folder)
    #             logger.info(f"成功删除文件夹: {folder}")
    #         except Exception as e:
    #             logger.error(f"删除文件夹失败 {folder}: {str(e)}")
    #     else:
    #         logger.debug(f"文件夹不存在，跳过删除: {folder}")
    
    # # 4. 删除文件
    # for file in files_to_delete:
    #     if file.exists() and file.is_file():
    #         try:
    #             file.unlink()
    #             logger.info(f"成功删除文件: {file}")
    #         except Exception as e:
    #             logger.error(f"删除文件失败 {file}: {str(e)}")
    #     else:
    #         logger.debug(f"文件不存在，跳过删除: {file}")


def merge_dataset_data(root_dir: str | Path, patch_features: list[str], merge_feature: str) -> None:
    merge_dataset_info_files(root_dir, patch_features, merge_feature)
    ep_num = get_episode_num(root_dir)
    merge_dataset_stats_jsonl_files(
        root_dir=root_dir,
        patch_features=patch_features,
        merge_feature=merge_feature,
        ep_num=ep_num,
    )
    merge_dataset_parquet_files(root_dir, patch_features, merge_feature)


def _sync_tasks(
    session: Session,
    task_status_field: str,
    task_version_field: str,
    dependencies: list[dict[str, any]],
) -> None:
    try:
        cur_status_col = getattr(DatasetDB, task_status_field)
        cur_version_col = getattr(DatasetDB, task_version_field)
    except AttributeError as e:
        raise ValueError(f"当前任务字段不存在: {e}")

    # 1. 所有前置任务必须满足状态要求
    dependency_status_conditions = []
    # 2. 判断是否任一依赖的 version > 其对应的 version_ps（版本过期）
    version_outdated_conditions = []

    for dep in dependencies:
        try:
            status_col = getattr(DatasetDB, dep["status_field"])
            version_col = getattr(DatasetDB, dep["version_field"])
            version_ps_col = getattr(DatasetDB, dep["version_ps_field"])
        except AttributeError as e:
            raise ValueError(f"字段不存在: {e}")

        # 条件1：前置任务状态达标
        dependency_status_conditions.append(status_col == dep["required_status"])

        # 条件2：当前任务已完成，但该依赖的版本 > 其对应的 version_ps（说明过期）
        version_outdated_conditions.append(version_col > version_ps_col)

    # 所有前置状态必须满足
    all_deps_satisfied = and_(*dependency_status_conditions)

    # 触发条件：两个分支
    trigger_condition = or_(
        # 分支1：当前任务已经是 PENDING（已在队列中）
        cur_status_col == TaskStatus.PENDING,
        # 分支2：已完成，但至少一个依赖版本更新了（version > version_ps）
        and_(
            cur_status_col == TaskStatus.COMPLETED,
            or_(*version_outdated_conditions),  # 任一依赖版本更新
        ),
    )

    query = session.query(DatasetDB).filter(and_(all_deps_satisfied, trigger_condition))
    items = query.all()

    for item in items:
        # 跳过已在 PENDING 的任务
        if getattr(item, task_status_field) == TaskStatus.PENDING:
            continue

        # 检查是否任一依赖版本更新（触发重跑判断）
        need_update = False
        for dep in dependencies:
            dep_status = getattr(item, dep["status_field"])
            dep_version = getattr(item, dep["version_field"])
            dep_version_ps = getattr(item, dep["version_ps_field"])

            if dep_status == dep["required_status"] and dep_version > dep_version_ps:
                need_update = True
                break

        if not need_update:
            continue

        # ✅ 动态更新：当前任务状态 + 版本
        setattr(item, task_status_field, TaskStatus.PENDING)
        setattr(item, task_version_field, getattr(item, task_version_field) + 1)

        # ✅ 关键：为每一个依赖项更新其对应的 version_ps 字段（这才是完整的适配）
        for dep in dependencies:
            current_version = getattr(item, dep["version_field"])
            setattr(item, dep["version_ps_field"], current_version)


def _sync_data_merge_tasks(
    session: Session, device_model: str | None = None, device_model_version: str | None = None
) -> None:
    # 仅筛选【运动标注完成】的数据集，删除视频嵌入、场景标注的筛选条件
    query = (
        session.query(DatasetDB)
        .filter(DatasetDB.qced_repo_gen_status == TaskStatus.COMPLETED)  # 新增：格式转换完成
        .filter(DatasetDB.motion_annotation_status == TaskStatus.COMPLETED)
        .filter(DatasetDB.is_ignore == False)
    )

    # 触发条件：仅判断运动标注的版本是否过期，删除另外两个版本的判断
    query = query.filter(
        or_(
            DatasetDB.data_merge_status == TaskStatus.PENDING,
            and_(
                DatasetDB.data_merge_status == TaskStatus.COMPLETED,
                # 仅保留运动标注的版本过期判断
                DatasetDB.data_merge_version_ps_ma < DatasetDB.motion_annotation_version,
            ),
        )
    )

    if device_model:
        query = query.filter(
            DatasetDB.device_model == device_model,
        )
        if device_model_version:
            query = query.filter(DatasetDB.device_model_version == device_model_version)

    items = query.all()
    if not items:
        return
    
    for item in items:
        item.data_merge_status = TaskStatus.PENDING
        # 仅同步运动标注的版本，删除另外两个版本的同步
        item.data_merge_version_ps_ma = item.motion_annotation_version

    session.commit()


def _gen_one_dataset_data_merge_task(
    session: Session,
) -> tuple[str | None, str | None]:
    query = (
        session.query(DatasetDB)
        .filter(
            DatasetDB.data_merge_status == TaskStatus.PENDING,
        )
        # .filter(
        #     DatasetDB.video_embed_subtask_annotation_status == TaskStatus.COMPLETED,
        # )
        .filter(
            DatasetDB.motion_annotation_status == TaskStatus.COMPLETED,
        )
        .filter(DatasetDB.qced_repo_gen_status == TaskStatus.COMPLETED,) 
        # .filter(
        #     DatasetDB.scene_annotation_status == TaskStatus.COMPLETED,
        # )
    )
    item = query.first()
    if not item:
        return None, None
    item.data_merge_status = TaskStatus.PROCESSING
    item.data_merge_version = item.data_merge_version + 1
    session.commit()
    # ========== 核心修改：将 convert_path 改为 qced_repo_gen_path ==========
    return item.dataset_uuid, item.qced_repo_gen_path


class DataMerger:
    def __init__(
        self,
        db_file_path: str | Path,
        logger: logging.Logger | None = None,
        data_merge_config: DataMergeConfig = DataMergeConfig(),
    ) -> None:
        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.logger = logger or logging.getLogger(__name__)
        self.data_merge_config = data_merge_config

    def merge_data_by_uuid(self, dataset_uuid: str) -> bool:
        """
        指定数据集UUID执行合并操作（用于测试）
        :param dataset_uuid: 数据集唯一标识
        :return: 合并是否成功
        """
        self.logger.info(f"开始处理指定UUID的数据集合并: {dataset_uuid}")
        
        # 1. 验证UUID是否存在并检查前置条件
        with self.db.with_session() as session:
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
            
            if not item:
                self.logger.error(f"UUID {dataset_uuid} 不存在于数据库中")
                return False
            
            # 检查前置任务状态
            pre_conditions = [
                (item.qced_repo_gen_status == TaskStatus.COMPLETED, "qc仓库生成未完成"),
                (item.motion_annotation_status == TaskStatus.COMPLETED, "运动标注未完成"),
                (item.qced_repo_gen_path is not None, "qc仓库路径为空"),
            ]
            
            for condition, err_msg in pre_conditions:
                if not condition:
                    self.logger.error(f"前置条件不满足: {err_msg}")
                    return False
            
            # 更新任务状态为PROCESSING
            item.data_merge_status = TaskStatus.PROCESSING
            item.data_merge_version = item.data_merge_version + 1
            item.data_merge_version_ps_ma = item.motion_annotation_version  # 同步运动标注版本
            repo_path = item.qced_repo_gen_path
            session.commit()
        
        # 2. 执行合并操作
        try:
            self._merge_data(repo_path)
            
            # 3. 更新任务状态为完成
            with self.db.with_session() as session:
                item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                if item:
                    item.data_merge_status = TaskStatus.COMPLETED
                    item.data_merge_err_msg = None
                    session.commit()
            
            self.logger.info(f"UUID {dataset_uuid} 合并完成，路径: {repo_path}")
            return True
            
        except Exception as e:
            error_msg = str(traceback.format_exc())
            self.logger.error(f"UUID {dataset_uuid} 合并失败: {error_msg}")
            
            # 4. 更新任务状态为失败
            with self.db.with_session() as session:
                item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                if item:
                    item.data_merge_status = TaskStatus.FAILED
                    item.data_merge_err_msg = error_msg
                    session.commit()
            
            return False

    def _merge_data(self, convert_path: str) -> None:
        self.logger.info(f"开始合并数据: {convert_path}")


         # 第一步：执行前置文件/文件夹检查
        pre_check_pass, pre_err_msg = check_required_files_and_folders(
            root_dir=convert_path,
            logger=self.logger
        )
        if not pre_check_pass:
            print(f"前置检查失败: {pre_err_msg}")
            return
         # 1. 检查文件存在性+合规性（新增文件类型返回值）
        file_exists, is_valid, err_msg, _, check_file_type = check_merged_info_file(
            root_dir=convert_path,
            merge_feature=self.data_merge_config.merge_feature,
            logger=self.logger
        )
        
        # 2. 文件存在 → 区分文件类型判断是否执行整理
        if file_exists and is_valid:
            self.logger.info(f"检测到有效文件（类型: {check_file_type}），跳过合并操作（合规性: {is_valid}）")
            
            # 仅当检查的是 merged_info.json 时，才执行数据整理
            if check_file_type == "merged_info":
                self.logger.info("检查到 merged_info.json 存在，执行数据整理")
                try:
                    reorganize_data_after_merge(convert_path, file_type="merged_info", logger=self.logger)
                except Exception as e:
                    self.logger.error(f"数据整理失败: {str(e)}", exc_info=True)
                    raise RuntimeError(f"合并跳过但数据整理失败: {str(e)}") from e
            else:
                self.logger.info("检查到 info.json 存在，执行数据整理")
                reorganize_data_after_merge(convert_path, file_type="info", logger=self.logger)
            return
        # if not is_valid:
        #     self.logger.info(f"检测到无效文件（类型: {check_file_type}），跳过合并操作, {convert_path}")
        #     return

        # 3. 文件不存在 → 执行合并流程
        self.logger.info("merged_info.json不存在，执行合并操作")
        merge_dataset_data(
            convert_path,
            self.data_merge_config.patch_features,
            self.data_merge_config.merge_feature,
        )
        try:
            reorganize_data_after_merge(convert_path, file_type="info", logger=self.logger)
        except Exception as e:
            self.logger.error(f"数据整理失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"合并成功但数据整理失败: {str(e)}") from e
        # 调用清理函数
        clean_merge_temp_files(convert_path, self.logger)

    def merge_data_one_dataset(self) -> None:
        with self.db.with_session() as session:
            _sync_data_merge_tasks(session)
            # 接收的路径已改为 qced_repo_gen_path，变量名可保留（不影响逻辑）
            dataset_uuid, repo_path = _gen_one_dataset_data_merge_task(session=session)
            if dataset_uuid is None:
                return
        try:
            self._merge_data(repo_path)
            with self.db.with_session() as session:
                item = (
                    session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                )

                if not item:
                    return
                item.data_merge_status = TaskStatus.COMPLETED
                session.commit()
        except Exception as e:
            with self.db.with_session() as session:
                query = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid)
                item = query.first()
                if not item:
                    return
                item.data_merge_status = TaskStatus.FAILED
                item.data_merge_err_msg = str(traceback.format_exc())
            self.logger.error(e)


class DataMergerServer(TaskServer):
    def __init__(
        self,
        db_file_path: str | Path,
        host: str = "0.0.0.0",
        port: int = 8765,
        heartbeat_interval: float = 30.0,  # 服务端每30秒发一次 ping
        timeout: float = 15.0,  # 等待 pong 超过15秒则断开
        logger: logging.Logger | None = None,
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

    def get_task_category(self) -> str:
        return "data_merge"

    def generate_task_content(self) -> dict | None:
        with self.db.with_session() as session:
            _sync_data_merge_tasks(session)
            # 接收的路径已改为 qced_repo_gen_path
            dataset_uuid, repo_path = _gen_one_dataset_data_merge_task(session)

            if not dataset_uuid:
                return None
            return {
                DATASET_UUID: dataset_uuid,
                LEFORMAT_PATH: repo_path,  # 传递的是 qced_repo_gen_path
            }

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        ds_uuid = task_content.get(DATASET_UUID)

        task_status = task_result_content.get(TASK_RESULT_STATUS)
        task_status_msg = task_result_content.get(ERR_MSG)

        data_merge_status = (
            TaskStatus.COMPLETED if task_status == TASK_SUCCESS else TaskStatus.FAILED
        )

        # 🆕 合并为单个session，保证原子性
        with self.db.with_session() as session:
            # 查询 device_model_version
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
            if item is None:
                self.logger.error(f"Dataset {ds_uuid} not found in dataset DB.")
                return  # 新增：避免后续空指针

            # 在同一个session中更新转换状态
            item.data_merge_status = data_merge_status
            item.data_merge_err_msg = task_status_msg
            session.commit()
            # ========== 日志中也改为 qced_repo_gen_path ==========
            self.logger.info(
                f"Upsert {item.qced_repo_gen_path} data merge status to {data_merge_status}, "
                f"update_message: {task_status_msg}"
            )


def merge_data_standalone(
    root_dir: str | Path,
    patch_features,
    merge_feature
) -> None:
    """
    独立可调用的数据合并函数
    :param root_dir: 数据集根目录路径
    :param data_merge_config: 合并配置类（包含patch_features、merge_feature）
    """
    # 转换为标准Path对象
    root_dir = Path(root_dir).expanduser().absolute()
    print(f"开始合并数据: {root_dir}")

    # 第一步：执行前置文件/文件夹检查
    pre_check_pass, pre_err_msg = check_required_files_and_folders(
        root_dir=root_dir,
        logger=None  # 独立函数传None，内部已用print
    )
    if not pre_check_pass:
        print(f"前置检查失败: {pre_err_msg}")
        return

    # 第二步：检查文件存在性+合规性
    file_exists, is_valid, err_msg, _, check_file_type = check_merged_info_file(
        root_dir=root_dir,
        merge_feature="",
        logger=None  # 独立函数传None
    )

    # 第三步：文件存在 → 跳过合并，仅执行整理
    if file_exists and is_valid:
        print(f"检测到有效文件（类型: {check_file_type}），跳过合并操作（合规性: {is_valid}）")
        
        if check_file_type == "merged_info":
            print("检查到 merged_info.json 存在，执行数据整理")
            try:
                reorganize_data_after_merge(root_dir, file_type="merged_info", logger=None)
            except Exception as e:
                print(f"数据整理失败: {str(e)}")
                print(traceback.format_exc())
                raise RuntimeError(f"合并跳过但数据整理失败: {str(e)}") from e
        else:
            print("检查到 info.json 存在，执行数据整理")
            reorganize_data_after_merge(root_dir, file_type="info", logger=None)
        return

    # 第四步：文件不存在 → 执行完整合并流程
    print("merged_info.json不存在，执行合并操作")
    merge_dataset_data(
        root_dir,
        patch_features,
        merge_feature,
    )

    # 第五步：合并后数据整理
    try:
        reorganize_data_after_merge(root_dir, file_type="info", logger=None)
    except Exception as e:
        print(f"数据整理失败: {str(e)}")
        print(traceback.format_exc())
        raise RuntimeError(f"合并成功但数据整理失败: {str(e)}") from e

    # 第六步：清理临时文件
    clean_merge_temp_files(root_dir, logger=None)
    print(f"数据合并+整理+清理全部完成: {root_dir}")

class DataMergerClient(TaskClient):
    def __init__(
        self,
        server_uri: str = "ws://localhost:8767",
        heartbeat_interval: float = 10.0,
        logger: logging.Logger | None = None,
        data_merge_config: DataMergeConfig = DataMergeConfig(),
    ) -> None:
        super().__init__(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            logger=logger,
        )
        self.data_merge_config = data_merge_config        

    def get_task_category(self) -> str:
        return "data_merge"

    def generate_task_request_desc(self) -> dict:
        """客户端可自定义任务请求参数"""
        return {}

    def _sync_process_task(self, task_content: dict) -> dict:
        try:
            # 接收的是 qced_repo_gen_path
            repo_path = task_content.get(LEFORMAT_PATH)
            self.logger.info(f"merge_data: {repo_path}")
            merge_data_standalone(repo_path, patch_features=self.data_merge_config.patch_features,
                merge_feature=self.data_merge_config.merge_feature)

            # merge_dataset_data(
            #     repo_path,
            #     patch_features=self.data_merge_config.patch_features,
            #     merge_feature=self.data_merge_config.merge_feature,
            # )
            # ========== 客户端也调用清理函数 ==========
            clean_merge_temp_files(repo_path, self.logger)
            return {}
        except Exception as e:
            # 异常日志中也使用 qced_repo_gen_path
            raise RuntimeError(f"data merge dataset {repo_path} failed") from e