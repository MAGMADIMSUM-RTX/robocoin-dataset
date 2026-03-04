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
            DatasetDB.motion_annotation_status,
            DatasetDB.motion_annotation_version,
            DatasetDB.data_merge_version_ps_ma,
        ),
    }
    patch_features = ["motion_annotation", "state_action"]
    merge_feature = None


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

# ========== 新增：清理函数 ==========
def clean_merge_temp_files(root_dir: str | Path, logger: logging.Logger | None = None) -> None:
    """
    合并完成后清理指定的临时文件和文件夹
    :param root_dir: 数据集根目录
    :param logger: 日志对象
    """
    logger = logger or logging.getLogger(__name__)
    root_dir = Path(root_dir).expanduser().absolute()
    meta_dir = root_dir / "meta"  # meta文件夹路径
    
    # 1. 定义需要删除的文件夹
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
    
    # 3. 删除文件夹
    for folder in folders_to_delete:
        if folder.exists() and folder.is_dir():
            try:
                shutil.rmtree(folder)
                logger.info(f"成功删除文件夹: {folder}")
            except Exception as e:
                logger.error(f"删除文件夹失败 {folder}: {str(e)}")
        else:
            logger.debug(f"文件夹不存在，跳过删除: {folder}")
    
    # 4. 删除文件
    for file in files_to_delete:
        if file.exists() and file.is_file():
            try:
                file.unlink()
                logger.info(f"成功删除文件: {file}")
            except Exception as e:
                logger.error(f"删除文件失败 {file}: {str(e)}")
        else:
            logger.debug(f"文件不存在，跳过删除: {file}")


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

    def _merge_data(self, convert_path: str) -> None:
        self.logger.info(f"merge_data: {convert_path}")
        merge_dataset_data(
            convert_path,
            self.data_merge_config.patch_features,
            self.data_merge_config.merge_feature,
        )
        # ========== 调用清理函数 ==========
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
            merge_dataset_data(
                repo_path,
                patch_features=self.data_merge_config.patch_features,
                merge_feature=self.data_merge_config.merge_feature,
            )
            # ========== 客户端也调用清理函数 ==========
            clean_merge_temp_files(repo_path, self.logger)
            return {}
        except Exception as e:
            # 异常日志中也使用 qced_repo_gen_path
            raise RuntimeError(f"data merge dataset {repo_path} failed") from e