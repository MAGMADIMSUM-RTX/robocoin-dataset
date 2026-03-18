import importlib
import logging
import traceback
from pathlib import Path

import yaml
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import and_, or_

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    DatasetDB,
    TaskStatus,
)
from robocoin_dataset.distribution_computation.constant import (
    DATASET_UUID,
    DEVICE_MODEL,
    ERR_MSG,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
)
from robocoin_dataset.distribution_computation.task_client import TaskClient
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.format_converter.tolerobot.constant import (
    LEFORMAT_PATH,
)
from robocoin_dataset.state_action_data_post_process.processors.state_action_data_processor_base import (
    StateActionDataPostProcessorBase,
)

PROCESSOR_MODULE_PATH = "processor_module_path"
PROCESSOR_CLASS_NAME = "processor_class_name"
PROCESSOR_LOG_DIR = "processor_log_dir"
PROCESSOR_LOG_NAME = "processor_log_name"
DEVICE_MODEL_VERSION = "device_model_version"

# 新增：定义 qced 相关常量（保持代码风格一致性）
QCED_REPO_GEN_PATH = "qced_repo_gen_path"


def _get_config_classes_dict(
    state_action_dpp_classes_config_file_path: str | Path,
) -> dict[tuple[str, str], tuple[str, str]]:
    if not state_action_dpp_classes_config_file_path.exists():
        raise FileNotFoundError(
            f"processor_class_config_path {state_action_dpp_classes_config_file_path} not exists"
        )
    with open(state_action_dpp_classes_config_file_path) as f:
        yaml_dict = yaml.safe_load(f)

    processor_config_dict = {}
    for device_model_name, configs in yaml_dict.items():
        for config in configs:
            device_version = config["version"]
            class_module_path = config.get("post_processor_module", None)
            if class_module_path is None:
                continue
            class_name = config.get("post_processor_class", None)
            if class_name is None:
                continue
            processor_config_dict[(device_model_name, device_version)] = (
                class_module_path,
                class_name,
            )

    return processor_config_dict


def _sync_state_action_data_post_processing_tasks(
    session: Session, 
    device_model: str | None = None, 
    device_model_version: str | None = None,
    dataset_uuid: str | None = None,  # 新增：支持指定uuid
) -> None:
    # 核心修改1：移除 convert_status 判断，新增 qced_repo_gen_status == TaskStatus.COMPLETED
    query = session.query(DatasetDB).filter(
        and_(
            # 新前提：qced repo 生成必须成功完成
            DatasetDB.qced_repo_gen_status == TaskStatus.COMPLETED,
            # 两个触发分支（逻辑不变，仅依赖新前提）
            or_(
                # 分支1: 正在排队
                DatasetDB.sa_dpp_status == TaskStatus.PENDING,
                # 分支2: 已完成但版本过期
                and_(
                    DatasetDB.sa_dpp_status == TaskStatus.COMPLETED,
                    DatasetDB.sa_dpp_version_ps < DatasetDB.qced_repo_gen_version,  # 同步：改为 qced 版本
                ),
            ),
        )
    )
    
    # 新增：支持指定 dataset_uuid 过滤
    if dataset_uuid:
        query = query.filter(DatasetDB.dataset_uuid == dataset_uuid)
    
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
        item.sa_dpp_status = TaskStatus.PENDING
        item.sa_dpp_version = item.sa_dpp_version + 1 if item.sa_dpp_version is not None else 1
        item.sa_dpp_version_ps = item.qced_repo_gen_version  # 同步：改为 qced 版本

    session.commit()


def _gen_one_state_action_data_post_processing_task(
    session: Session, 
    device_model: str = "", 
    device_model_version: str = "",
    dataset_uuid: str = "",  # 新增：支持指定uuid
) -> tuple[str | None, str | None, str | None, str | None]:
    # 核心修改2：移除 convert_status 判断，新增 qced_repo_gen_status == TaskStatus.COMPLETED
    query = session.query(DatasetDB).filter(
        and_(
            DatasetDB.qced_repo_gen_status == TaskStatus.COMPLETED,
            DatasetDB.sa_dpp_status == TaskStatus.PENDING,
        )
    )
    
    # 新增：支持指定 dataset_uuid 过滤
    if dataset_uuid:
        query = query.filter(DatasetDB.dataset_uuid == dataset_uuid)
    
    if device_model:
        query = query.filter(
            DatasetDB.device_model == device_model,
        )
        if device_model_version:
            query = query.filter(DatasetDB.device_model_version == device_model_version)

    item = query.first()
    if not item:
        return None, None, None, None
    
    item.sa_dpp_status = TaskStatus.PROCESSING
    item.sa_dpp_version_ps = item.qced_repo_gen_version  # 同步：改为 qced 版本
    item.sa_dpp_version = item.sa_dpp_version + 1
    session.commit()
    
    # 核心修改3：返回 qced_repo_gen_path 而非 convert_path
    return item.dataset_uuid, item.qced_repo_gen_path, item.device_model, item.device_model_version


def _state_action_data_post_process(
    convert_path: str | Path,  # 变量名保留（兼容原有逻辑），实际传入的是 qced_repo_gen_path
    processor_class: type[StateActionDataPostProcessorBase],
) -> None:
    if processor_class is None:
        raise ValueError("processor_class is None")
    processor: StateActionDataPostProcessorBase = processor_class(convert_path=convert_path)
    
    # 使得代码不生产新的 json 和 parquet 文件，而是直接修改原来的 json 和 parquet 文件
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from robocoin_dataset.utils.le_path import get_episodes_stats_jsonl_file

    processor.new_parquet_files = processor.parquet_files
    processor.new_info_file = processor.info_file_path
    processor.new_episodes_stats_file_path = get_episodes_stats_jsonl_file(processor.convert_path)

    # 替换写文件方法，使其在原表基础上修改
    def custom_write_new_episode_file(new_data, episode_idx):
        file_path = processor.new_parquet_files[episode_idx]
        table = pq.read_table(file_path)
        for col_name, arr in new_data.items():
            if arr is None:
                continue
            if arr.ndim == 1:
                pa_type = processor._get_pa_type(arr.dtype)
                pa_array = pa.array(arr, type=pa_type)
            elif arr.ndim == 2:
                value_type = processor._get_pa_type(arr.dtype)
                list_type = pa.list_(value_type)
                pa_array = pa.array([row.tolist() for row in arr], type=list_type)
            else:
                raise ValueError(f"Unsupported array dimension: {arr.ndim} for '{col_name}'")
                
            if col_name in table.column_names:
                idx = table.column_names.index(col_name)
                table = table.set_column(idx, table.schema.field(idx).with_type(pa_array.type), pa_array)
            else:
                table = table.append_column(pa.field(col_name, pa_array.type), pa_array)
        pq.write_table(table, file_path)

    processor.write_new_episode_file = custom_write_new_episode_file

    # 替换写 info 方法，使其在原 info.json 基础上修改
    def custom_write_new_info_file():
        with open(processor.info_file_path, "r") as f:
            json_dict = json.load(f)
        if "features" not in json_dict:
            json_dict["features"] = {}
        for feature_key, names in processor.get_modified_feature_names().items():
            if names is not None:
                flattened_names = []
                for item in names:
                    if isinstance(item, list):
                        flattened_names.extend(item)
                    elif isinstance(item, str):
                        flattened_names.append(item)
                names = flattened_names
            if feature_key not in json_dict["features"]:
                json_dict["features"][feature_key] = {}
            json_dict["features"][feature_key]["names"] = names
        with open(processor.new_info_file, "w") as f:
            json.dump(json_dict, f, indent=4)

    processor.write_new_info_file = custom_write_new_info_file

    # 替换写 stats 方法，使其在原 episodes_stats.jsonl 基础上修改
    def custom_write_new_episodes_stats_file():
        # 读取原有的 stats
        ori_stats = []
        if processor.new_episodes_stats_file_path.exists():
            with open(processor.new_episodes_stats_file_path, "r") as f:
                for line in f:
                    if line.strip():
                        ori_stats.append(json.loads(line))
        
        # 将新的 stats 更新到 ori_stats 中
        # processor.episodes_stats 包含了新特征的 stats
        for new_stat in processor.episodes_stats:
            ep_idx = new_stat.get("episode_index")
            # 找到对应的原 stat
            ori_stat = next((s for s in ori_stats if s.get("episode_index") == ep_idx), None)
            if ori_stat is None:
                ori_stats.append(new_stat)
            else:
                if "stats" not in ori_stat:
                    ori_stat["stats"] = {}
                for k, v in new_stat.get("stats", {}).items():
                    ori_stat["stats"][k] = v

        with open(processor.new_episodes_stats_file_path, "w") as f:
            for stat in ori_stats:
                json.dump(stat, f)
                f.write("\n")

    processor._write_new_episodes_stats_file = custom_write_new_episodes_stats_file

    processor.process()


class StateActionDataPostProcess:
    def __init__(
        self,
        db_file_path: str | Path,
        processor_class_config_path: str | Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.logger = logger or logging.getLogger(__name__)
        self.processor_classes_dict = _get_config_classes_dict(processor_class_config_path)

    def state_action_data_post_process_one_dataset(
        self, 
        device_model: str = "", 
        device_model_version: str = "",
        dataset_uuid: str = "",  # 新增：支持指定uuid
    ) -> None:
        with self.db.with_session() as session:
            # 传入 dataset_uuid 进行同步
            _sync_state_action_data_post_processing_tasks(
                session, device_model, device_model_version, dataset_uuid
            )
            # 传入 dataset_uuid 生成任务
            dataset_uuid, qced_repo_gen_path, device_model, device_model_version = (
                _gen_one_state_action_data_post_processing_task(
                    session, device_model, device_model_version, dataset_uuid
                )
            )
            if dataset_uuid is None:
                self.logger.info("No dataset to process")
                return

        try:
            key = (device_model, device_model_version)
            processor_module_path, processor_class_name = self.processor_classes_dict[key]
            processor_class = importlib.import_module(processor_module_path).__getattribute__(
                processor_class_name
            )
            if processor_class is None:
                raise ValueError(
                    f"processor_class not found for {device_model} {device_model_version}"
                )
            # 传入 qced_repo_gen_path 进行处理
            _state_action_data_post_process(qced_repo_gen_path, processor_class)
            with self.db.with_session() as session:
                item = (
                    session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                )
                item.sa_dpp_status = TaskStatus.COMPLETED
                session.commit()
        except Exception as e:
            with self.db.with_session() as session:
                item = (
                    session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                )
                if item is None:
                    raise ValueError(f"Dataset {dataset_uuid} not found")
                item.sa_dpp_status = TaskStatus.FAILED
                item.sa_dpp_err_msg = str(traceback.format_exc())
                session.commit()
            self.logger.error(f"State Action Data post process dataset {qced_repo_gen_path} failed: {e}")


class StateActionDataPostProcessServer(TaskServer):
    def __init__(
        self,
        db_file_path: str | Path,
        state_action_dpp_classes_config_path: str | Path,
        host: str = "0.0.0.0",
        port: int = 8765,
        heartbeat_interval: float = 30.0,  # 服务端每30秒发一次 ping
        device_model: str = "",
        device_model_version: str = "",
        dataset_uuid: str = "",  # 核心新增：添加 dataset_uuid 实例变量
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
        self.device_model = device_model
        self.device_model_version = device_model_version
        self.dataset_uuid = dataset_uuid  # 核心新增：存储 UUID 为实例变量

        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.logger = logger or logging.getLogger(__name__)
        state_action_dpp_classes_config_path = (
            Path(state_action_dpp_classes_config_path).expanduser().absolute()
        )
        if not state_action_dpp_classes_config_path.exists():
            raise FileNotFoundError(
                f"processor_classes_file_path {state_action_dpp_classes_config_path} not exists"
            )

        self.processor_classes_config_dict = _get_config_classes_dict(
            state_action_dpp_classes_config_path
        )

        # 优化日志：打印 UUID 信息
        log_msg = (
            f"State Action Data Post Process Server started, "
            f"device_model={self.device_model}, "
            f"device_model_version={self.device_model_version}"
        )
        if self.dataset_uuid:
            log_msg += f", dataset_uuid={self.dataset_uuid}"
        self.logger.info(log_msg)

    def get_task_category(self) -> str:
        return "state_action_data_post_process"

    def generate_task_content(self) -> dict | None:
        with self.db.with_session() as session:
            # 核心修改4：服务端生成任务时，移除 convert_status，新增 qced_repo_gen_status 判断
            query = session.query(DatasetDB).filter(
                and_(
                    # 新前提：qced repo 生成必须成功完成
                    DatasetDB.qced_repo_gen_status == TaskStatus.COMPLETED,
                    # 两个触发分支（逻辑不变，仅依赖新前提）
                    or_(
                        # 分支1: 正在排队
                        DatasetDB.sa_dpp_status == TaskStatus.PENDING,
                        # 分支2: 已完成但版本过期
                        and_(
                            DatasetDB.sa_dpp_status == TaskStatus.COMPLETED,
                            DatasetDB.sa_dpp_version_ps < DatasetDB.qced_repo_gen_version,  # 同步：改为 qced 版本
                        ),
                    ),
                )
            )
            # 核心新增：使用实例变量 dataset_uuid 过滤查询
            if self.dataset_uuid and self.dataset_uuid != "":
                query = query.filter(DatasetDB.dataset_uuid == self.dataset_uuid)
            
            if self.device_model is not None and self.device_model != "":
                query = query.filter(
                    DatasetDB.device_model == self.device_model,
                )

            if self.device_model_version is not None and self.device_model_version != "":
                query = query.filter(DatasetDB.device_model_version == self.device_model_version)

            item = query.first()

            if not item:
                return None

            item.sa_dpp_status = TaskStatus.PROCESSING
            item.sa_dpp_version = item.sa_dpp_version + 1
            item.sa_dpp_version_ps = item.qced_repo_gen_version  # 同步：改为 qced 版本

            session.commit()
            converter_module_path, converter_class_name = self.processor_classes_config_dict.get(
                (item.device_model, item.device_model_version),
                (None, None),
            )
            if converter_module_path is None or converter_class_name is None:
                raise ValueError(
                    f"No processor config found for device model {item.device_model} and version {item.device_model_version}"
                )

            # 核心修改5：任务内容中返回 qced_repo_gen_path 而非 convert_path
            return {
                DATASET_UUID: item.dataset_uuid,
                LEFORMAT_PATH: item.convert_path,  # 保留兼容，新增 qced 路径字段
                QCED_REPO_GEN_PATH: item.qced_repo_gen_path,  # 新增：传递 qced repo 生成路径
                DEVICE_MODEL: item.device_model,
                DEVICE_MODEL_VERSION: item.device_model_version,
                PROCESSOR_MODULE_PATH: converter_module_path,
                PROCESSOR_CLASS_NAME: converter_class_name,
            }

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        ds_uuid = task_content.get(DATASET_UUID)

        task_status = task_result_content.get(TASK_RESULT_STATUS)
        task_status_msg = task_result_content.get(ERR_MSG)

        sa_dpp_status = TaskStatus.COMPLETED if task_status == TASK_SUCCESS else TaskStatus.FAILED

        # 合并为单个session，保证原子性
        with self.db.with_session() as session:
            # 查询 device_model_version
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
            if item is None:
                self.logger.error(f"Dataset {ds_uuid} not found in dataset DB.")
                return

            # 在同一个session中更新转换状态
            item.sa_dpp_status = sa_dpp_status
            item.sa_dpp_err_msg = task_status_msg
            session.commit()
            self.logger.info(
                f"Upsert {item.qced_repo_gen_path} state action data post process status to {sa_dpp_status}, "  # 改为 qced 路径
                f"update_message: {task_status_msg}"
            )


class StateActionDataPostProcessClient(TaskClient):
    def __init__(
        self,
        server_uri: str = "ws://localhost:8767",
        heartbeat_interval: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            logger=logger,
        )

    def get_task_category(self) -> str:
        return "state_action_data_post_process"

    def generate_task_request_desc(self) -> dict:
        """客户端可自定义任务请求参数"""
        return {}

    def _sync_process_task(self, task_content: dict) -> dict:
        try:
            # 核心修改6：客户端获取 qced_repo_gen_path 而非 convert_path
            qced_repo_gen_path = task_content.get(QCED_REPO_GEN_PATH)
            if not qced_repo_gen_path:
                raise ValueError("qced_repo_gen_path not found in task content")
            
            processor_module_path = task_content.get(PROCESSOR_MODULE_PATH)
            processor_class_name = task_content.get(PROCESSOR_CLASS_NAME)

            processor_class = importlib.import_module(processor_module_path).__getattribute__(
                processor_class_name
            )

            # 传入 qced_repo_gen_path 进行处理
            _state_action_data_post_process(
                convert_path=qced_repo_gen_path, processor_class=processor_class
            )

            return {}
        except Exception as e:
            # 修正：使用 qced 路径打印错误日志
            qced_repo_gen_path = task_content.get(QCED_REPO_GEN_PATH, "unknown path")
            raise RuntimeError(
                f"state action data post process dataset {qced_repo_gen_path} failed"
            ) from e
