import logging

# import uuid
from pathlib import Path

import yaml
from sqlalchemy import and_, or_

from robocoin_dataset.constant import ROBOCOIN_PLATFORM
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    DatasetDB,
    TaskStatus,
)

# from robocoin_dataset.database.services.leformat_converter import upsert_leformat_convert
from robocoin_dataset.distribution_computation.constant import (
    DATASET_NAME,
    DATASET_PATH,
    DATASET_UUID,
    DEVICE_MODEL,
    ERR_MSG,
    TASK_RESULT_CONTENT,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
)
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.format_converter.tolerobot.constant import (
    AUTO_REENCODE,
    CONVERTER_CLASS_NAME,
    CONVERTER_CONFIG,
    CONVERTER_LOG_DIR,
    CONVERTER_LOG_NAME,
    CONVERTER_MODULE_PATH,
    DEFAULT_DEVICE_MODEL_VERSION,
    DEVICE_MODEL_VERSION_KEY,
    IMAGE_WRITER_PROCESSES,
    IMAGE_WRITER_THREADS,
    IS_TEST,
    LEFORMAT_PATH,
    REPO_ID,
    VIDEO_BACKEND,
)


class LeFormatConverterTaskServer(TaskServer):
    def __init__(
        self,
        db_file: Path,
        convert_root_path: Path,
        converter_factory_config_path: Path,
        host: str = "0.0.0.0",
        port: int = 8765,
        heartbeat_interval: float = 30.0,  # 服务端每30秒发一次 ping
        timeout: float = 15.0,  # 等待 pong 超过15秒则断开
        specific_device_model: str | None = None,
        logger: logging.Logger | None = None,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
        is_test: bool = False,
        auto_reencode: bool = False,
    ) -> None:
        super().__init__(
            logger=logger,
            host=host,
            port=port,
            heartbeat_interval=heartbeat_interval,
            timeout=timeout,
        )
        db_file_path = Path(db_file).expanduser().absolute()
        self.db = DatasetDatabase(db_file=db_file_path)
        if not convert_root_path:
            raise ValueError("convert_root must be provided")
        self.convert_root_path = convert_root_path
        self.specific_device_model = specific_device_model
        self.factory_config_dir = Path(converter_factory_config_path).parent
        self.video_backend = video_backend
        self.image_writer_processes = image_writer_processes
        self.image_writer_threads = image_writer_threads

        self.is_test = is_test
        self.auto_reencode = auto_reencode  # 🎬 自动重编码标志

        try:
            with open(converter_factory_config_path) as f:
                self.converter_factory_config = yaml.safe_load(f)
        except Exception:
            raise
        return

    def _get_converter_class_name(self, device_model: str) -> str:
        if device_model not in self.converter_factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")
        return self.converter_factory_config[device_model]["class"]

    def _get_converter_module_path(self, device_model: str) -> str:
        if device_model not in self.converter_factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")
        return self.converter_factory_config[device_model]["module"]

    def _get_converter_config(self, device_model: str) -> dict:
        if device_model not in self.converter_factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")
        convertor_config_path = (
            self.factory_config_dir
            / self.converter_factory_config[device_model]["converter_config_path"]
        )
        try:
            with open(convertor_config_path) as f:
                convertor_config = yaml.safe_load(f)
        except Exception:
            raise
        return convertor_config

    def _get_converter_module_class_config(
        self, device_model: str, device_model_version: str = ""
    ) -> tuple[str, str, dict]:
        if device_model not in self.converter_factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")

        device_model_configs_list = self.converter_factory_config[device_model]
        if device_model_configs_list is None:
            raise ValueError(f"Device model {device_model} not found in factory config.")

        if not isinstance(device_model_configs_list, list):
            raise ValueError(f"Device model {device_model} config must be a list.")

        converter_module_path: str | None = None
        for device_model_config in device_model_configs_list:
            if not device_model_version:
                if device_model_config[DEVICE_MODEL_VERSION_KEY] == DEFAULT_DEVICE_MODEL_VERSION:
                    converter_module_path = device_model_config["module"]
                    converter_class_name = device_model_config["class"]
                    converter_config_file_path = (
                        self.factory_config_dir / device_model_config["converter_config_path"]
                    )
                break
            if device_model_config[DEVICE_MODEL_VERSION_KEY] == device_model_version:
                converter_module_path = device_model_config["module"]
                converter_class_name = device_model_config["class"]
                converter_config_file_path = (
                    self.factory_config_dir / device_model_config["converter_config_path"]
                )
                break

        if converter_module_path is None:
            raise ValueError(
                f"Device model {device_model} with version {device_model_version} not found in factory config."
            )
        with open(converter_config_file_path) as f:
            converter_config = yaml.safe_load(f)
        return converter_module_path, converter_class_name, converter_config

    def get_task_category(self) -> str:
        return "lerobot_format_convert"

    def generate_task_content(self) -> dict | None:
        with self.db.with_session() as session:
            if self.is_test:
                query = session.query(DatasetDB).filter(
                    DatasetDB.convert_test_status == TaskStatus.PENDING,
                )

                if self.specific_device_model:
                    query = query.filter(
                        DatasetDB.device_model == self.specific_device_model,
                    )
                item = query.first()
                if item is None:
                    return None

                # 兼容数据库旧数据：convert_version 可能为 None
                base_version = item.convert_version if isinstance(item.convert_version, int) else 0
                item.convert_test_version = base_version + 1
                item.convert_test_status = TaskStatus.PROCESSING

            else:
                query = session.query(DatasetDB).filter(
                    and_(
                        # 必要前提：convert_test 必须成功
                        DatasetDB.convert_test_status == TaskStatus.COMPLETED,
                        # 两个触发分支
                        or_(
                            # 分支1: 正在排队
                            DatasetDB.convert_status == TaskStatus.PENDING,
                            # 分支2: 已完成但版本过期
                            and_(
                                DatasetDB.convert_status == TaskStatus.COMPLETED,
                                DatasetDB.convert_version_ps < DatasetDB.convert_test_version,
                            ),
                        ),
                    )
                )
                if self.specific_device_model:
                    query = query.filter(
                        DatasetDB.device_model == self.specific_device_model,
                    )
                item = query.first()
                if item is None:
                    return None
                item.convert_status = TaskStatus.PROCESSING
                # 兼容数据库旧数据：convert_test_version / convert_version 可能为 None
                test_version = item.convert_test_version if isinstance(item.convert_test_version, int) else 0
                item.convert_version_ps = test_version
                current_version = item.convert_version if isinstance(item.convert_version, int) else 0
                item.convert_version = current_version + 1

            item.convert_path = str(
                Path(self.convert_root_path) / f"{item.dataset_name}_{item.dataset_name_id}"
            )
            item.data_path = str(Path(item.yaml_file_path).parent)
            session.commit()
            converter_module_path, converter_class_name, converter_config = (
                self._get_converter_module_class_config(
                    device_model=item.device_model,
                    device_model_version=item.device_model_version,
                )
            )
            leformat_name = f"{item.dataset_name.lower()}_{item.dataset_name_id}"
            client_log_path = Path(self.convert_root_path) / "client_logs" / leformat_name
            repo_id = f"{ROBOCOIN_PLATFORM}/{item.dataset_name}_{item.dataset_name_id}"

            return {
                DATASET_UUID: item.dataset_uuid,
                DATASET_NAME: f"{item.dataset_name}_{item.dataset_name_id}",
                LEFORMAT_PATH: item.convert_path,
                DATASET_PATH: item.data_path,
                DEVICE_MODEL: item.device_model,
                CONVERTER_CONFIG: converter_config,
                CONVERTER_MODULE_PATH: converter_module_path,
                CONVERTER_CLASS_NAME: converter_class_name,
                VIDEO_BACKEND: self.video_backend,
                IMAGE_WRITER_PROCESSES: self.image_writer_processes,
                IMAGE_WRITER_THREADS: self.image_writer_threads,
                CONVERTER_LOG_DIR: str(client_log_path),
                REPO_ID: repo_id,
                CONVERTER_LOG_NAME: leformat_name,
                IS_TEST: self.is_test,
                AUTO_REENCODE: self.auto_reencode,
            }

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        ds_uuid = task_content.get(DATASET_UUID)

        task_status = task_result_content.get(TASK_RESULT_STATUS)
        task_status_msg = task_result_content.get(ERR_MSG)

        # 🔧 修复：先获取TASK_RESULT_CONTENT，实际数据在这里面
        actual_result = task_result_content.get(TASK_RESULT_CONTENT, {})
        
        # 🆕 提取转换统计信息（从actual_result中获取）
        total_episodes = actual_result.get("total_episodes")
        converted_episodes = actual_result.get("converted_episodes")
        skipped_episodes = actual_result.get("skipped_episodes")

        convert_status = TaskStatus.COMPLETED if task_status == TASK_SUCCESS else TaskStatus.FAILED

        # 🆕 合并为单个session，保证原子性
        with self.db.with_session() as session:
            # 查询 device_model_version
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
            if item is None:
                self.logger.error(f"❌ Dataset {ds_uuid} not found in dataset DB, cannot update status!")
                return  # 🔧 修复：找不到数据集时直接返回，避免后续错误

            # 在同一个session中更新转换状态
            item.convert_err_msg = task_status_msg
            item.converted_episodes = converted_episodes
            item.total_episodes = total_episodes
            item.skipped_episodes = skipped_episodes
            if self.is_test:
                item.convert_test_status = convert_status
            else:
                item.convert_status = convert_status
            session.commit()
            self.logger.info(
                f"Upsert {ds_uuid} convert status to {convert_status}, "
                f"device_model={item.device_model}, device_model_version={item.device_model_version}, "
                f"total={total_episodes}, converted={converted_episodes}, skipped={skipped_episodes}, "
                f"update_message: {task_status_msg}"
            )
