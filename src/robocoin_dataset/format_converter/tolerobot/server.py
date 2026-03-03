import logging
from datetime import datetime  # 仅保留datetime处理时间
# import uuid
from pathlib import Path

import yaml
from sqlalchemy import and_, or_

# 导入外部的夹爪归一化类
from robocoin_dataset.format_converter.tolerobot.gripper_normalization import GripperOpenNormalizer
from robocoin_dataset.format_converter.tolerobot.data_range_check import LerobotDatasetValidator
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
        enable_gripper_normalization: bool = True,  # 新增：是否启用夹爪归一化
        enable_dataset_validation: bool = True
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
        self.enable_gripper_normalization = enable_gripper_normalization  # 保存归一化开关
        self.enable_dataset_validation = enable_dataset_validation

        try:
            with open(converter_factory_config_path) as f:
                self.converter_factory_config = yaml.safe_load(f)
        except Exception:
            raise
        return

    # ✅ 核心工具函数：生成「2026-03-03 12:40」格式字符串（到分钟）+ 可序列化的时间戳
    def _get_formatted_time(self) -> tuple[str, float]:
        """
        生成：
        - 易读字符串（2026-03-03 12:40，到分钟）
        - 时间戳（float，用于计算耗时，可JSON序列化）
        返回：(格式化字符串, 时间戳)
        """
        now = datetime.now()
        # 格式化为「2026-03-03 12:40」（到分钟，无秒）
        formatted_str = now.strftime("%Y-%m-%d %H:%M")
        # 生成时间戳（float，可JSON序列化）
        timestamp = datetime.timestamp(now)
        return formatted_str, timestamp

    # ✅ 辅助函数：解析易读字符串为datetime（用于计算耗时）
    def _parse_time_str_to_datetime(self, time_str: str) -> datetime:
        """
        将「2026-03-03 12:40」字符串转回datetime对象
        """
        try:
            # 解析到分钟的格式
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        except ValueError:
            # 兼容你的原始需求格式（2026年3月3日12点40分）
            import re
            pattern = r"(\d+)年(\d+)月(\d+)日(\d+)点(\d+)分"
            match = re.match(pattern, time_str)
            if not match:
                raise ValueError(f"无法解析时间字符串：{time_str}")
            
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            hour = int(match.group(4))
            minute = int(match.group(5))
            
            return datetime(year, month, day, hour, minute)

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
        # ✅ 生成易读时间字符串 + 可序列化的时间戳（替代datetime对象）
        convert_start_str, convert_start_ts = self._get_formatted_time()
        
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
            
            # ✅ 核心：直接覆盖原有时间戳字段为易读字符串（2026-03-03 12:40）
            item.convert_start_timestamp = convert_start_str  # 覆盖旧时间戳字段
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

            # ✅ 修复：移除datetime对象，仅传递可序列化的字段
            task_content = {
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
                "convert_start_str": convert_start_str,  # 易读字符串
                "convert_start_ts": convert_start_ts,    # 时间戳（float，可JSON序列化）
                "dataset_uuid": item.dataset_uuid
            }
            
            # ✅ 日志输出易读时间
            self.logger.info(
                f"开始转换数据集 {task_content[DATASET_NAME]} (UUID: {item.dataset_uuid}), 开始时间: {convert_start_str}"
            )
            return task_content

    def _normalize_gripper_open(self, dataset_path: str):
        """
        执行夹爪开度归一化
        Args:
            dataset_path: 转换后数据集的根路径
        """
        if not self.enable_gripper_normalization:
            self.logger.info("夹爪归一化功能已禁用，跳过该步骤")
            return
            
        try:
            self.logger.info(f"开始对数据集 {dataset_path} 执行夹爪开度归一化...")
            # 初始化归一化器
            normalizer = GripperOpenNormalizer(dataset_path)
            # 执行完整的归一化流程
            success = normalizer.run()
            
            if success:
                self.logger.info(f"数据集 {dataset_path} 夹爪归一化完成")
            else:
                self.logger.warning(f"数据集 {dataset_path} 夹爪归一化未完全执行")
                
        except FileNotFoundError as e:
            self.logger.warning(f"归一化所需文件不存在: {e}，跳过归一化")
        except Exception as e:
            self.logger.error(f"执行夹爪归一化失败: {e}", exc_info=True)

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        # ✅ 生成结束时间（易读字符串 + 可序列化时间戳）
        convert_end_str, convert_end_ts = self._get_formatted_time()
        # 获取开始时间相关数据
        convert_start_str = task_content.get("convert_start_str")
        convert_start_ts = task_content.get("convert_start_ts")
        
        # ✅ 计算耗时（基于时间戳，精准且可序列化）
        if convert_start_ts:
            convert_duration = convert_end_ts - convert_start_ts
        else:
            # 降级方案：从字符串解析datetime计算
            try:
                convert_start_dt = self._parse_time_str_to_datetime(convert_start_str)
                convert_end_dt = self._parse_time_str_to_datetime(convert_end_str)
                convert_duration = (convert_end_dt - convert_start_dt).total_seconds()
            except:
                convert_duration = 0.0
        
        ds_uuid = task_content.get(DATASET_UUID)
        dataset_name = task_content.get(DATASET_NAME, "未知数据集")
        dataset_path = task_content.get(LEFORMAT_PATH)

        task_status = task_result_content.get(TASK_RESULT_STATUS)
        task_status_msg = task_result_content.get(ERR_MSG)

        actual_result = task_result_content.get(TASK_RESULT_CONTENT, {})
        total_episodes = actual_result.get("total_episodes")
        converted_episodes = actual_result.get("converted_episodes")
        skipped_episodes = actual_result.get("skipped_episodes")

        convert_status = TaskStatus.COMPLETED if task_status == TASK_SUCCESS else TaskStatus.FAILED

        with self.db.with_session() as session:
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
            if item is None:
                self.logger.error(f"❌ Dataset {ds_uuid} not found in dataset DB, cannot update status!")
                return

            # ✅ 核心：覆盖原有结束时间戳字段为易读字符串
            item.convert_err_msg = task_status_msg
            item.converted_episodes = converted_episodes
            item.total_episodes = total_episodes
            item.skipped_episodes = skipped_episodes
            item.convert_end_timestamp = convert_end_str  # 覆盖旧结束时间戳字段
            item.convert_duration_seconds = convert_duration  # 耗时（秒）
            if self.is_test:
                item.convert_test_status = convert_status
            else:
                item.convert_status = convert_status
            session.commit()
            
            # ✅ 日志格式化
            duration_str = f"{convert_duration:.2f} 秒"
            if convert_duration > 60:
                duration_str += f" ({convert_duration/60:.2f} 分钟)"
                
            self.logger.info(
                f"✅ 数据集 {dataset_name} (UUID: {ds_uuid}) 转换完成！"
                f"开始时间: {convert_start_str}, 结束时间: {convert_end_str}, "
                f"状态: {convert_status}, 耗时: {duration_str}, "
                f"总片段数: {total_episodes}, 成功转换: {converted_episodes}, 跳过: {skipped_episodes}"
            )
        
        # 夹爪归一化 + 数据校验
        if task_status == TASK_SUCCESS and dataset_path:
            self._normalize_gripper_open(dataset_path)

            self.logger.info(f"开始对数据集 {dataset_path} 执行后置物理极限校验...")
            try: 
                validator = LerobotDatasetValidator(dataset_path)
                is_valid, error_details = validator.run()
                
                if is_valid:
                    self.logger.info(f"✅ 数据集 {dataset_path} 后置校验完美通过！")
                else:
                    self.logger.error(f"❌ 数据集 {dataset_path} 校验未通过！注意：数据库中该任务状态仍为 COMPLETED，请根据日志手动核查脏数据。")
                    
            except Exception as e:
                self.logger.error(f"执行数据集校验时发生崩溃: {e}", exc_info=True)