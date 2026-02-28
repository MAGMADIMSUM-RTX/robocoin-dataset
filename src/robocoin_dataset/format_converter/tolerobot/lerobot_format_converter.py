import importlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import glob
import shutil

from robocoin_dataset.constant import (
    LOCAL_DATASET_INFO_FILE,
    LOCAL_TASK_INFO_FILE_NAME,
    TASK_DESCRIPTIONS_KEY,
    TASK_INDEX_KEY,
)
from robocoin_dataset.format_converter.tolerobot.constant import (
    ACTION_KEY,
    ARGS_KEY,
    CAM_NAME_KEY,
    CONVERT_FUNC_KEY,
    DEFAULT_IMAGE_SHAPE_NAMES,
    DTYPE_KEY,
    FEATURES_KEY,
    FLOAT32,
    FPS,
    FRAME_IDX_KEY,
    IMAGE_DTYPE_VALUE,
    IMAGE_KEY,
    LEROBOT_FEATURE_KEY,
    NAME_KEY,
    OBSERVATION_KEY,
    SHAPE_KEY,
    STATE_KEY,
    SUB_ACTION_KEY,
    SUB_STATE_KEY,
    TIMELINE_OFFSET_KEY,
)
from robocoin_dataset.format_converter.tolerobot.exceptions import (
    ConfigError,
    CriticalDataError,
    DataQualityError,
    FrameCountMismatchError,
)
from robocoin_dataset.format_converter.utils.spatial_data_convertor import spatial_covertor_funcs
import re


# 校验规则配置
VALIDATION_CONFIG = {
    "cam_name": {
        "valid_positions": [
            "left", "right", "front", "rear", "upper", 
            "lower", "middle", "top", "side", "global", "env"
        ],
        "valid_parts": [
            "wrist", "head", "chest", "arm", "leg", "torso"
        ],
        "encoding_map": ["rgb", "depth"]
    },
    "state_action_names": {
    # 核心修改：将 gripper_open_scale 加入正则匹配规则
    "pattern": "^(left|right)_(arm_joint_\\d+_rad|gripper_open|eef_pos_[xyz]_m|eef_rot_euler_[xyz]_rad)$",
    "valid_prefixes": ["left", "right"],
    "valid_types": [
        "arm_joint_1_rad", "arm_joint_2_rad", "arm_joint_3_rad", 
        "arm_joint_4_rad", "arm_joint_5_rad", "arm_joint_6_rad",
        "gripper_open",  # 保留该类型，与正则匹配
        "eef_pos_x_m", "eef_pos_y_m", "eef_pos_z_m",
        "eef_rot_euler_x_rad", "eef_rot_euler_y_rad", "eef_rot_euler_z_rad"
    ],
    "unit_map": {
        "rad": "弧度",
        "m": "米",
    }
}
}

# 预编译正则表达式
STATE_ACTION_PATTERN = re.compile(VALIDATION_CONFIG["state_action_names"]["pattern"])


class LerobotFormatConverter(ABC):
    """
    Base class for converting datasets to the LeRobot format.
    This class should be extended by specific dataset format converters.
    """

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        converter_config: dict,
        repo_id: str,
        device_model: str | None = None,
        logger: logging.Logger | None = None,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
        strict_episodes: int = 3,
        failure_threshold: float = 0.8,
    ) -> None:
        if not dataset_path:
            raise ValueError("Dataset path must be provided.")
        if not output_path:
            raise ValueError("LeRobot destination path must be provided.")

        if dataset_path == output_path:
            raise ValueError("Dataset path and LeRobot destination path cannot be the same.")

        self.dataset_path = Path(dataset_path).expanduser().absolute()
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist.")
        self.output_path = Path(output_path).expanduser().absolute()

        if not converter_config:
            raise ValueError("Convertor config must be provided.")

        self.converter_config = converter_config

        self.repo_id = repo_id

        if not logger:
            raise ValueError("Logger must be provided.")
        self.logger = logger
        self.logger.info(f"Using dataset path: {self.dataset_path}")
        self.device_model = device_model

        # Fault tolerance configuration
        self.strict_episodes = strict_episodes  # 前N个episode使用严格模式
        self.failure_threshold = failure_threshold  # 失败率阈值，超过则认为是配置错误
        
        # Conversion statistics
        self._conversion_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'skipped_episodes': 0,
            'total_frames': 0,
            'skipped_frames': 0,
            'skip_details': [],
        }
        
        # Episode source file mapping: {global_ep_idx: {task, task_ep_idx, source_files}}
        self.episode_source_mapping: dict[int, dict] = {}

        try:
            self._validate_convertor_config()
        except Exception as e:
            raise ValueError(f"Invalid features description: {e}") from e

        # Standard converter initialization
        self.tasks = self._get_tasks()
        self.path_task_dict: dict[Path, str] = self._get_dataset_task_paths()
        self.task_episodes_num: dict[Path, int] = {}
        for task_path in self.path_task_dict.keys():
            self.task_episodes_num[task_path] = self._get_task_episodes_num(task_path)

        self.fps = self.converter_config[FPS]

        self._gen_image_configs()
        self._gen_action_configs()
        self._gen_state_configs()

        self.video_backend = video_backend
        self.image_writer_processes = image_writer_processes
        self.image_writer_threads = image_writer_threads

        self._prevalidate_files()

    def __del__(self) -> None:
        """析构函数：清理资源，防止semaphore泄漏"""
        try:
            if hasattr(self, 'lerobot_dataset') and self.lerobot_dataset is not None:
                # 停止image writer进程池，释放semaphore
                self.lerobot_dataset.stop_image_writer()
                if self.logger:
                    self.logger.debug("✅ Image writer进程池已清理")
        except Exception as e:
            # 析构函数中不应该抛出异常
            if self.logger:
                self.logger.warning(f"⚠️  清理资源时出错: {e}")

    @abstractmethod
    def _prevalidate_files(self) -> None:
        raise NotImplementedError

    def _get_episode_source_files(self, task_path: Path, ep_idx: int) -> dict:
        """获取 episode 的源文件信息（供子类重写）
        
        Args:
            task_path: 任务路径
            ep_idx: episode 索引
        
        Returns:
            dict: 包含源文件详细信息的字典
                建议包含以下字段（视具体数据格式而定）：
                - h5_file: H5文件相对路径
                - video_file: 视频文件相对路径
                - json_file: JSON文件相对路径
                - episode_directory: Episode目录相对路径
                等等
        
        Note:
            默认实现返回空字典，子类可根据需要重写此方法
        """
        return {}

    @abstractmethod
    def _get_frame_image(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        images_buffer: any = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _get_frame_sub_states(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_states_buffer: any = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _get_frame_sub_actions(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_actions_buffer: any = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def _get_task_episodes_num(self, task_path: Path) -> int:
        raise NotImplementedError

    def _prepare_episode_images_buffer(self, task_path: Path, ep_idx: int, is_test: bool = False) -> any:
        """Prepare images buffer for an episode.
        
        Args:
            task_path: Path to the task directory
            ep_idx: Episode index
            is_test: Whether in test mode (子类可选实现优化)
            
        Returns:
            Images buffer (format depends on subclass implementation)
        """
        return None

    def _prepare_episode_states_buffer(self, task_path: Path, ep_idx: int, is_test: bool = False) -> any:
        """Prepare states buffer for an episode.
        
        Args:
            task_path: Path to the task directory
            ep_idx: Episode index
            is_test: Whether in test mode (子类可选实现优化)
            
        Returns:
            States buffer (format depends on subclass implementation)
        """
        return None

    def _prepare_episode_actions_buffer(self, task_path: Path, ep_idx: int, is_test: bool = False) -> any:
        """Prepare actions buffer for an episode.
        
        Args:
            task_path: Path to the task directory
            ep_idx: Episode index
            is_test: Whether in test mode (子类可选实现优化)
            
        Returns:
            Actions buffer (format depends on subclass implementation)
        """
        return None

    def _get_task_episodes_num(self, task_path: Path) -> int:
        if task_path in self.task_episodes_num:
            return self.task_episodes_num[task_path]
        raise ValueError(f"Dataset task_path {task_path} not found")

    def _gen_episode_frame(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        images_buffer: any,
        states_buffer: any,
        actions_buffer: any,
    ) -> dict:
        """
        Generate a single frame for the episode.
        """
        frame_data = {}
        frame_data[FRAME_IDX_KEY] = frame_idx
        frame_data[IMAGE_KEY] = self._get_frame_images(task_path, ep_idx, frame_idx, images_buffer)
        frame_data[STATE_KEY] = self._get_frame_states(task_path, ep_idx, frame_idx, states_buffer)
        frame_data[ACTION_KEY] = self._get_frame_actions(
            task_path, ep_idx, frame_idx, actions_buffer
        )

        return frame_data

    def _validate_convertor_config(self) -> None:
        """
        Validate the convertor config, including cam_name and state/action name format check.
        """
        # check features:
        if FEATURES_KEY not in self.converter_config:
            raise ConfigError(f"Convertion config must contain {FEATURES_KEY} key.")

        # check features.observation:
        if OBSERVATION_KEY not in self.converter_config[FEATURES_KEY]:
            raise ConfigError(
                f"Convertion config must contain {OBSERVATION_KEY} key in {FEATURES_KEY}."
            )

        # check features.observation.images:
        if IMAGE_KEY not in self.converter_config[FEATURES_KEY][OBSERVATION_KEY]:
            raise ConfigError(
                f"Convertion config must contain {IMAGE_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}."
            )

        cam_names: list[str] = []
        for image_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]:
            if CAM_NAME_KEY not in image_config:
                raise ConfigError(
                    f"Convertion config must contain {CAM_NAME_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}.{IMAGE_KEY}."
                )
            
            # 校验摄像头名称格式
            cam_name = image_config[CAM_NAME_KEY]
            LerobotFormatConverter._validate_cam_name(cam_name)
            
            cam_names.append(image_config[CAM_NAME_KEY])
            if ARGS_KEY not in image_config:
                raise ConfigError(
                    f"Convertion config must contain {ARGS_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}.{IMAGE_KEY}."
                )

        if len(set(cam_names)) != len(cam_names):
            raise ConfigError(
                f"Convertion config has same cam_names in {FEATURES_KEY}.{OBSERVATION_KEY}.{IMAGE_KEY}"
            )

        if STATE_KEY not in self.converter_config[FEATURES_KEY][OBSERVATION_KEY]:
            raise ConfigError(
                f"Convertion config must contain {STATE_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}."
            )

        if SUB_STATE_KEY not in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY]:
            raise ConfigError(
                f"Convertion config must contain {SUB_STATE_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}.{STATE_KEY}."
            )

        sub_state_names = []
        for sub_state_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            SUB_STATE_KEY
        ]:
            if NAME_KEY not in sub_state_config:
                raise ConfigError(
                    f"Convertion config must contain {NAME_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}.{STATE_KEY}.{SUB_STATE_KEY}."
                )
            
            # 校验子状态名称格式
            for name in sub_state_config[NAME_KEY]:
                LerobotFormatConverter._validate_state_action_name(name, "state")
            
            sub_state_names.extend(sub_state_config[NAME_KEY])
            if ARGS_KEY not in sub_state_config:
                raise ConfigError(
                    f"Convertion config must contain {ARGS_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}.{STATE_KEY}.{SUB_STATE_KEY}."
                )

        if len(set(sub_state_names)) != len(sub_state_names):
            seen = set()
            duplicates = {x for x in sub_state_names if x in seen or seen.add(x)}
            raise ConfigError(
                f"Convertion config has same state names in {FEATURES_KEY}.{OBSERVATION_KEY}.{STATE_KEY}. "
                f"Duplicates: {list(duplicates)}"
            )

        # check features.action:
        if ACTION_KEY not in self.converter_config[FEATURES_KEY]:
            raise ConfigError(f"Convertion config must contain {ACTION_KEY} key in {FEATURES_KEY}.")

        if SUB_ACTION_KEY not in self.converter_config[FEATURES_KEY][ACTION_KEY]:
            raise ConfigError(
                f"Convertion config must contain {SUB_ACTION_KEY} key in {FEATURES_KEY}.{ACTION_KEY}."
            )

        sub_action_names = []
        for sub_action_config in self.converter_config[FEATURES_KEY][ACTION_KEY][SUB_ACTION_KEY]:
            if NAME_KEY not in sub_action_config:
                raise ConfigError(
                    f"Convertion config must contain {NAME_KEY} key in {FEATURES_KEY}.{ACTION_KEY}.{SUB_ACTION_KEY}."
                )
            
            # 校验子动作名称格式
            for name in sub_action_config[NAME_KEY]:
                LerobotFormatConverter._validate_state_action_name(name, "action")
            
            sub_action_names.extend(sub_action_config[NAME_KEY])
            if ARGS_KEY not in sub_action_config:
                raise ConfigError(
                    f"Convertion config must contain {ARGS_KEY} key in {FEATURES_KEY}.{ACTION_KEY}.{SUB_ACTION_KEY}."
                )

        if len(set(sub_action_names)) != len(sub_action_names):
            raise ConfigError(
                f"Convertion config has same state names in {FEATURES_KEY}.{OBSERVATION_KEY}.{ACTION_KEY}"
            )
 
    @staticmethod
    def _validate_cam_name(cam_name: str) -> None:
        """
        校验摄像头名称格式：cam_<位置>_<模态>
        位置格式：<direction>_<part> (part 可选)
        """
        cam_config = VALIDATION_CONFIG["cam_name"]
        parts = cam_name.split("_")
        
        # 基础格式校验：必须以 cam 开头，且至少包含 3 部分（cam + 位置 + 模态）
        if len(parts) < 3 or parts[0] != "cam":
            raise ValueError(
                f"无效的摄像头名称格式: {cam_name}。"
                f"正确格式应为: cam_<位置>_<模态>，例如 cam_high_rgb、cam_left_wrist_rgb"
            )
        
        # 校验模态（最后一部分）
        encoding = parts[-1]
        if encoding not in cam_config["encoding_map"]:
            raise ValueError(
                f"摄像头名称 {cam_name} 中包含无效的模态 '{encoding}'。"
                f"有效模态列表: {cam_config['encoding_map']}"
            )
        
        # 解析位置部分（cam 和 模态 之间的所有部分）
        position_parts = parts[1:-1]
        if not position_parts:
            raise ValueError(f"摄像头名称 {cam_name} 缺少位置信息（方向/部位）")
        
        # 校验方向（位置第一部分）
        direction = position_parts[0]
        if direction not in cam_config["valid_positions"]:
            raise ValueError(
                f"摄像头名称 {cam_name} 中包含无效的方向 '{direction}'。"
                f"有效方向列表: {cam_config['valid_positions']}"
            )
        
        # 校验部位（位置第二部分及以后，可选）
        if len(position_parts) > 1:
            part = "_".join(position_parts[1:])
            if part not in cam_config["valid_parts"]:
                raise ValueError(
                    f"摄像头名称 {cam_name} 中包含无效的部位 '{part}'。"
                    f"有效部位列表: {cam_config['valid_parts']}"
                )

    @staticmethod
    def _validate_state_action_name(name: str, name_type: str) -> None:
        """
        校验状态/动作名称是否符合规则
        :param name: 要校验的名称
        :param name_type: 名称类型（"state" 或 "action"）
        """
        config = VALIDATION_CONFIG["state_action_names"]
        
        # 1. 正则格式校验
        if not STATE_ACTION_PATTERN.match(name):
            raise ValueError(
                f"无效的{name_type}名称格式: {name}。"
                f"正确格式应匹配正则表达式: {config['pattern']}"
            )
        
        # 2. 拆分前缀和类型，分别校验
        parts = name.split("_", 1)  # 只拆分一次，分离前缀和类型
        if len(parts) != 2:
            raise ValueError(f"无效的{name_type}名称格式: {name}。应包含前缀（left/right）和类型两部分")
        
        prefix, type_part = parts
        
        # 3. 校验前缀
        if prefix not in config["valid_prefixes"]:
            raise ValueError(
                f"{name_type}名称 {name} 中包含无效的前缀 '{prefix}'。"
                f"有效前缀列表: {config['valid_prefixes']}"
            )
        
        # 4. 校验类型
        if type_part not in config["valid_types"]:
            raise ValueError(
                f"{name_type}名称 {name} 中包含无效的类型 '{type_part}'。"
                f"有效类型列表: {config['valid_types']}"
            )


    def _get_tasks(self) -> list[str]:
        dataset_info_file_path = self.dataset_path / LOCAL_DATASET_INFO_FILE
        if not dataset_info_file_path.exists():
            raise ValueError(f"Dataset info file {dataset_info_file_path} not found.")
        with open(dataset_info_file_path) as file:
            ds_info_dict = yaml.safe_load(file)
            if not ds_info_dict:
                raise ValueError(f"Dataset info file {dataset_info_file_path} is empty.")
            if TASK_DESCRIPTIONS_KEY not in ds_info_dict:
                raise ValueError(
                    f"Dataset info file {dataset_info_file_path} does not contain task_description."
                )
            if not isinstance(ds_info_dict[TASK_DESCRIPTIONS_KEY], list):
                raise ValueError(
                    f"Dataset info file {dataset_info_file_path} does not contain task_description as list[str]."
                )
            return ds_info_dict[TASK_DESCRIPTIONS_KEY]

    def _get_dataset_task_paths(self) -> dict[Path, str]:
        """Get the paths of all tasks in the dataset."""
        dirs_to_scan = [self.dataset_path]
        task_paths_dict = {}
        try:
            while len(dirs_to_scan) > 0:
                current_path = dirs_to_scan.pop(0)
                files = Path(current_path).glob(LOCAL_TASK_INFO_FILE_NAME)
                has_task_file = False
                for file in files:
                    try:
                        with file.open("r") as f:
                            task_info_dict = yaml.safe_load(f)
                            task_index = task_info_dict[TASK_INDEX_KEY]
                            task = self.tasks[task_index]
                            task_paths_dict[file.parent] = task
                            has_task_file = True
                    except Exception as e:  # noqa: PERF203
                        raise ValueError(f"Found task index error from {file}") from e

                if not has_task_file:
                    sub_dirs = [item for item in Path(current_path).glob("*") if item.is_dir()]
                    dirs_to_scan.extend(sub_dirs)

        except Exception as e:
            raise e

        return task_paths_dict

    def _get_one_frame_image(self, args_dict: dict) -> np.ndarray:
        """尝试从多个task_path和episode获取样本图像，支持容错
        
        Args:
            args_dict: 相机配置参数
            
        Returns:
            样本图像
            
        Raises:
            RuntimeError: 所有尝试都失败时
            
        Notes:
            为了支持初始化阶段的容错，会尝试多个task_path和episode：
            - 遍历所有task_path
            - 每个task_path尝试前5个episode
            - 找到第一个可用的样本图像即返回
            这样即使第一个episode有问题（损坏、编解码器不兼容等），
            也能成功初始化并转换其他正常的episode
            
            🎯 初始化模式优化（sample_only）：
            - 部分converter支持sample_only参数，用于只加载1帧样本
            - 使用PyAV而非OpenCV，更好地处理编解码器兼容性问题
            - 如果子类支持sample_only，优先使用
        """
        cam_name = args_dict.get(CAM_NAME_KEY, "unknown")
        
        # 检查子类是否支持sample_only参数
        import inspect
        frame_image_sig = inspect.signature(self._get_frame_image)
        supports_sample_only = 'sample_only' in frame_image_sig.parameters
        
        # 尝试多个task_path和episode
        attempted = []
        for task_path in self.path_task_dict.keys():
            num_episodes = self.task_episodes_num.get(task_path, 0)
            # 尝试前几个episode（最多5个）
            for ep_idx in range(min(5, num_episodes)):
                try:
                    # 🎯 如果支持sample_only，优先使用（更兼容的读取方式）
                    if supports_sample_only:
                        image = self._get_frame_image(
                            task_path=task_path, ep_idx=ep_idx, frame_idx=0, 
                            args_dict=args_dict, sample_only=True
                        )
                    else:
                        image = self._get_frame_image(
                            task_path=task_path, ep_idx=ep_idx, frame_idx=0, args_dict=args_dict
                        )
                    
                    if self.logger:
                        mode_info = " (sample mode)" if supports_sample_only else ""
                        self.logger.info(
                            f"✅ Successfully got sample image for camera '{cam_name}' "
                            f"from task_path={task_path.relative_to(self.dataset_path)}, episode={ep_idx}{mode_info}"
                        )
                    return image
                except Exception as e:
                    attempted.append((task_path, ep_idx, str(e)))
                    if self.logger:
                        self.logger.warning(
                            f"⚠️  Failed to get sample image for camera '{cam_name}' from "
                            f"task_path={task_path.relative_to(self.dataset_path)}, "
                            f"episode={ep_idx}: {type(e).__name__}: {e}. Trying next..."
                        )
                    continue
        
        # 所有尝试都失败
        attempted_summary = "\n".join([
            f"  - task_path={tp.relative_to(self.dataset_path)}, episode={ep}: {err}"
            for tp, ep, err in attempted[:10]  # 只显示前10个
        ])
        raise RuntimeError(
            f"❌ Failed to get sample image for camera '{cam_name}' after trying "
            f"{len(attempted)} task_path/episode combinations.\n"
            f"This indicates a systematic problem with the data or configuration.\n"
            f"Attempted:\n{attempted_summary}"
        )

    def _get_one_frame_images(self, image_configs: list[dict]) -> dict[str, np.ndarray]:
        images = {}
        for image_config in image_configs:
            image_name = image_config[CAM_NAME_KEY]
            args_dict = image_config[ARGS_KEY]
            image = self._get_one_frame_image(args_dict)
            images[image_name] = image
        return images

    def _gen_image_configs(self) -> None:
        sample_frame_images = self._get_one_frame_images(
            self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]
        )
        for image_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]:
            image_config[LEROBOT_FEATURE_KEY] = (
                f"{OBSERVATION_KEY}.{IMAGE_KEY}.{image_config[CAM_NAME_KEY]}"
            )
            try:
                image = sample_frame_images[image_config[CAM_NAME_KEY]]
                image_config[DTYPE_KEY] = IMAGE_DTYPE_VALUE
                image_config[NAME_KEY] = DEFAULT_IMAGE_SHAPE_NAMES
                image_config[SHAPE_KEY] = image.shape
            except KeyError as e:
                raise ValueError(
                    f"Convertion config has no {image_config[CAM_NAME_KEY]} in sample frame images"
                ) from e

    def _gen_state_configs(self) -> None:
        sub_state_names = []
        for sub_state_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            SUB_STATE_KEY
        ]:
            sub_state_names.extend(sub_state_config[NAME_KEY])
        self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][NAME_KEY] = sub_state_names
        self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][SHAPE_KEY] = (
            len(sub_state_names),
        )
        self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][LEROBOT_FEATURE_KEY] = (
            f"{OBSERVATION_KEY}.{STATE_KEY}"
        )

    def _gen_action_configs(self) -> None:
        sub_action_names = []
        for sub_action_config in self.converter_config[FEATURES_KEY][ACTION_KEY][SUB_ACTION_KEY]:
            sub_action_names.extend(sub_action_config[NAME_KEY])
        self.converter_config[FEATURES_KEY][ACTION_KEY][NAME_KEY] = sub_action_names

        self.converter_config[FEATURES_KEY][ACTION_KEY][SHAPE_KEY] = (len(sub_action_names),)

        self.converter_config[FEATURES_KEY][ACTION_KEY][LEROBOT_FEATURE_KEY] = ACTION_KEY

    def _get_frame_images(
        self, task_path: Path, ep_idx: int, frame_idx: int, images_buffer: any
    ) -> dict[str, np.ndarray]:
        images = {}
        for image_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]:
            try:
                lerobot_feature = image_config[LEROBOT_FEATURE_KEY]
                args_dict = image_config[ARGS_KEY]
                image = self._get_frame_image(
                    task_path=task_path,
                    ep_idx=ep_idx,
                    frame_idx=frame_idx,
                    args_dict=args_dict,
                    images_buffer=images_buffer,
                )
                images[lerobot_feature] = image
            except (CriticalDataError, DataQualityError):
                # 🔧 容错机制：让 CriticalDataError 和 DataQualityError 直接传播
                # 上层的 _convert_episode_with_fault_tolerance 会捕获并跳过episode
                raise
            except Exception as e:  # noqa: PERF203
                raise Exception(f"Failed to get frame images for {lerobot_feature} failed") from e

        return images

    def _get_frame_states(
        self, task_path: Path, ep_idx: int, frame_idx: int, states_buffer: any = None
    ) -> dict[str, np.ndarray]:
        lerobot_feature = self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            LEROBOT_FEATURE_KEY
        ]

        sub_states_datas: list[np.ndarray] = []
        for state_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            SUB_STATE_KEY
        ]:
            args_dict = state_config[ARGS_KEY]
            sub_states_data = self._get_frame_sub_states(
                task_path=task_path,
                ep_idx=ep_idx,
                frame_idx=frame_idx,
                args_dict=args_dict,
                sub_states_buffer=states_buffer,
            )
            if CONVERT_FUNC_KEY in state_config:
                if state_config[CONVERT_FUNC_KEY] in spatial_covertor_funcs:
                    if spatial_covertor_funcs[state_config[CONVERT_FUNC_KEY]]:
                        sub_states_data = spatial_covertor_funcs[state_config[CONVERT_FUNC_KEY]](
                            sub_states_data
                        ).astype(np.float32)

            sub_states_datas.append(sub_states_data)

        result = {lerobot_feature: np.concatenate(sub_states_datas).astype(np.float32)}
        
        # 🔍 调试：打印总维度
        if self.logger and frame_idx == 0:
            total_dims = result[lerobot_feature].shape[0]
            self.logger.info(f"🔍 总计 observation.state: {total_dims}维 (合并了{len(sub_states_datas)}个sub_state)")
        
        return result

    def _get_frame_actions(
        self, task_path: Path, ep_idx: int, frame_idx: int, actions_buffer: any = None
    ) -> dict[str, np.ndarray]:
        lerobot_feature = self.converter_config[FEATURES_KEY][ACTION_KEY][LEROBOT_FEATURE_KEY]
        
        # Apply timeline_offset to get action from a future frame
        timeline_offset = self.converter_config[FEATURES_KEY][ACTION_KEY].get(TIMELINE_OFFSET_KEY, 0)
        action_frame_idx = frame_idx + timeline_offset
        
        sub_actions_datas: list[np.ndarray] = []
        for action_config in self.converter_config[FEATURES_KEY][ACTION_KEY][SUB_ACTION_KEY]:
            args_dict = action_config[ARGS_KEY]
            sub_actions_data = self._get_frame_sub_actions(
                task_path=task_path,
                ep_idx=ep_idx,
                frame_idx=action_frame_idx,  # Use offset frame index for actions
                args_dict=args_dict,
                sub_actions_buffer=actions_buffer,
            )

            if CONVERT_FUNC_KEY in action_config:
                if action_config[CONVERT_FUNC_KEY] in spatial_covertor_funcs:
                    if spatial_covertor_funcs[action_config[CONVERT_FUNC_KEY]]:
                        sub_actions_data = spatial_covertor_funcs[action_config[CONVERT_FUNC_KEY]](
                            sub_actions_data
                        ).astype(np.float32)
            sub_actions_datas.append(sub_actions_data)

        return {lerobot_feature: np.concatenate(sub_actions_datas).astype(np.float32)}

    def _get_lerobot_datas(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        images_buffer: any = None,
        states_buffer: any = None,
        actions_buffer: any = None,
    ) -> dict[str, np.ndarray]:
        return {
            **self._get_frame_images(
                task_path=task_path,
                ep_idx=ep_idx,
                frame_idx=frame_idx,
                images_buffer=images_buffer,
            ),
            **self._get_frame_states(
                task_path=task_path,
                ep_idx=ep_idx,
                frame_idx=frame_idx,
                states_buffer=states_buffer,
            ),
            **self._get_frame_actions(
                task_path=task_path,
                ep_idx=ep_idx,
                frame_idx=frame_idx,
                actions_buffer=actions_buffer,
            ),
        }

    def _prepare_episode_buffers(self, task_path: Path, ep_idx: int, is_test: bool = False) -> tuple[any, any, any]:
        """Prepare all buffers for an episode.
        
        智能调用子类方法：如果子类方法支持 is_test 参数则传入，否则只传基本参数。
        这样保证了向后兼容性 - 旧的子类实现不需要修改。
        """
        import inspect
        from collections.abc import Callable
        
        # 检查子类方法是否接受 is_test 参数
        images_method = self._prepare_episode_images_buffer
        states_method = self._prepare_episode_states_buffer
        actions_method = self._prepare_episode_actions_buffer
        
        # 智能调用：检查方法签名
        def smart_call(method: Callable, task_path: Path, ep_idx: int, is_test: bool) -> any:
            sig = inspect.signature(method)
            if 'is_test' in sig.parameters:
                return method(task_path=task_path, ep_idx=ep_idx, is_test=is_test)
            return method(task_path=task_path, ep_idx=ep_idx)
        
        return (
            smart_call(images_method, task_path, ep_idx, is_test),
            smart_call(states_method, task_path, ep_idx, is_test),
            smart_call(actions_method, task_path, ep_idx, is_test),
        )

    def _get_lerobot_image_features(self) -> dict:
        lerobot_image_features = {}
        for image_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]:
            lerobot_feature_key = image_config[LEROBOT_FEATURE_KEY]
            image_feature = {}
            image_feature[DTYPE_KEY] = image_config[DTYPE_KEY]
            image_feature[SHAPE_KEY] = image_config[SHAPE_KEY]
            image_feature[NAME_KEY] = image_config[NAME_KEY]
            lerobot_image_features[lerobot_feature_key] = image_feature
        return lerobot_image_features

    def _get_lerobot_state_feature(self) -> dict:
        lerobot_state_feature = {}
        lerobot_feature_key = self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            LEROBOT_FEATURE_KEY
        ]
        state_feature = {}
        state_feature[DTYPE_KEY] = FLOAT32
        state_feature[SHAPE_KEY] = self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            SHAPE_KEY
        ]
        state_feature[NAME_KEY] = self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            NAME_KEY
        ]
        lerobot_state_feature[lerobot_feature_key] = state_feature
        return lerobot_state_feature

    def _get_lerobot_action_feature(self) -> dict:
        lerobot_action_feature = {}
        lerobot_feature_key = self.converter_config[FEATURES_KEY][ACTION_KEY][LEROBOT_FEATURE_KEY]
        action_feature = {}
        action_feature[DTYPE_KEY] = FLOAT32
        action_feature[SHAPE_KEY] = self.converter_config[FEATURES_KEY][ACTION_KEY][SHAPE_KEY]
        action_feature[NAME_KEY] = self.converter_config[FEATURES_KEY][ACTION_KEY][NAME_KEY]
        lerobot_action_feature[lerobot_feature_key] = action_feature
        return lerobot_action_feature

    def _get_lerobot_features(self) -> dict:
        return {
            **self._get_lerobot_image_features(),
            **self._get_lerobot_state_feature(),
            **self._get_lerobot_action_feature(),
        }

    def _create_lerobot_dataset(self) -> LeRobotDataset:
        """创建或加载LeRobot数据集（支持断点续转）
        
        🔥 断点续转功能：
        - 如果output_path已存在且包含有效数据集，则加载它（续转）
        - 否则创建新的数据集
        - 支持重试机制，处理并发创建的竞态条件
        
        Returns:
            LeRobotDataset: 创建或加载的数据集对象
        """
        import shutil
        import time
        from pathlib import Path
        
        output_path = Path(self.output_path)
        
        # 🆕 检查是否存在有效的数据集（断点续转）
        info_file = output_path / "meta" / "info.json"
        if output_path.exists() and info_file.exists():
            if self.logger:
                self.logger.info(f"🔄 检测到已存在的数据集: {output_path.name}")
                self.logger.info(f"   尝试加载以支持断点续转...")
            
            try:
                # 尝试加载已存在的数据集
                dataset = LeRobotDataset(
                    repo_id=self.repo_id,
                    root=self.output_path,
                    local_files_only=True  # 只使用本地文件，不从Hub下载
                )
                
                if self.logger:
                    self.logger.info(f"✅ 成功加载已存在的数据集")
                    self.logger.info(f"   当前已有 {len(dataset)} 个episodes")
                    self.logger.info(f"   将继续转换剩余episodes...")
                
                return dataset
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️  加载已存在数据集失败: {e}")
                    self.logger.warning(f"   将删除并重新创建数据集...")
                # 加载失败，继续下面的删除+重建流程
        
        # 最多重试3次（处理并发竞态条件）
        max_retries = 3
        for retry in range(max_retries):
            # 如果目录已存在，删除它（通常是之前失败的转换）
            if output_path.exists():
                if self.logger:
                    self.logger.warning(
                        f"⚠️  Output directory already exists: {output_path.name}\n"
                        f"   This is likely from a previous failed conversion.\n"
                        f"   🗑️  Removing old directory to retry... (attempt {retry + 1}/{max_retries})"
                    )
                try:
                    shutil.rmtree(output_path)
                    if self.logger:
                        self.logger.info(f"✅ Removed old directory: {output_path.name}")
                except Exception as e:
                    if retry == max_retries - 1:  # 最后一次重试失败
                        raise RuntimeError(
                            f"Failed to remove existing output directory after {max_retries} attempts: {output_path}\n"
                            f"Error: {e}\n"
                            f"Please manually delete this directory and retry."
                        ) from e
                    else:
                        if self.logger:
                            self.logger.warning(f"Failed to remove directory (attempt {retry + 1}), retrying in 2s...")
                        time.sleep(2)
                        continue
            
            # 尝试创建数据集
            try:
                return LeRobotDataset.create(
                    repo_id=self.repo_id,
                    features=self._get_lerobot_features(),
                    fps=self.fps,
                    robot_type=self.device_model,
                    root=self.output_path,
                    video_backend=self.video_backend,
                    image_writer_processes=self.image_writer_processes,
                    image_writer_threads=self.image_writer_threads,
                )
            except FileExistsError as e:
                # 并发创建导致的竞态条件
                if retry == max_retries - 1:  # 最后一次重试
                    raise RuntimeError(
                        f"Failed to create dataset after {max_retries} attempts due to concurrent access.\n"
                        f"Output path: {output_path}\n"
                        f"This indicates multiple clients are trying to convert the same task.\n"
                        f"Original error: {e}"
                    ) from e
                else:
                    if self.logger:
                        self.logger.warning(
                            f"⚠️  FileExistsError during creation (race condition?), "
                            f"retrying in 2s... (attempt {retry + 1}/{max_retries})"
                        )
                    time.sleep(2)
                    continue
        
        # 不应该到达这里
        raise RuntimeError("Unexpected error in _create_lerobot_dataset")

    def _get_episode_task(self, ep_idx: int) -> str:
        return ""

    def _gen_episode_frames(
        self,
        task_path: Path,
        ep_idx: int,
        images_buffer: any = None,
        states_buffer: any = None,
        actions_buffer: any = None,
    ) -> Iterable[dict]:
        total_frames = self._get_episode_frames_num(task_path=task_path, ep_idx=ep_idx)
        
        # Check if episode should be skipped (indicated by total_frames == -1)
        # This happens when data quality issues are detected and file is auto-moved to error/
        if total_frames == -1:
            if self.logger:
                self.logger.info(
                    f"⏭️  Skipping episode {ep_idx} at {task_path.name} "
                    f"(auto-moved to error/ due to data quality issues)"
                )
            return  # Return empty iterator to skip this episode
        
        # Get timeline_offset from action config to determine how many frames to generate
        timeline_offset = self.converter_config[FEATURES_KEY][ACTION_KEY].get(TIMELINE_OFFSET_KEY, 0)
        
        # When timeline_offset > 0, we need to stop earlier to avoid accessing frames beyond the episode
        # For example, if timeline_offset=1, we can only use frames 0 to total_frames-2,
        # because frame total_frames-1 would need to access frame total_frames (which doesn't exist)
        max_frame_idx = total_frames - timeline_offset if timeline_offset > 0 else total_frames
        
        for frame_idx in range(max_frame_idx):
            try:
                frame_data = self._gen_episode_frame(
                    task_path, ep_idx, frame_idx, images_buffer, states_buffer, actions_buffer
                )
            except (CriticalDataError, DataQualityError):
                # 🔧 容错机制：让 CriticalDataError 和 DataQualityError 直接传播
                # 上层的 _convert_episode_with_fault_tolerance 会捕获并跳过episode
                raise
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Error generating frame data: task_path={task_path}, "
                        f"episode={ep_idx}, frame={frame_idx}/{max_frame_idx}, "
                        f"timeline_offset={timeline_offset}. Error: {e}"
                    )
                raise RuntimeError(
                    f"Failed to generate frame at task_path={task_path}, "
                    f"episode={ep_idx}, frame={frame_idx}/{max_frame_idx} "
                    f"(timeline_offset={timeline_offset})"
                ) from e
            yield frame_data

        pass

    def _convert_episode_with_fault_tolerance(
        self,
        dataset: LeRobotDataset,
        task_path: Path,
        task: str,
        task_ep_idx: int,
        global_ep_idx: int,
        is_strict: bool,
        is_test: bool,
    ) -> tuple[int, int]:
        """转换单个episode，支持episode级容错
        
        重要：为了保持时序数据的连续性，任何单帧错误都会导致整个episode被跳过。
        这是因为跳过单帧会破坏observation-action的时间对齐关系。
        
        Args:
            dataset: LeRobot数据集对象
            task_path: 任务路径
            task: 任务名称
            task_ep_idx: 任务内episode索引
            global_ep_idx: 全局episode索引
            is_strict: 是否为严格模式
            is_test: 是否为测试模式
        
        Returns:
            (converted_frames, 0): 成功转换的帧数（跳过的帧数始终为0，因为要么全转要么全跳）
            
        Raises:
            ConfigError: 严格模式下遇到数据错误（表明配置可能有问题）
            CriticalDataError: 非严格模式下遇到数据错误（跳过整个episode）
        """
        # 🆕 容错：处理字段缺失和文件损坏的情况
        try:
            images_buffer, states_buffer, actions_buffer = self._prepare_episode_buffers(
                task_path, task_ep_idx, is_test=is_test
            )
        except KeyError as e:
            # H5字段缺失或配置不匹配
            if is_strict:
                # 严格模式：视为配置错误
                raise ConfigError(
                    f"严格模式下检测到字段缺失（可能是配置错误）:\n"
                    f"  Episode: {global_ep_idx} (task episode: {task_ep_idx})\n"
                    f"  Task: {task}\n"
                    f"  Error: {e}\n"
                    f"\n💡 在前{self.strict_episodes}个episode中发现此问题，"
                    f"可能是配置错误而非数据问题"
                ) from e
            else:
                # 非严格模式：跳过这个episode
                if self.logger:
                    self.logger.warning(
                        f"⚠️  Episode {global_ep_idx} (task episode: {task_ep_idx}) 字段缺失，跳过:\n"
                        f"   Task: {task}\n"
                        f"   Reason: {str(e)[:200]}..."  # 只显示前200字符
                    )
                # 返回(0, 0)表示跳过整个episode
                return 0, 0
        except Exception as e:
            # MCAP文件损坏或其他buffer准备错误
            if 'RecordLengthLimitExceeded' in type(e).__name__ or 'mcap' in str(type(e)).lower():
                if self.logger:
                    self.logger.warning(
                        f"⚠️  Episode {global_ep_idx} (task episode: {task_ep_idx}) 文件损坏，跳过:\n"
                        f"   Task: {task}\n"
                        f"   Error: {type(e).__name__}: {str(e)[:200]}..."
                    )
                # 返回(0, 0)表示跳过整个episode
                return 0, 0
            else:
                # 其他未知错误，重新抛出
                raise
        
        converted_frames = 0
        
        for frame_data in self._gen_episode_frames(
            task_path, task_ep_idx, images_buffer, states_buffer, actions_buffer
        ):
            frame_idx = frame_data[FRAME_IDX_KEY]
            
            try:
                lerobot_datas = self._get_lerobot_datas(
                    task_path=task_path,
                    ep_idx=task_ep_idx,
                    frame_idx=frame_idx,
                    images_buffer=images_buffer,
                    states_buffer=states_buffer,
                    actions_buffer=actions_buffer,
                )

                # 修改：测试模式也保存数据，只是加日志标记
                dataset.add_frame(frame=lerobot_datas, task=task)
                # if is_test:
                #     self.logger.info(f"🧪 测试模式：已将帧 {frame_idx} 添加到数据集（task: {task}）")
                
                converted_frames += 1
                
            except DataQualityError as e:
                # 数据质量问题：
                # - 严格模式：升级为配置错误，停止整个转换
                # - 非严格模式：升级为严重数据错误，跳过整个episode
                if is_strict:
                    raise ConfigError(
                        f"严格模式下检测到数据质量问题（可能是配置错误）:\n"
                        f"  Episode: {global_ep_idx} (task episode: {task_ep_idx})\n"
                        f"  Frame: {frame_idx}\n"
                        f"  Error: {e}\n"
                        f"\n💡 在前{self.strict_episodes}个episode中发现此问题，"
                        f"可能是配置错误而非数据问题"
                    ) from e
                else:
                    # 非严格模式：跳过整个episode以保持时序连续性
                    raise CriticalDataError(
                        f"Episode {global_ep_idx} 数据质量问题，跳过整个episode:\n"
                        f"  任务: {task}\n"
                        f"  Episode索引: {task_ep_idx}\n"
                        f"  问题帧: {frame_idx}\n"
                        f"  错误: {e}\n"
                        f"\n⚠️  为保持时序连续性，不能跳过单帧，必须跳过整个episode"
                    ) from e
                
            except ValueError as e:
                # ValueError: 图像尺寸不匹配等格式验证错误
                # 跳过整个episode（数据采集过程中设备配置可能发生变化）
                if self.logger:
                    self.logger.warning(
                        f"⚠️  Episode {global_ep_idx} (task_ep: {task_ep_idx}, frame {frame_idx}) ValueError (likely image size mismatch):\n"
                        f"   {e}\n"
                        f"   Skipping entire episode. Can be traced via episode_source_mapping.json"
                    )
                
                # 🆕 抛出CriticalDataError，标记为"data_quality"（纯数据质量问题，不计入配置错误率）
                raise CriticalDataError(
                    f"Episode {global_ep_idx} 图像尺寸不匹配，跳过整个episode:\n"
                    f"  任务: {task}\n"
                    f"  Episode索引: {task_ep_idx}\n"
                    f"  问题帧: {frame_idx}\n"
                    f"  错误: {e}\n"
                    f"\n💡 这通常是数据采集过程中设备配置变化导致的",
                    error_category="data_quality"  # 标记为纯数据质量问题
                ) from e
                
            except Exception as e:
                # 未分类的异常：在严格模式下作为配置错误处理
                if is_strict:
                    raise ConfigError(
                        f"严格模式下遇到未预期的错误:\n"
                        f"  Episode: {global_ep_idx} (task episode: {task_ep_idx})\n"
                        f"  Frame: {frame_idx}\n"
                        f"  Error type: {type(e).__name__}\n"
                        f"  Error: {e}"
                    ) from e
                else:
                    # 非严格模式：也升级为CriticalDataError
                    raise CriticalDataError(
                        f"Episode {global_ep_idx} 遇到错误，跳过整个episode:\n"
                        f"  任务: {task}\n"
                        f"  Episode索引: {task_ep_idx}\n"
                        f"  问题帧: {frame_idx}\n"
                        f"  错误类型: {type(e).__name__}\n"
                        f"  错误: {e}"
                    ) from e
        
        # 如果成功遍历所有帧，返回转换的帧数
        # 注意：skipped_frames始终为0，因为我们不支持跳过单帧
        return converted_frames, 0

    def _get_conversion_report(self) -> dict:
        """生成转换报告"""
        stats = self._conversion_stats
        total = stats['total_episodes']
        successful = stats['successful_episodes']
        
        return {
            'dataset': str(self.dataset_path.name),
            'total_episodes_attempted': total,
            'successful_episodes': successful,
            'skipped_episodes': stats['skipped_episodes'],
            'total_frames_converted': stats['total_frames'],
            'total_frames_skipped': stats['skipped_frames'],
            'success_rate': successful / total if total > 0 else 0,
            'skip_details': stats['skip_details'][-50:],  # 只保留最近50条
        }

    def convert(self, is_test: bool = False) -> Iterable[tuple[str, int, int]]:
        """转换数据集，使用智能容错机制（支持断点续转）
        
        Args:
            is_test: 是否为测试模式（只转换第一个episode）
        
        Yields:
            (task, task_ep_idx, global_ep_idx): 成功转换的episode信息
        
        Raises:
            ConfigError: 检测到配置错误（前N个episode高失败率）
        """
        if is_test:
            # 1. 定义基础temp目录路径（优先使用当前工作目录下的temp）
            base_temp_dir = Path.cwd() / "temp"
            
            # 2. 匹配 temp/ 下所有 robocoin_* 格式的目录/文件
            if base_temp_dir.exists():
                # 构建匹配模式：temp/robocoin_*
                cache_pattern = str(base_temp_dir / "robocoin_*")
                
                # 查找所有匹配的路径
                cache_paths = glob.glob(cache_pattern)
                
                for cache_path in cache_paths:
                    cache_path = Path(cache_path)
                    try:
                        # 删除目录（递归删除）
                        if cache_path.is_dir():
                            shutil.rmtree(cache_path, ignore_errors=True)
                            self.logger.info(f"🧹 测试模式：已清理缓存目录 {cache_path}")
                        # 删除文件
                        elif cache_path.is_file():
                            cache_path.unlink(missing_ok=True)
                            self.logger.info(f"🧹 测试模式：已清理缓存文件 {cache_path}")
                    except Exception as e:
                        self.logger.warning(f"⚠️  测试模式：清理缓存 {cache_path} 失败: {e}")

        # 🆕 初始化self.lerobot_dataset，确保清理代码可以访问
        self.lerobot_dataset = None
        
        # 修改1：测试模式也创建数据集（不再设置dataset=None）
        dataset = self._create_lerobot_dataset()
        self.lerobot_dataset = dataset  # 🆕 保存到self，用于清理
        
        # 🔄 断点续转：获取已存在的episodes数量
        existing_episodes = len(dataset) if dataset else 0
        if existing_episodes > 0 and self.logger:
            self.logger.info(f"🔄 检测到 {existing_episodes} 个已存在的episodes，将跳过它们")
        
        global_ep_idx = existing_episodes  # 从已存在的episodes数量开始计数
        original_ep_idx = 0  # 原始数据中的全局episode索引（包含所有episode，包括跳过的）
        task_stats = {}  # 每个task的统计信息
        
        # 🆕 Test模式：限制处理的tasks数量
        tasks_to_process = list(self.path_task_dict.items())
        if is_test:
            max_test_tasks = 2  # Test模式只处理前2个tasks
            tasks_to_process = tasks_to_process[:max_test_tasks]
            if self.logger:
                self.logger.info(
                    f"🧪 Test mode: processing only {len(tasks_to_process)} tasks "
                    f"out of {len(self.path_task_dict)} total tasks"
                )
        
        try:
            for task_path, task in tasks_to_process:
                episodes_num = self._get_task_episodes_num(task_path)
                if is_test:
                    episodes_num = 2 # Test模式每个task只处理1个episode
                
                # 初始化任务统计
                task_stats[task] = {
                    'attempted': 0,
                    'successful': 0,
                    'skipped': 0,
                    'skipped_frames': 0,
                }
                
                for task_ep_idx in range(episodes_num):
                    # 🔄 断点续转：跳过已经转换的episodes
                    if original_ep_idx < existing_episodes:
                        if self.logger and original_ep_idx == 0:
                            self.logger.info(f"⏭️  跳过已转换的 {existing_episodes} 个episodes...")
                        original_ep_idx += 1
                        continue
                    
                    # 🧪 测试模式不使用严格模式（允许跳过字段缺失的episode）
                    # 正式模式：前N个episode使用严格模式（检测配置错误）
                    is_strict = global_ep_idx < self.strict_episodes and not is_test
                    
                    # 更新统计
                    self._conversion_stats['total_episodes'] += 1
                    task_stats[task]['attempted'] += 1
                    
                    try:
                        converted_frames, skipped_frames = self._convert_episode_with_fault_tolerance(
                            dataset=dataset,  # 测试模式也传入dataset
                            task_path=task_path,
                            task=task,
                            task_ep_idx=task_ep_idx,
                            global_ep_idx=global_ep_idx,
                            is_strict=is_strict,
                            is_test=is_test,
                        )
                        
                        # Episode完全为空，跳过
                        if converted_frames == 0 and skipped_frames == 0:
                            self._conversion_stats['skipped_episodes'] += 1
                            task_stats[task]['skipped'] += 1
                            
                            skip_reason = 'Empty episode or data quality issue'
                            self._conversion_stats['skip_details'].append({
                                'episode': original_ep_idx,
                                'task': task,
                                'task_episode': task_ep_idx,
                                'reason': skip_reason,
                                'skipped_entire_episode': True,
                            })
                            
                            # 🆕 记录跳过的episode到mapping（使用original_ep_idx）
                            source_files = self._get_episode_source_files(task_path, task_ep_idx)
                            self.episode_source_mapping[original_ep_idx] = {
                                "task": task,
                                "task_path": str(task_path),
                                "task_ep_idx": task_ep_idx,
                                "original_ep_idx": original_ep_idx,
                                "global_ep_idx": None,  # 未转换，无LeRobot索引
                                "status": "skipped",
                                "skip_reason": skip_reason,
                                "source_files": source_files,
                                "converted_frames": 0,
                                "skipped_frames": 0,
                            }
                            
                            self.logger.info(
                                f"⏭️ 跳过 episode {original_ep_idx} "
                                f"(task: {task}, task_ep: {task_ep_idx}): 空episode"
                            )
                            original_ep_idx += 1  # 🆕 original_ep_idx继续递增
                            continue
                        
                        # 更新统计
                        self._conversion_stats['successful_episodes'] += 1
                        self._conversion_stats['total_frames'] += converted_frames
                        self._conversion_stats['skipped_frames'] += skipped_frames
                        task_stats[task]['successful'] += 1
                        task_stats[task]['skipped_frames'] += skipped_frames
                        
                        if skipped_frames > 0:
                            self._conversion_stats['skip_details'].append({
                                'episode': global_ep_idx,
                                'task': task,
                                'task_episode': task_ep_idx,
                                'converted_frames': converted_frames,
                                'skipped_frames': skipped_frames,
                            })
                        
                        # 保存episode（带NAS错误重试）
                        # 修改2：测试模式也保存episode
                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                dataset.save_episode()
                                if is_test:
                                    self.logger.info(f"🧪 测试模式：已保存episode {global_ep_idx} 到数据集")
                                break  # 成功则退出重试
                            except OSError as e:
                                # Stale file handle (Errno 116) 或其他NAS错误
                                if e.errno == 116 or 'Stale file handle' in str(e):
                                    if retry < max_retries - 1:
                                        if self.logger:
                                            self.logger.warning(
                                                f"⚠️  NAS文件句柄错误 (Stale file handle)，"
                                                f"重试 {retry + 1}/{max_retries}... "
                                                f"Episode {global_ep_idx}, Task: {task}"
                                            )
                                        import time
                                        time.sleep(2 ** retry)  # 指数退避: 1s, 2s, 4s
                                        continue
                                    else:
                                        # 最后一次重试也失败
                                        error_msg = (
                                            f"❌ NAS文件系统错误 (Stale file handle)\n"
                                            f"   Episode: {global_ep_idx} (task episode: {task_ep_idx})\n"
                                            f"   Task: {task}\n"
                                            f"   重试 {max_retries} 次后仍然失败\n"
                                            f"   💡 建议:\n"
                                            f"      1. 检查NAS网络连接\n"
                                            f"      2. 重新挂载NAS: sudo umount && sudo mount\n"
                                            f"      3. 清除转换记录后重试该数据集\n"
                                            f"   Error: {e}"
                                        )
                                        if self.logger:
                                            self.logger.error(error_msg)
                                        raise RuntimeError(error_msg) from e
                                else:
                                    # 其他OSError，直接抛出
                                    raise
                        
                        # 🔧 收集源文件映射信息（成功转换的episode）
                        source_files = self._get_episode_source_files(task_path, task_ep_idx)
                        self.episode_source_mapping[original_ep_idx] = {
                            "task": task,
                            "task_path": str(task_path),
                            "task_ep_idx": task_ep_idx,
                            "original_ep_idx": original_ep_idx,  # 🆕 原始索引
                            "global_ep_idx": global_ep_idx,  # 🆕 LeRobot索引
                            "status": "converted",  # 🆕 状态标记
                            "source_files": source_files,
                            "converted_frames": converted_frames,
                            "skipped_frames": skipped_frames,
                        }
                        
                        # 🆕 清理episode缓存（MCAP等大文件格式需要释放内存）
                        # 优先调用更全面的资源清理方法（MCAP converter实现）
                        if hasattr(self, '_cleanup_episode_resources'):
                            self._cleanup_episode_resources()
                        elif hasattr(self, '_clear_episode_cache'):
                            self._clear_episode_cache()
                        
                        # 检查失败率（在严格阶段结束时）
                        if global_ep_idx == self.strict_episodes - 1:
                            self._check_failure_rate_threshold(task_stats)
                        
                        yield (task, task_ep_idx, global_ep_idx)
                        global_ep_idx += 1  # LeRobot索引递增
                        original_ep_idx += 1  # 🆕 原始索引递增
                        
                    except CriticalDataError as e:
                        # 严重数据错误：跳过整个episode
                        self._conversion_stats['skipped_episodes'] += 1
                        task_stats[task]['skipped'] += 1
                        
                        # 🆕 提取错误类别（用于区分数据质量问题和配置错误）
                        error_category = getattr(e, 'error_category', 'potential_config')
                        
                        skip_reason = str(e)
                        self._conversion_stats['skip_details'].append({
                            'episode': original_ep_idx,
                            'task': task,
                            'task_episode': task_ep_idx,
                            'reason': skip_reason,
                            'skipped_entire_episode': True,
                            'error_category': error_category,  # 🆕 记录错误类别
                        })
                        
                        # 🆕 记录跳过的episode到mapping
                        source_files = self._get_episode_source_files(task_path, task_ep_idx)
                        self.episode_source_mapping[original_ep_idx] = {
                            "task": task,
                            "task_path": str(task_path),
                            "task_ep_idx": task_ep_idx,
                            "original_ep_idx": original_ep_idx,
                            "global_ep_idx": None,  # 未转换，无LeRobot索引
                            "status": "skipped",
                            "skip_reason": skip_reason,
                            "source_files": source_files,
                            "converted_frames": 0,
                            "skipped_frames": 0,
                        }
                        
                        self.logger.warning(
                            f"⏭️ 跳过 episode {original_ep_idx} "
                            f"(task: {task}, task_ep: {task_ep_idx}): {e}"
                        )
                        
                        # 🔥 清理内存（尤其对MCAP大文件很重要）
                        if hasattr(self, '_cleanup_episode_resources'):
                            self._cleanup_episode_resources()
                        elif hasattr(self, '_clear_episode_cache'):
                            self._clear_episode_cache()
                        
                        original_ep_idx += 1  # 🆕 original_ep_idx继续递增
                        continue
                        
                    except ConfigError:
                        # 配置错误：立即停止（仍然需要清理资源）
                        if hasattr(self, '_cleanup_episode_resources'):
                            self._cleanup_episode_resources()
                        elif hasattr(self, '_clear_episode_cache'):
                            self._clear_episode_cache()
                        self.logger.error("检测到配置错误，停止转换")
                        raise
                        
                    except Exception as e:
                        # 其他未处理的异常（清理资源后抛出）
                        if hasattr(self, '_cleanup_episode_resources'):
                            self._cleanup_episode_resources()
                        elif hasattr(self, '_clear_episode_cache'):
                            self._clear_episode_cache()
                        self.logger.error(
                            f"处理episode时发生未预期的错误: "
                            f"task={task}, task_ep={task_ep_idx}, global_ep={global_ep_idx}. "
                            f"Error: {e}"
                        )
                        raise RuntimeError(
                            f"Failed to process episode {task_ep_idx} (global: {global_ep_idx}) "
                            f"at task {task}"
                        ) from e
        
        finally:
            # ========== 核心新增：转换完成后的内存清理 ==========
            self.logger.info("🧹 开始清理转换完成后的内存资源...")
            
            # 1. 停止LeRobot数据集的image writer进程池
            if hasattr(self, 'lerobot_dataset') and self.lerobot_dataset is not None:
                try:
                    self.lerobot_dataset.stop_image_writer()
                    self.logger.info("✅ 已停止image writer进程池")
                except Exception as e:
                    self.logger.warning(f"⚠️  停止image writer失败: {e}")
            
            # 2. 清理各类缓冲区
            if hasattr(self, '_cleanup_episode_resources'):
                try:
                    self._cleanup_episode_resources()
                    self.logger.info("✅ 已清理episode资源缓冲区")
                except Exception as e:
                    self.logger.warning(f"⚠️  清理episode资源失败: {e}")
            
            # 3. 手动触发Python垃圾回收
            import gc
            collected = gc.collect()
            self.logger.info(f"✅ 垃圾回收完成，释放了 {collected} 个对象")
            
            # 4. 清空大的内存对象
            if hasattr(self, 'episode_source_mapping') and len(self.episode_source_mapping) > 1000:
                # 保留关键信息，清空大字典
                self.episode_source_mapping = {}
                self.logger.info("✅ 已清空episode_source_mapping大字典")
            
            # 5. 重置统计信息（如果不需要保留）
            self._conversion_stats['skip_details'] = []
            self.logger.info("✅ 已重置skip_details统计信息")
            
            self.logger.info("✅ 所有内存清理操作完成")
            # ========== 内存清理结束 ==========
        
        # 转换完成后打印统计信息
        self._print_conversion_summary(task_stats)
        
        # 修改3：测试模式也保存映射文件
        if is_test:
            self.logger.info("🧪 测试模式：保存episode映射文件...")
            self.save_episode_source_mapping()
            self.save_original_data_paths()


    def _check_failure_rate_threshold(self, task_stats: dict) -> None:
        """检查失败率是否超过阈值
        
        🆕 只统计"potential_config"类型的错误，纯数据质量问题（如图像尺寸不匹配）不计入失败率
        
        Args:
            task_stats: 任务统计信息
            
        Raises:
            ConfigError: 失败率超过阈值
        """
        total_attempted = self._conversion_stats['total_episodes']
        total_skipped = self._conversion_stats['skipped_episodes']
        
        # 🆕 只统计可能是配置错误的失败episodes
        skip_details = self._conversion_stats['skip_details']
        config_related_failures = [
            f for f in skip_details 
            if f.get('error_category', 'potential_config') == 'potential_config'
        ]
        data_quality_failures = [
            f for f in skip_details 
            if f.get('error_category', 'potential_config') == 'data_quality'
        ]
        
        config_failure_count = len(config_related_failures)
        data_quality_count = len(data_quality_failures)
        
        # 🆕 使用config相关失败数计算失败率
        failure_rate = config_failure_count / total_attempted if total_attempted > 0 else 0
        
        if failure_rate > self.failure_threshold:
            # 收集详细错误信息（只显示config相关的）
            recent_failures = config_related_failures[-self.strict_episodes:]
            
            raise ConfigError(
                f"前{self.strict_episodes}个episode失败率过高，可能存在配置错误:\n"
                f"  尝试转换: {total_attempted} episodes\n"
                f"  跳过（配置相关）: {config_failure_count} episodes\n"
                f"  跳过（数据质量）: {data_quality_count} episodes (不计入失败率)\n"
                f"  总跳过: {total_skipped} episodes\n"
                f"  配置错误率: {failure_rate:.1%}\n"
                f"  阈值: {self.failure_threshold:.1%}\n"
                f"\n最近失败的episodes (配置相关):\n" +
                "\n".join(
                    f"  - Episode {f['episode']} (task: {f['task']}): {f.get('reason', 'Unknown')}"
                    for f in recent_failures
                ) +
                f"\n\n💡 建议：\n"
                f"  1. 检查配置文件中的字段路径是否正确\n"
                f"  2. 使用 diagnose_converter_config.py 诊断配置\n"
                f"  3. 检查数据集格式是否与配置匹配\n"
                f"\n📊 注意：{data_quality_count}个episodes因纯数据质量问题被跳过（如图像尺寸不一致），这不算配置错误"
            )

    def _print_conversion_summary(self, task_stats: dict) -> None:
        """打印转换统计摘要"""
        stats = self._conversion_stats
        
        self.logger.info("="*70)
        self.logger.info("转换完成 - 统计摘要")
        self.logger.info("="*70)
        self.logger.info(f"总Episodes: {stats['total_episodes']}")
        self.logger.info(f"成功: {stats['successful_episodes']}")
        self.logger.info(f"跳过: {stats['skipped_episodes']}")
        self.logger.info(f"成功率: {stats['successful_episodes']/stats['total_episodes']*100:.1f}%")
        self.logger.info(f"总帧数: {stats['total_frames']}")
        self.logger.info(f"跳过帧数: {stats['skipped_frames']}")
        
        if task_stats:
            self.logger.info("\n任务详情:")
            for task, task_stat in task_stats.items():
                success_rate = (task_stat['successful'] / task_stat['attempted'] * 100 
                               if task_stat['attempted'] > 0 else 0)
                self.logger.info(
                    f"  {task}: {task_stat['successful']}/{task_stat['attempted']} "
                    f"({success_rate:.1f}%), 跳过帧: {task_stat['skipped_frames']}"
                )
        
        if stats['skip_details']:
            skipped_count = len([d for d in stats['skip_details'] 
                                if d.get('skipped_entire_episode')])
            self.logger.info(f"\n完全跳过的episodes: {skipped_count}")
        
        self.logger.info("="*70)

    def save_episode_source_mapping(self, mapping_filename: str = "episode_source_mapping.json") -> None:
        """保存 episode 源文件映射到 JSON 文件
        
        🆕 新版本同时记录成功转换和跳过的episodes
        
        Args:
            mapping_filename: 映射文件名，默认为 "episode_source_mapping.json"
        
        Note:
            映射文件保存在 output_path 目录下，与 meta/info.json 同级
        """
        import json
        from datetime import datetime
        
        if not self.episode_source_mapping:
            self.logger.warning("没有 episode 映射信息可保存")
            return
        
        # 🆕 统计转换和跳过的episodes
        converted_episodes = []
        skipped_episodes = []
        
        for original_idx in sorted(self.episode_source_mapping.keys()):
            episode_info = self.episode_source_mapping[original_idx]
            if episode_info["status"] == "converted":
                converted_episodes.append(episode_info)
            elif episode_info["status"] == "skipped":
                skipped_episodes.append(episode_info)
        
        # 🆕 构建增强的映射文件数据
        mapping_data = {
            "dataset_info": {
                "source_dataset_path": str(self.dataset_path),
                "output_dataset_path": str(self.output_path),
                "repo_id": self.repo_id,
                "device_model": self.device_model or "unknown",
                "conversion_date": datetime.now().isoformat(),
                "fps": self.fps,
                # 🆕 详细统计
                "total_original_episodes": len(self.episode_source_mapping),
                "total_converted_episodes": len(converted_episodes),
                "total_skipped_episodes": len(skipped_episodes),
                "skipped_episode_indices": [ep["original_ep_idx"] for ep in skipped_episodes],
            },
            "converted_episodes": [],  # 🆕 成功转换的episodes
            "skipped_episodes": [],    # 🆕 跳过的episodes
        }
        
        # 🆕 添加成功转换的episodes（按LeRobot global_ep_idx排序）
        for episode_info in sorted(converted_episodes, key=lambda x: x["global_ep_idx"]):
            mapping_data["converted_episodes"].append({
                "global_episode_index": episode_info["global_ep_idx"],  # LeRobot索引
                "original_episode_index": episode_info["original_ep_idx"],  # 原始索引
                "task": episode_info["task"],
                "task_path": episode_info["task_path"],
                "task_episode_index": episode_info["task_ep_idx"],
                "converted_frames": episode_info["converted_frames"],
                "skipped_frames": episode_info["skipped_frames"],
                "source_files": episode_info["source_files"],
            })
        
        # 🆕 添加跳过的episodes（按原始索引排序）
        for episode_info in sorted(skipped_episodes, key=lambda x: x["original_ep_idx"]):
            mapping_data["skipped_episodes"].append({
                "original_episode_index": episode_info["original_ep_idx"],
                "task": episode_info["task"],
                "task_path": episode_info["task_path"],
                "task_episode_index": episode_info["task_ep_idx"],
                "skip_reason": episode_info.get("skip_reason", "Unknown"),
                "source_files": episode_info["source_files"],
            })
        
        # 保存到文件
        mapping_file_path = self.output_path / mapping_filename
        with open(mapping_file_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Episode 源文件映射已保存: {mapping_file_path}")
        self.logger.info(f"   - 总 episodes: {len(self.episode_source_mapping)}")
        self.logger.info(f"   - 成功转换: {len(converted_episodes)}")
        self.logger.info(f"   - 跳过: {len(skipped_episodes)}")
        self.logger.info(f"   - 映射文件: {mapping_filename}")
    
    def save_original_data_paths(self, mapping_filename: str = "original_data_paths.json") -> None:
        """保存原始数据文件的绝对路径映射到 JSON 文件
        
        🆕 新增功能：记录所有源文件的绝对路径，用于数据溯源
        
        Args:
            mapping_filename: 映射文件名，默认为 "original_data_paths.json"
        
        Note:
            映射文件保存在 output_path 目录下，与 meta/info.json 同级
            此文件与 episode_source_mapping.json 互补，专注于绝对路径信息
        """
        import json
        from datetime import datetime
        
        if not self.episode_source_mapping:
            self.logger.warning("没有 episode 映射信息可保存")
            return
        
        # 🆕 构建绝对路径映射数据
        paths_data = {
            "dataset_info": {
                "source_dataset_path": str(self.dataset_path.absolute()),
                "output_dataset_path": str(self.output_path.absolute()),
                "repo_id": self.repo_id,
                "device_model": self.device_model or "unknown",
                "extraction_date": datetime.now().isoformat(),
            },
            "episode_paths": []
        }
        
        # 🆕 为每个episode提取绝对路径信息
        for original_idx in sorted(self.episode_source_mapping.keys()):
            episode_info = self.episode_source_mapping[original_idx]
            source_files = episode_info.get("source_files", {})
            
            # 构建episode路径记录
            episode_path_record = {
                "original_episode_index": episode_info["original_ep_idx"],
                "task": episode_info["task"],
                "task_path": episode_info["task_path"],
                "status": episode_info["status"],
            }
            
            # 添加LeRobot索引（如果已转换）
            if episode_info["status"] == "converted":
                episode_path_record["global_episode_index"] = episode_info["global_ep_idx"]
            
            # 🆕 提取所有绝对路径
            absolute_paths = {}
            
            # 处理各种数据格式的路径
            if "absolute_path" in source_files:
                absolute_paths["primary"] = source_files["absolute_path"]
            
            if "h5_absolute_path" in source_files:
                absolute_paths["h5_file"] = source_files["h5_absolute_path"]
            
            if "video_files" in source_files:
                absolute_paths["videos"] = [
                    {"camera": v["camera"], "path": v["absolute_path"]}
                    for v in source_files["video_files"]
                ]
            
            # 添加格式信息
            if "format" in source_files:
                episode_path_record["data_format"] = source_files["format"]
            
            episode_path_record["absolute_paths"] = absolute_paths
            
            paths_data["episode_paths"].append(episode_path_record)
        
        # 保存到文件
        paths_file_path = self.output_path / mapping_filename
        with open(paths_file_path, 'w', encoding='utf-8') as f:
            json.dump(paths_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ 原始数据绝对路径映射已保存: {paths_file_path}")
        self.logger.info(f"   - 总 episodes: {len(paths_data['episode_paths'])}")
        self.logger.info(f"   - 映射文件: {mapping_filename}")

    def get_episodes_num(self) -> int:
        return sum(self._get_task_episodes_num(task) for task in self.path_task_dict.keys())


class LerobotFormatConverterFactory:
    @staticmethod
    def create_converter(
        dataset_path: Path,
        device_model: str,
        output_path: Path,
        converter_config: dict,
        converter_module_path: str,
        converter_class_name: str,
        repo_id: str,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
        logger: logging.Logger | None = None,
        converter_log_dir: Path | None = None,
        strict_episodes: int = 3,
        failure_threshold: float = 0.8,
        auto_reencode: bool = False,
    ) -> LerobotFormatConverter:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        # Create logger from converter_log_dir if provided and logger is None
        if logger is None and converter_log_dir is not None:
            from robocoin_dataset.utils.logger import setup_logger

            logger = setup_logger(
                name="lerobot_format_converter", log_dir=converter_log_dir, level=logging.INFO
            )

        module = importlib.import_module(converter_module_path)
        convertor_class = getattr(module, converter_class_name)
        
        # 构建基础参数
        init_kwargs = {
            'dataset_path': dataset_path,
            'output_path': output_path,
            'converter_config': converter_config,
            'repo_id': repo_id,
            'device_model': device_model,
            'logger': logger,
            'video_backend': video_backend,
            'image_writer_processes': image_writer_processes,
            'image_writer_threads': image_writer_threads,
        }
        
        # 检查子类是否支持新的容错参数（向后兼容）
        import inspect
        sig = inspect.signature(convertor_class.__init__)
        
        if 'strict_episodes' in sig.parameters:
            init_kwargs['strict_episodes'] = strict_episodes
        if 'failure_threshold' in sig.parameters:
            init_kwargs['failure_threshold'] = failure_threshold
        if 'auto_reencode' in sig.parameters:
            init_kwargs['auto_reencode'] = auto_reencode
        
        return convertor_class(**init_kwargs)
