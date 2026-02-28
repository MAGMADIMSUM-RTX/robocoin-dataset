"""
Constants used across the robocoin_datasets.format_convertor.tolerobot module.

This module defines various constants, enums, and configuration values used
throughout the robocoin datasets functionality including file paths, patterns,
and platform-specific settings.
"""

FPS = "fps"

FEATURES_KEY = "features"

OBSERVATION_KEY = "observation"

IMAGE_KEY = "images"

DEFAULT_IMAGE_SHAPE_NAMES = ("height", "width", "channels")

STATE_KEY = "state"

SUB_STATE_KEY = "sub_state"

ACTION_KEY = "action"

SUB_ACTION_KEY = "sub_action"

TIMELINE_OFFSET_KEY = "timeline_offset"

CAM_NAME_KEY = "cam_name"

ARGS_KEY = "args"

CONVERT_FUNC_KEY = "convert_func"

LEROBOT_FEATURE_KEY = "lerobot_feature"

NAME_KEY = "names"
SHAPE_KEY = "shape"
DTYPE_KEY = "dtype"

IMAGE_DTYPE_VALUE = "video"

FRAME_IDX_KEY = "frame_idx"

MODULE_KEY = "module"

CLASS_KEY = "class"

FLOAT32 = "float32"

LEFORMAT_PATH = "leformat_path"
HARD_LINK_PATH = "hard_link_path"

CONVERTER_FACTORY_CONFIG = "converter_factory_config"

CONVERTER_CONFIG = "converter_config"

CONVERTER_CLASS_NAME = "converter_class_name"

CONVERTER_MODULE_PATH = "converter_module_path"

CONVERTER_KWARGS = "kwargs"

REPO_ID = "repo_id"

DEVICE_MODEL = "device_model"

VIDEO_BACKEND = "video_backend"
IMAGE_WRITER_PROCESSES = "image_writer_processes"
IMAGE_WRITER_THREADS = "image_writer_threads"

CONVERTER_LOG_DIR = "converter_log_dir"

CONVERTER_LOG_NAME = "converter_log_name"

# Acceptable files
LOCAL_DATASET_INFO_FILE = "local_dataset_info.yaml"
LOCAL_TASK_INFO_FILE = "local_task_info.yaml"
DATASET_UUID_FILE = "dataset_uuid.yaml"
DESCRIPTION_TXT_FILE = "description.txt"
DESCRIBE_TXT_FILE = "describe.txt"
DEVICE_MODEL_ANNOTATION_FILE = "device_model_annotation.yaml"

# YAML field keys
TASK_DESCRIPTIONS_KEY = "task_instruction"

# Acceptable file extensions
H5_SUFFIX = ".h5"
HDF5_SUFFIX = ".hdf5"

DEVICE_MODEL_VERSION_KEY = "version"
DEFAULT_DEVICE_MODEL_VERSION = "default_version"

IS_TEST = "is_test"
AUTO_REENCODE = "auto_reencode"
