import json
from pathlib import Path
import pandas as pd
import numpy as np

from robocoin_dataset.data_post_process import DataPostProcessorBase


class StateActionDataPostProcessorBase(DataPostProcessorBase):
    def __init__(self, convert_path: str | Path) -> None:
        super().__init__(
            convert_path=convert_path,
            data_post_process_type="state_action",
            data_feature_keys={
                "observation.state",
                "action",
                "gripper_open_scale_state",
                "gripper_open_scale_action",
            },
        )
        self.convert_path = Path(convert_path)
        if not self.convert_path.exists():
            raise ValueError(f"{self.convert_path} does not exist")

    # def get_modified_feature_names(self) -> dict[str, list[str]]:
    #     return self.get_ori_state_action_feature_names()
    def smooth_data(self, data: np.ndarray, window_size: int = 4) -> np.ndarray:
        """
        对数据进行平滑滤波
        
        Args:
            data: 输入数据，shape 为 (n_frames, n_features)
            window_size: 滑动窗口大小
            
        Returns:
            平滑后的数据
        """
        if data.shape[0] < window_size:
            # 如果数据长度小于窗口大小，直接返回原数据
            return data
        
        smoothed_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            series = pd.Series(data[:, i])
            smoothed_series = series.rolling(window=window_size, min_periods=1, center=True).mean()
            smoothed_data[:, i] = smoothed_series.values
        
        return smoothed_data

    
    def _swap_left_right(self, data: np.ndarray) -> np.ndarray:
        """交换前13维和后13维"""
        if data.ndim != 2 or data.shape[1] < 26:
            return data
        swapped_data = data.copy()
        left = swapped_data[:, :13].copy()
        right = swapped_data[:, 13:26].copy()
        swapped_data[:, :13] = right
        swapped_data[:, 13:26] = left
        return swapped_data

    # def _setup_replayer(self) -> LerobotSimReplayer | None:
    #     """动态加载并实例化 LerobotSimReplayer"""
    #     try:
    #         # 检查 replayer 是否已存在且路径匹配
    #         if self.replayer is not None and self.replayer_convert_path == self.convert_path:
    #             return self.replayer
            
    #         project_root = Path(__file__).resolve().parents[4]
    #         config_path = project_root / "scripts/sim_replay/configs/sim_replay_config_path.yaml"

    #         if not config_path.exists():
    #             logger.warning(f"Sim replay config file not found at {config_path}. Replayer will not be available.")
    #             return None

    #         with open(config_path, "r") as f:
    #             all_configs = yaml.safe_load(f)

    #         device_configs = all_configs.get("agilex_cobot_decoupled_magic", [])
    #         config_info = next((c for c in device_configs if c.get("version") == "default_version"), None)

    #         if not config_info:
    #             logger.warning("Config for 'agilex_cobot_decoupled_magic' with 'default_version' not found. Replayer will not be available.")
    #             return None

    #         module_name = config_info["mujoco_sim_replay_config_module"]
    #         class_name = config_info["mujoco_sim_replay_config_class"]

    #         module = importlib.import_module(module_name)
    #         config_class = getattr(module, class_name)
    #         replay_config = config_class()

    #         # repo_path 是 LeRobot 数据集目录
    #         new_replayer = LerobotSimReplayer(replay_config=replay_config, repo_path=self.convert_path)
    #         self.replayer_convert_path = self.convert_path
    #         return new_replayer

    #     except Exception as e:
    #         logger.error(f"Failed to setup LerobotSimReplayer. Reason: {e}", exc_info=True)
    #         return None

    def set_episode_index(self, episode_index: int):
        """设置当前处理的 episode 索引"""
        self.episode_index = episode_index


    def get_ori_state_action_feature_names(self) -> dict[str, list[str]]:
        if not self.info_file_path.exists():
            raise ValueError(f"{self.info_file_path} does not exist")
        with open(self.info_file_path) as f:
            json_dict = json.load(f)

        if "features" not in json_dict:
            raise ValueError(f"{self.info_file_path} does not contain features")

        if "observation.state" not in json_dict["features"]:
            raise ValueError(f"{self.info_file_path} does not contain observation.state")

        if "action" not in json_dict["features"]:
            raise ValueError(f"{self.info_file_path} does not contain action")

        if not isinstance(json_dict["features"]["observation.state"]["names"], list):
            raise ValueError("value of observation.state.names is not list[str]")

        if not isinstance(json_dict["features"]["action"]["names"], list):
            raise ValueError("value of action.names is not list[str]")

        result = {
            "observation.state": json_dict["features"]["observation.state"]["names"],
            "action": json_dict["features"]["action"]["names"],
        }

        if "gripper_open_scale_state" in json_dict["features"]:
            result["gripper_open_scale_state"] = json_dict["features"][
                "gripper_open_scale_state"
            ]["names"]

        if "gripper_open_scale_action" in json_dict["features"]:
            result["gripper_open_scale_action"] = json_dict["features"][
                "gripper_open_scale_action"
            ]["names"]

        return result

    def process_episode_data(self, ori_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = {
            "observation.state": self.process_episode_state_data(ori_data["observation.state"]),
            "action": self.process_episode_action_data(ori_data["action"]),
        }
        if "gripper_open_scale_state" in ori_data and ori_data["gripper_open_scale_state"] is not None:
            result["gripper_open_scale_state"] = self.process_episode_gripper_state_data(
                ori_data["gripper_open_scale_state"]
            )
        if "gripper_open_scale_action" in ori_data and ori_data["gripper_open_scale_action"] is not None:
            result["gripper_open_scale_action"] = self.process_episode_gripper_action_data(
                ori_data["gripper_open_scale_action"]
            )
        return result

    def get_modified_feature_names(self) -> dict[str, list[str]]:
        state_names = self.get_modified_state_feature_names()
        action_names = self.get_modified_action_feature_names()

        result = {
            "observation.state": state_names,
            "action": action_names,
        }

        ori_features = self.get_ori_state_action_feature_names()
        if "gripper_open_scale_state" in ori_features:
            result["gripper_open_scale_state"] = self.get_modified_gripper_state_feature_names()

        if "gripper_open_scale_action" in ori_features:
            result["gripper_open_scale_action"] = self.get_modified_gripper_action_feature_names()

        return result

    # 在这里填入修改后的state特征名称列表，如果没有修改，则不需要重写函数
    def get_modified_state_feature_names(self) -> list[str]:
        return self.get_ori_state_action_feature_names()["observation.state"]

    # 在这里填入修改后的action特征名称列表，如果没有修改，则不需要重写函数
    def get_modified_action_feature_names(self) -> list[str]:
        return self.get_ori_state_action_feature_names()["action"]

    def get_modified_gripper_state_feature_names(self) -> list[str]:
        return self.get_ori_state_action_feature_names().get("gripper_open_scale_state", [])

    def get_modified_gripper_action_feature_names(self) -> list[str]:
        return self.get_ori_state_action_feature_names().get("gripper_open_scale_action", [])

    # 将处理episode数据的准备工作放在这里
    def prepare_processing(self) -> None:
        pass

    # 该方法将ori_state_data进行后处理，返回结果为后处理后的数据
    def process_episode_state_data(self, ori_state_data: np.ndarray) -> np.ndarray:
        return ori_state_data.copy()

    # 该方法将ori_action_data进行后处理，返回结果为后处理后的数据
    def process_episode_action_data(self, ori_action_data: np.ndarray) -> np.ndarray:
        return ori_action_data.copy()

    def process_episode_gripper_state_data(self, ori_gripper_state_data: np.ndarray) -> np.ndarray:
        return ori_gripper_state_data.copy()

    def process_episode_gripper_action_data(
        self, ori_gripper_action_data: np.ndarray
    ) -> np.ndarray:
        return ori_gripper_action_data.copy()
