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

    def smooth_dict_data(self, data_dict: dict[str, np.ndarray], window_size: int = 4) -> dict[str, np.ndarray]:
        """
        对字典中的每个数据数组调用 smooth_data 进行平滑滤波
        
        Args:
            data_dict: 键为字符串，值为 numpy 数组的字典
            window_size: 滑动窗口大小
            
        Returns:
            平滑处理后的字典
        """
        result = {}
        for key, value in data_dict.items():
            result[key] = self.smooth_data(value, window_size=window_size)
        return result
    
    def sync_state_action(self, data_dict: dict[str, np.ndarray], source: str = "state", offset: int = 30) -> dict[str, np.ndarray]:
        """
        将 observation.state / gripper_open_scale_state 的值覆盖 action / gripper_open_scale_action，
        或者反向复制。注意：action 相比于 state 提前 offset 帧。
        
        Args:
            data_dict: 包含特征数据的字典
            source: "state" 表示将 state 覆盖到 action；"action" 表示将 action 覆盖到 state
            offset: 提前的帧数（默认为 30 帧）
            
        Returns:
            处理后的字典副本
        """
        result = {k: v.copy() for k, v in data_dict.items()}
        
        state_key = "observation.state"
        action_key = "action"
        g_state_key = "gripper_open_scale_state"
        g_action_key = "gripper_open_scale_action"

        if source == "state":
            # state -> action (action[t] = state[t+offset])
            if state_key in result and action_key in result:
                n_frames = len(result[state_key])
                new_action = np.zeros_like(result[action_key])
                if n_frames > offset:
                    new_action[:-offset] = result[state_key][offset:]
                    new_action[-offset:] = result[state_key][-1]  # 剩余帧补齐
                else:
                    new_action[:] = result[state_key][-1]
                result[action_key] = new_action

            if g_state_key in result and g_action_key in result and result[g_state_key] is not None:
                n_frames = len(result[g_state_key])
                new_g_action = np.zeros_like(result[g_action_key])
                if n_frames > offset:
                    new_g_action[:-offset] = result[g_state_key][offset:]
                    new_g_action[-offset:] = result[g_state_key][-1]
                else:
                    new_g_action[:] = result[g_state_key][-1]
                result[g_action_key] = new_g_action

        elif source == "action":
            # action -> state (state[t+offset] = action[t])
            if state_key in result and action_key in result:
                n_frames = len(result[action_key])
                new_state = result[state_key].copy()
                if n_frames > offset:
                    new_state[offset:] = result[action_key][:-offset]
                    # 第 0 帧到 offset-1 帧 state 保持不变
                result[state_key] = new_state
                
            if g_state_key in result and g_action_key in result and result[g_action_key] is not None:
                n_frames = len(result[g_action_key])
                new_g_state = result[g_state_key].copy()
                if n_frames > offset:
                    new_g_state[offset:] = result[g_action_key][:-offset]
                result[g_state_key] = new_g_state

        return result

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
