from pathlib import Path
import logging
import numpy as np
import yaml
import importlib
import pandas as pd
from pathlib import Path

from robocoin_dataset.sim_replay.lerobot_sim_replayer import LerobotSimReplayer
from .state_action_data_processor_base import StateActionDataPostProcessorBase

logger = logging.getLogger(__name__)

class AgilexCobotDecoupledMagicProcessor(StateActionDataPostProcessorBase):
    def __init__(self, convert_path: str | Path) -> None:
        super().__init__(convert_path)
        self.replayer: LerobotSimReplayer | None = None
        self.replayer_convert_path: Path | None = None  # 记录 replayer 对应的路径
        self.episode_index = 0  # 默认值

    def prepare_processing(self) -> None:
        pass

    def process_episode_data(self, ori_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # 从基类获取当前正在处理的 episode 索引
        if self.episode_idx is not None:
            self.episode_index = self.episode_idx


        sync_state_action = self.sync_state_action(ori_data)
        print("sync_state_action keys:", sync_state_action.keys())
        processed_data=self.smooth_dict_data(sync_state_action, 2)
        
        # 先进行缩放处理
        processed_state = processed_data["observation.state"]
        processed_action = processed_data["action"]
        # processed_gripper_open_scale_state = processed_data.get("gripper_open_scale_state") if "gripper_open_scale_state" in processed_data else None
        # processed_gripper_open_scale_action = processed_data.get("gripper_open_scale_action") if "gripper_open_scale_action" in processed_data else None

        result = {
            "observation.state": processed_state,
            "action": processed_action,
        }
        if "gripper_open_scale_state" in ori_data:
            result["gripper_open_scale_state"] = processed_data["gripper_open_scale_state"]
        if "gripper_open_scale_action" in ori_data:
            result["gripper_open_scale_action"] = processed_data["gripper_open_scale_action"]
        return result

    # 该方法将ori_state_data进行后处理，返回结果为后处理后的数据
    def process_episode_state_data(self, ori_state_data: np.ndarray) -> np.ndarray:
        # 这个方法现在由 process_episode_data 调用，逻辑已集中处理
        return ori_state_data

    # 该方法将ori_action_data进行后处理，返回结果为后处理后的数据
    def process_episode_action_data(self, ori_action_data: np.ndarray) -> np.ndarray:
        # 这个方法现在由 process_episode_data 调用，逻辑已集中处理
        return ori_action_data
    

class AgilexCobotDecoupledRealsenseMagicProcessor(StateActionDataPostProcessorBase):
    def __init__(self, convert_path: str | Path) -> None:
        super().__init__(convert_path)
        self.replayer: LerobotSimReplayer | None = None
        self.replayer_convert_path: Path | None = None  # 记录 replayer 对应的路径
        self.episode_index = 0  # 默认值

    def prepare_processing(self) -> None:
        pass

    def process_episode_data(self, ori_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # 从基类获取当前正在处理的 episode 索引
        if self.episode_idx is not None:
            self.episode_index = self.episode_idx

        processed_data=self.smooth_dict_data(ori_data, 2)
        
        # 先进行缩放处理
        processed_state = processed_data["observation.state"]
        processed_action = processed_data["action"]
        # processed_gripper_open_scale_state = processed_data.get("gripper_open_scale_state") if "gripper_open_scale_state" in processed_data else None
        # processed_gripper_open_scale_action = processed_data.get("gripper_open_scale_action") if "gripper_open_scale_action" in processed_data else None

        result = {
            "observation.state": processed_state,
            "action": processed_action,
        }
        if "gripper_open_scale_state" in ori_data:
            result["gripper_open_scale_state"] = processed_data["gripper_open_scale_state"]
        if "gripper_open_scale_action" in ori_data:
            result["gripper_open_scale_action"] = processed_data["gripper_open_scale_action"]
        return result


    # def process_episode_data(self, ori_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    #     # 从基类获取当前正在处理的 episode 索引
    #     if self.episode_idx is not None:
    #         self.episode_index = self.episode_idx
        
    #     # 先进行缩放处理
    #     processed_state = ori_data["observation.state"]
    #     processed_action = ori_data["action"]

    #     # central_logger = logging.getLogger("state action data post process server")
    #     # # 根据EEF距离判断是否需要交换
    #     # need_swap = False
        
    #     # self.replayer = self._setup_replayer()

    #     # if self.replayer is None:
    #     #     central_logger.warning(f"Episode {self.episode_index}: Replayer not available, skipping EEF distance check. Will not swap arms.")
    #     # else:
    #     #     try:
    #     #         # 假设 replayer 已经配置好，直接调用
    #     #         self.replayer.mjcf_model.opt.gravity[2] = -9.81 # 确保重力正确
    #     #         eef_data_list_state = self.replayer.replay_episode_background(self.episode_index, is_state=True, is_sa_dpp=True, is_eef=False)
                
    #     #         if eef_data_list_state and len(eef_data_list_state) > 0 and eef_data_list_state[0].shape[0] >= 12:
    #     #             eef_data = np.array(eef_data_list_state)
    #     #             # 左臂末端执行器位置在前3个元素(index 0:3)，右臂末端执行器位置在6:9
    #     #             left_eef_positions = eef_data[:, :3]
    #     #             right_eef_positions = eef_data[:, 6:9]
                    
    #     #             # 计算双臂末端执行器y轴平均位置的差
    #     #             mean_left_y = np.mean(left_eef_positions[:, 1])
    #     #             mean_right_y = np.mean(right_eef_positions[:, 1])
    #     #             mean_eef_distance = np.abs(mean_left_y - mean_right_y)
                    
    #     #             DISTANCE_THRESHOLD = 0.6
                    
    #     #             central_logger.info(f"Episode {self.episode_index}: Mean EEF Y-axis distance: {mean_eef_distance:.4f}m {'>=' if mean_eef_distance >= DISTANCE_THRESHOLD else '<'} Threshold: {DISTANCE_THRESHOLD}m")

    #     #             if mean_eef_distance >= DISTANCE_THRESHOLD:
    #     #                 need_swap = True
    #     #         else:
    #     #             central_logger.warning(f"Episode {self.episode_index}: EEF data is invalid or insufficient. Skipping distance check. Data shape: {eef_data_list_state[0].shape if eef_data_list_state else 'Empty'}")

    #     #     except Exception as e:
    #     #         central_logger.error(f"Episode {self.episode_index}: Failed to get EEF data or perform distance check. Reason: {e}", exc_info=True)
        
    #     # # 如果需要交换，执行交换操作
    #     # if need_swap:
    #     #     central_logger.info(f"Episode {self.episode_index}: Swapping left and right arms.")
    #     #     processed_state = self._swap_left_right(processed_state)
    #     #     processed_action = self._swap_left_right(processed_action)

    #     result = {
    #         "observation.state": processed_state,
    #         "action": processed_action,
    #     }
    #     if "gripper_open_scale_state" in ori_data:
    #         result["gripper_open_scale_state"] = ori_data["gripper_open_scale_state"]
    #     if "gripper_open_scale_action" in ori_data:
    #         result["gripper_open_scale_action"] = ori_data["gripper_open_scale_action"]
    #     return result

    # 该方法将ori_state_data进行后处理，返回结果为后处理后的数据
    def process_episode_state_data(self, ori_state_data: np.ndarray) -> np.ndarray:
        # 这个方法现在由 process_episode_data 调用，逻辑已集中处理
        return ori_state_data

    # 该方法将ori_action_data进行后处理，返回结果为后处理后的数据
    def process_episode_action_data(self, ori_action_data: np.ndarray) -> np.ndarray:
        # 这个方法现在由 process_episode_data 调用，逻辑已集中处理
        return ori_action_data

class AgilexCobotDecoupledMagicH5Mp4Processor(StateActionDataPostProcessorBase):
    def __init__(self, convert_path: str | Path) -> None:
        super().__init__(convert_path)
        self.episode_index = 0  # 默认值

    def prepare_processing(self) -> None:
        self.left_gripper_open_state_data_idx = 6
        self.right_gripper_open_state_data_idx = 13
        # self.left_gripper_open_action_data_idx = self.left_gripper_open_state_data_idx
        # self.right_gripper_open_action_data_idx = self.right_gripper_open_state_data_idx
        pass

    # 该方法将ori_state_data进行后处理，返回结果为后处理后的数据
    def process_episode_state_data(self, ori_state_data: np.ndarray) -> np.ndarray:
        new_state_data = ori_state_data.copy()
        return new_state_data

    # 该方法将ori_action_data进行后处理，返回结果为后处理后的数据
    def process_episode_action_data(self, ori_action_data: np.ndarray) -> np.ndarray:
        new_action_data = ori_action_data.copy()
        return new_action_data

    # # 忽略原始 action 数据，全部使用 state 数据覆盖
    # def process_episode_data(self, ori_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    #     processed_state = self.process_episode_state_data(ori_data["observation.state"])
    #     # action 特征在本 processor 中与 state 特征长度一致，直接复制
    #     processed_action = self.process_episode_state_data(ori_data["action"])

    #     result = {"observation.state": processed_state, "action": processed_action}
    #     if "gripper_open_scale_state" in ori_data:
    #         result["gripper_open_scale_state"] = ori_data["gripper_open_scale_state"]
    #     if "gripper_open_scale_action" in ori_data:
    #         result["gripper_open_scale_action"] = ori_data["gripper_open_scale_action"]
    #     return result

    def process_episode_data(self, ori_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # 从基类获取当前正在处理的 episode 索引
        if self.episode_idx is not None:
            self.episode_index = self.episode_idx

        processed_data=self.smooth_dict_data(ori_data, 2)
        
        # 先进行缩放处理
        processed_state = processed_data["observation.state"]
        processed_action = processed_data["action"]
        # processed_gripper_open_scale_state = processed_data.get("gripper_open_scale_state") if "gripper_open_scale_state" in processed_data else None
        # processed_gripper_open_scale_action = processed_data.get("gripper_open_scale_action") if "gripper_open_scale_action" in processed_data else None

        result = {
            "observation.state": processed_state,
            "action": processed_action,
        }
        if "gripper_open_scale_state" in ori_data:
            result["gripper_open_scale_state"] = processed_data["gripper_open_scale_state"]
        if "gripper_open_scale_action" in ori_data:
            result["gripper_open_scale_action"] = processed_data["gripper_open_scale_action"]
        return result


    # 该方法返回处理后的state数据名称
    def get_modified_feature_names(self):
        return super().get_modified_feature_names()

    def get_modified_info_state_names(self) -> dict[str, str]:
        return {}

    # 该方法返回处理后的action数据名称
    def get_modified_info_action_names(self) -> dict[str, str]:
        return {}


class AgilexCobotDecoupledMagicMultSensorProcessor(StateActionDataPostProcessorBase):
    def __init__(self, convert_path: str | Path) -> None:
        super().__init__(convert_path)
        self.replayer: LerobotSimReplayer | None = None
        self.replayer_convert_path: Path | None = None  # 记录 replayer 对应的路径
        self.episode_index = 0  # 默认值

    def prepare_processing(self) -> None:
        pass

    def process_episode_data(self, ori_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # 从基类获取当前正在处理的 episode 索引
        if self.episode_idx is not None:
            self.episode_index = self.episode_idx

        sync_state_action = self.sync_state_action(ori_data)

        processed_data=self.smooth_dict_data(sync_state_action, 2)
        
        # 先进行缩放处理
        processed_state = processed_data["observation.state"]
        processed_action = processed_data["action"]
        # processed_gripper_open_scale_state = processed_data.get("gripper_open_scale_state") if "gripper_open_scale_state" in processed_data else None
        # processed_gripper_open_scale_action = processed_data.get("gripper_open_scale_action") if "gripper_open_scale_action" in processed_data else None
        
        result = {
            "observation.state": processed_state,
            "action": processed_action,
        }
        if "gripper_open_scale_state" in ori_data:
            result["gripper_open_scale_state"] = processed_data["gripper_open_scale_state"]
        if "gripper_open_scale_action" in ori_data:
            result["gripper_open_scale_action"] = processed_data["gripper_open_scale_action"]
        return result

    # 该方法将ori_state_data进行后处理，返回结果为后处理后的数据
    def process_episode_state_data(self, ori_state_data: np.ndarray) -> np.ndarray:
        # 这个方法现在由 process_episode_data 调用，逻辑已集中处理
        return ori_state_data

    # 该方法将ori_action_data进行后处理，返回结果为后处理后的数据
    def process_episode_action_data(self, ori_action_data: np.ndarray) -> np.ndarray:
        # 这个方法现在由 process_episode_data 调用，逻辑已集中处理
        return ori_action_data
