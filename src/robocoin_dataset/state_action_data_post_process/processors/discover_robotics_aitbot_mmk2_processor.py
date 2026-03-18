from pathlib import Path

import numpy as np
import pandas as pd

from .state_action_data_processor_base import StateActionDataPostProcessorBase

class ThirdViewProcessor(StateActionDataPostProcessorBase):
    def __init__(self, convert_path: str | Path) -> None:
        super().__init__(convert_path)
        
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
        
        # 前12维数据不变，后面的所有数据都要 *2pi/360
        scale_factor = 2 * np.pi / 360
        processed_state[:, 12:] = processed_state[:, 12:] * scale_factor
        processed_action[:, 12:] = processed_action[:, 12:] * scale_factor


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
        new_state_data = ori_state_data.copy()
        # 应用平滑滤波
        return new_state_data

    # 该方法将ori_action_data进行后处理，返回结果为后处理后的数据
    def process_episode_action_data(self, ori_action_data: np.ndarray) -> np.ndarray:
        new_action_data = ori_action_data.copy()
        # 应用平滑滤波
        return new_action_data
