from pathlib import Path

import numpy as np
import pandas as pd

from .state_action_data_processor_base import StateActionDataPostProcessorBase


class UnitreeG1Processor(StateActionDataPostProcessorBase):
    def __init__(self, convert_path: str | Path) -> None:
        super().__init__(convert_path)

    def prepare_processing(self) -> None:
        pass

    # 该方法将ori_state_data进行后处理，返回结果为后处理后的数据
    def process_episode_state_data(self, ori_state_data: np.ndarray) -> np.ndarray:
        new_state_data = ori_state_data.copy()
        return new_state_data

    # 该方法将ori_action_data进行后处理，返回结果为后处理后的数据
    def process_episode_action_data(self, ori_action_data: np.ndarray) -> np.ndarray:
        new_action_data = ori_action_data.copy()
        return new_action_data


class UnitreeG1ThreeFingerOutProcessor(StateActionDataPostProcessorBase):
    def __init__(self, convert_path: str | Path) -> None:
        super().__init__(convert_path)

    def prepare_processing(self) -> None:
        pass

    # 该方法将ori_state_data进行后处理，返回结果为后处理后的数据
    def process_episode_state_data(self, ori_state_data: np.ndarray) -> np.ndarray:
        new_state_data = ori_state_data.copy()
        return new_state_data

    # 该方法将ori_action_data进行后处理，返回结果为后处理后的数据
    def process_episode_action_data(self, ori_action_data: np.ndarray) -> np.ndarray:
        new_action_data = ori_action_data.copy()
        return new_action_data
    
    def process_episode_data(self, ori_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        将 action 的手部数据（后14维）复制给 state 的手部数据
        state 和 action 的前14维（双臂）保持不变
        然后对两者都应用平滑滤波
        """
        state = ori_data.get("observation.state")
        action = ori_data.get("action")
        
        if state is None or action is None:
            raise ValueError("ori_data must contain 'observation.state' and 'action'")
        
        if not isinstance(state, np.ndarray) or not isinstance(action, np.ndarray):
            raise ValueError("state and action must be numpy arrays")
        
        if state.shape[0] != action.shape[0]:
            raise ValueError("state and action must have same number of frames")
        
        # 复制 state 和 action
        new_state = state.copy()
        new_action = action.copy()
        
        return {"observation.state": new_state, "action": new_action}

