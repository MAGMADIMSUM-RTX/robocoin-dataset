from pathlib import Path
import logging

import numpy as np
import pandas as pd

from .state_action_data_processor_base import StateActionDataPostProcessorBase


class YinheProcessor(StateActionDataPostProcessorBase):
    def __init__(self, convert_path: str | Path) -> None:
        super().__init__(convert_path)
        self.left_gripper_state_idx = None
        self.right_gripper_state_idx = None
        self.left_gripper_action_idx = None
        self.right_gripper_action_idx = None
        self.episode_index = 0

    def prepare_processing(self) -> None:
        # 获取原始的 state 和 action 特征名称
        ori_names = self.get_ori_state_action_feature_names()
        state_names = ori_names["observation.state"]
        action_names = ori_names["action"]
        
        # 查找 gripper 字段在 state 中的索引
        try:
            self.left_gripper_state_idx = state_names.index("left_gripper_width_m")
        except ValueError:
            self.left_gripper_state_idx = None
            
        try:
            self.right_gripper_state_idx = state_names.index("right_gripper_width_m")
        except ValueError:
            self.right_gripper_state_idx = None
        # print("索引是 {} 和 {}。".format(self.left_gripper_state_idx, self.right_gripper_state_idx))
        # 查找 gripper 字段在 action 中的索引
        try:
            self.left_gripper_action_idx = action_names.index("left_gripper_open")
        except ValueError:
            self.left_gripper_action_idx = 7
            
        try:
            self.right_gripper_action_idx = action_names.index("right_gripper_open")
        except ValueError:
            self.right_gripper_action_idx = 15

    def _smooth_data(self, data: np.ndarray, window_size: int = 4) -> np.ndarray:
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

    def _scale_gripper_columns(self, data: np.ndarray, gripper_indices: list[int]) -> np.ndarray:
        """如果 gripper 列的最大值 > 0.05，则递归除以10直到 <= 0.05"""
        scaled_data = data.copy()
        logger = logging.getLogger("state action data post process server")
        for col_idx in gripper_indices:
            if col_idx is None:
                continue
            col_max = float(np.max(scaled_data[:, col_idx]))
            division_count = 0
            multiply_count = 0
            # 先处理大于 0.05 的情况
            # while col_max > 0.05:
            #     scaled_data[:, col_idx] = scaled_data[:, col_idx] / 3.0
            #     division_count += 1
            #     col_max = float(np.max(scaled_data[:, col_idx]))
            # if division_count > 0:
            #     logger.info(f"Episode {self.episode_index} divided by 3 {division_count} time(s), final max value: {col_max:.6f}")
            # 新增：如果最大值在 (0.1, 1.0) 之间，循环乘以 20
            if col_max < 0.0:
                raise ValueError(f"Gripper column at index {col_idx} has negative max value {col_max}")
            while 0.1 > col_max and col_max > 0 and multiply_count < 10:  # 添加 col_max > 0 和最大迭代次数限制
                scaled_data[:, col_idx] = scaled_data[:, col_idx] * 20.0
                multiply_count += 1
                col_max = float(np.max(scaled_data[:, col_idx]))
            if col_max > 1.0:
                scaled_data[:, col_idx] = scaled_data[:, col_idx] / 2.0
                col_max = float(np.max(scaled_data[:, col_idx]))
            if multiply_count > 0:
                logger.info(f"Episode {self.episode_index} multiplied by 20 {multiply_count} time(s), final max value: {col_max:.6f}")
            # if col_max > 1.0 or col_max < 0.4:
            #     raise ValueError(f"After scaling, gripper column at index {col_idx} still has max value {col_max} > 1.0 or < 0.4")
        return scaled_data

    # 该方法将ori_state_data进行后处理，返回结果为后处理后的数据
    def process_episode_state_data(self, ori_state_data: np.ndarray) -> np.ndarray:
        new_state_data = ori_state_data.copy()
        # 先缩放 gripper 列
        if self.left_gripper_state_idx is not None or self.right_gripper_state_idx is not None:
            gripper_indices = [self.left_gripper_state_idx, self.right_gripper_state_idx]
            new_state_data = self._scale_gripper_columns(new_state_data, gripper_indices)
        # 应用平滑滤波
        new_state_data = self._smooth_data(new_state_data, window_size=5)
        return new_state_data

    # 该方法将ori_action_data进行后处理，返回结果为后处理后的数据
    def process_episode_action_data(self, ori_action_data: np.ndarray) -> np.ndarray:
        new_action_data = ori_action_data.copy()
        # 应用平滑滤波
        new_action_data = self._smooth_data(new_action_data, window_size=5)
        return new_action_data
    
    # 该方法将episode数据进行后处理，将 state 中的 gripper 数据复制到 action
    def process_episode_data(self, ori_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # 从基类获取当前正在处理的 episode 索引
        if self.episode_idx is not None:
            self.episode_index = self.episode_idx
        
        state = ori_data.get("observation.state")
        action = ori_data.get("action")

        if state is None or action is None:
            raise ValueError("ori_data must contain 'observation.state' and 'action'")

        if not isinstance(state, np.ndarray) or not isinstance(action, np.ndarray):
            raise ValueError("state and action must be numpy arrays")

        if state.shape[0] != action.shape[0]:
            raise ValueError("state and action must have same number of frames")

        out_state = self.process_episode_state_data(state)
        out_action = self.process_episode_action_data(action)

        # 将 state 中的 gripper 数据插入到 action 中
        # 按照 action 的特征顺序重新构建数组
        if self.left_gripper_state_idx is not None and self.right_gripper_state_idx is not None:
            # 提取 gripper 数据
            left_gripper_data = out_state[:, self.left_gripper_state_idx:self.left_gripper_state_idx+1]
            right_gripper_data = out_state[:, self.right_gripper_state_idx:self.right_gripper_state_idx+1]
            
            # 在索引 7 和 15 位置插入 gripper 数据
            # left_gripper_open 插入到索引 7 (在 left_arm_joint_7_rad 之后)
            # right_gripper_open 插入到索引 15 (在 right_arm_joint_7_rad 之后，但要考虑已插入的 left_gripper)
            out_action = np.concatenate([
                out_action[:, :7],              # left_arm_joint_1 到 joint_7
                left_gripper_data,              # left_gripper_open
                out_action[:, 7:14],            # right_arm_joint_1 到 joint_7
                right_gripper_data,             # right_gripper_open
            ], axis=1)
            # print("已将 gripper 数据从 state 复制到 action。")

        result = {"observation.state": out_state, "action": out_action}
        if "gripper_open_scale_state" in ori_data:
            result["gripper_open_scale_state"] = ori_data["gripper_open_scale_state"]
        if "gripper_open_scale_action" in ori_data:
            result["gripper_open_scale_action"] = ori_data["gripper_open_scale_action"]
        return result
    
    def get_modified_feature_names(self):
        return super().get_modified_feature_names()

    def get_modified_info_state_names(self) -> dict[str, str]:
        return {}

    # 该方法返回处理后的action数据名称
    def get_modified_info_action_names(self) -> dict[str, str]:
        return {}

