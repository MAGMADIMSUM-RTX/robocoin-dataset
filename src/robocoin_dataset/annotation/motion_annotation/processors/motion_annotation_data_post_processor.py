import json
from pathlib import Path

import numpy as np

from robocoin_dataset.data_post_process import DataPostProcessorBase
from robocoin_dataset.sim_replay.configs.lerobot_sim_replay_config import (
    LerobotSimReplayConfig,
)
from robocoin_dataset.sim_replay.lerobot_sim_replayer import LerobotSimReplayer
from robocoin_dataset.utils.le_path import (
    get_meta_info_file,
)


class MotionAnnotationDataPostProcessorConfig:
    frame_rate: float = 30.0
    win_size: int = 3

    eef_direction_annotation_dict: dict[str, int] = {
        "forward": 0,
        "backward": 1,
        "left": 2,
        "right": 3,
        "up": 4,
        "down": 5,
        "still": 6,
    }
    direction_threshold: float = 0.02
    eef_velocity_annotation_dict: dict[str, int] = {
        "still": 0,
        "slow": 1,
        "fast": 2,
    }
    velocity_slow_threshold: float = 0.02
    velocity_fast_threshold: float = 0.1
    eef_acc_mag_annotation_dict: dict[str, int] = {
        "constant": 0,
        "accelerating": 1,
        "decelerating": 2,
    }
    acc_mag_threshold: float = 0.1
    gripper_mode_annotation_dict: dict[str, int] = {
        "open": 0,
        "closed": 1,
        "unknown": 2,
    }
    gripper_mode_threshold: float = 0.5
    gripper_activity_annotation_dict: dict[str, int] = {
        "openning": 0,
        "closing": 1,
        "holding": 2,
        "unknown": 3,
    }
    gripper_activity_threshold: float = 0.01


class MotionAnnotationDataPostProcessor(DataPostProcessorBase):
    def __init__(
        self,
        convert_path: str | Path,
        sim_replay_config: LerobotSimReplayConfig,
        motion_annotation_config: MotionAnnotationDataPostProcessorConfig = MotionAnnotationDataPostProcessorConfig(),
    ) -> None:
        # 将 has_gripper 传入实例，后续用于决定生成的特征名称
        self.gripper_value_open = sim_replay_config.gripper_value_open
        self.gripper_value_close = sim_replay_config.gripper_value_close
        self.sim_replay_config = sim_replay_config
        self.motion_annotation_config = motion_annotation_config

        eef_data_feature_keys = {
            "eef_sim_pose_state",
            "eef_sim_pose_action",
            "eef_direction_state",
            "eef_direction_action",
            "eef_velocity_state",
            "eef_velocity_action",
            "eef_acc_mag_state",
            "eef_acc_mag_action",
        }
        gripper_state_feature_keys = {
            "gripper_open_scale_state",
            "gripper_mode_state",
            "gripper_activity_state",
        }
        gripper_action_feature_keys = {
            "gripper_open_scale_action",
            "gripper_mode_action",
            "gripper_activity_action",
        }
        data_feature_keys = (
            eef_data_feature_keys | gripper_state_feature_keys | gripper_action_feature_keys
            if self.sim_replay_config.has_gripper
            else eef_data_feature_keys
        )
        # 所以这里 data_feature_keys 使用输入数据的键名
        super().__init__(
            convert_path=convert_path,
            data_post_process_type="motion_annotation",
            data_feature_keys=data_feature_keys,
        )

        # 创建 simulator（使用父类已经设置的 self.convert_path）
        self.simulator = LerobotSimReplayer(self.sim_replay_config, self.convert_path)

    # }
    def prepare_processing(self) -> None:
        eef_direction_annotation_jsonl_path = (
            self.convert_path / "annotations" / "eef_direction_annotation.jsonl"
        )
        eef_velocity_annotation_jsonl_path = (
            self.convert_path / "annotations" / "eef_velocity_annotation.jsonl"
        )
        eef_acc_mag_annotation_jsonl_path = (
            self.convert_path / "annotations" / "eef_acc_mag_annotation.jsonl"
        )
        gripper_mode_annotation_jsonl_path = (
            self.convert_path / "annotations" / "gripper_mode_annotation.jsonl"
        )
        gripper_activity_annotation_jsonl_path = (
            self.convert_path / "annotations" / "gripper_activity_annotation.jsonl"
        )

        eef_direction_annotation_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_entries = sorted(
            self.motion_annotation_config.eef_direction_annotation_dict.items(),
            key=lambda item: item[1],
        )
        with open(eef_direction_annotation_jsonl_path, "w") as f:
            for direction, idx in sorted_entries:
                json_data = {
                    "eef_direction_index": idx,
                    "eef_direction": direction,
                }
                json.dump(json_data, f)
                f.write("\n")

        eef_velocity_annotation_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_entries = sorted(
            self.motion_annotation_config.eef_velocity_annotation_dict.items(),
            key=lambda item: item[1],
        )
        with open(eef_velocity_annotation_jsonl_path, "w") as f:
            for velocity, idx in sorted_entries:
                json_data = {
                    "eef_velocity_index": idx,
                    "eef_velocity": velocity,
                }
                json.dump(json_data, f)
                f.write("\n")

        eef_acc_mag_annotation_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_entries = sorted(
            self.motion_annotation_config.eef_acc_mag_annotation_dict.items(),
            key=lambda item: item[1],
        )
        with open(eef_acc_mag_annotation_jsonl_path, "w") as f:
            for acc_mag, idx in sorted_entries:
                json_data = {
                    "eef_acc_mag_index": idx,
                    "eef_acc_mag": acc_mag,
                }
                json.dump(json_data, f)
                f.write("\n")

        gripper_mode_annotation_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_entries = sorted(
            self.motion_annotation_config.gripper_mode_annotation_dict.items(),
            key=lambda item: item[1],
        )
        with open(gripper_mode_annotation_jsonl_path, "w") as f:
            for mode, idx in sorted_entries:
                json_data = {
                    "gripper_mode_index": idx,
                    "gripper_mode": mode,
                }
                json.dump(json_data, f)
                f.write("\n")

        gripper_activity_annotation_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_entries = sorted(
            self.motion_annotation_config.gripper_activity_annotation_dict.items(),
            key=lambda item: item[1],
        )
        with open(gripper_activity_annotation_jsonl_path, "w") as f:
            for activity, idx in sorted_entries:
                json_data = {
                    "gripper_activity_index": idx,
                    "gripper_activity": activity,
                }
                json.dump(json_data, f)
                f.write("\n")

    def _compute_eef_gripper_sim_data(
        self, ori_data: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        state_arr = ori_data.get("observation.state")
        action_arr = ori_data.get("action")

        new_state = self.process_episode_state_data(state_arr)
        new_action = self.process_episode_action_data(action_arr)
        # 根据 has_gripper 决定输出结构
        # 分离 EEF 和夹爪数据
        num_sites = len(self.sim_replay_config.mjcf_site_names)
        num_eef_cols = num_sites * 6

        state_eef = new_state[:, :num_eef_cols] if new_state is not None else None
        action_eef = new_action[:, :num_eef_cols] if new_action is not None else None

        if self.sim_replay_config.state_gripper_lerobot_names:
            state_gripper = new_state[:, num_eef_cols:] if new_state is not None else None
        else:
            state_gripper = None
        if self.sim_replay_config.action_gripper_lerobot_names:
            action_gripper = new_action[:, num_eef_cols:] if new_action is not None else None
        else:
            action_gripper = None

        return {
            "eef_sim_pose_state": state_eef,
            "eef_sim_pose_action": action_eef,
            "gripper_open_scale_state": state_gripper,
            "gripper_open_scale_action": action_gripper,
        }
        # # 没有夹爪，只返回 EEF 数据
        # return {
        #     "eef_sim_pose_state": state_eef,
        #     "eef_sim_pose_action": action_eef,
        # }

    def _compute_difference_padded(
        self, position_array: np.ndarray, win_size: int = 3
    ) -> np.ndarray:
        frame_num = position_array.shape[0]

        if win_size >= frame_num:
            # 如果 win_size 太大，所有帧都用最后一帧计算
            win_size = frame_num - 1

        difference_padded = np.zeros_like(position_array, dtype=float)

        for t in range(frame_num):
            if t + win_size < frame_num:
                # 正常情况：后面第 n 帧减当前帧
                difference = position_array[t + win_size] - position_array[t]
            else:
                # 边界情况：用最后一帧减当前帧
                difference = position_array[-1] - position_array[t]
                # 时间步长为 (T-1 - t) * dt
                steps = frame_num - 1 - t
                if steps == 0:
                    # 最后一帧自身，速度为 0
                    difference_padded[t] = 0.0
                    continue

            difference_padded[t] = difference / win_size

        return difference_padded

    def _annotate_eef_motion(self, sim_eef_data: np.ndarray) -> dict[str, np.ndarray]:
        left_eef_position = sim_eef_data[:, :3]
        right_eef_position = sim_eef_data[:, 6:9]
        frame_rate = self.motion_annotation_config.frame_rate
        win_size = self.motion_annotation_config.win_size
        direction_threshold = self.motion_annotation_config.direction_threshold

        eef_direction_annotation = np.zeros((len(sim_eef_data), 2), dtype=np.int32)
        eef_velocity_annotation = np.zeros((len(sim_eef_data), 2), dtype=np.int32)
        eef_acc_mag_annotation = np.zeros((len(sim_eef_data), 2), dtype=np.int32)
        left_eef_velocity = (
            self._compute_difference_padded(left_eef_position, win_size=win_size) * frame_rate
        )
        right_eef_velocity = (
            self._compute_difference_padded(right_eef_position, win_size=win_size) * frame_rate
        )
        for i, row in enumerate(left_eef_velocity):
            if np.linalg.norm(row) <= direction_threshold:
                eef_direction_annotation[i, 0] = 6
            elif abs(row[0]) >= abs(row[1]) and abs(row[0]) >= abs(row[2]):
                if row[0] > 0:
                    eef_direction_annotation[i, 0] = 0
                else:
                    eef_direction_annotation[i, 0] = 1
            elif abs(row[1]) >= abs(row[0]) and abs(row[1]) >= abs(row[2]):
                if row[1] > 0:
                    eef_direction_annotation[i, 0] = 2
                else:
                    eef_direction_annotation[i, 0] = 3
            elif abs(row[2]) >= abs(row[0]) and abs(row[2]) >= abs(row[1]):
                if row[2] > 0:
                    eef_direction_annotation[i, 0] = 4
                else:
                    eef_direction_annotation[i, 0] = 5

        for i, row in enumerate(right_eef_velocity):
            if np.linalg.norm(row) <= direction_threshold:
                eef_direction_annotation[i, 1] = 6
            elif abs(row[0]) >= abs(row[1]) and abs(row[0]) >= abs(row[2]):
                if row[0] > 0:
                    eef_direction_annotation[i, 1] = 0
                else:
                    eef_direction_annotation[i, 1] = 1
            elif abs(row[1]) >= abs(row[0]) and abs(row[1]) >= abs(row[2]):
                if row[1] > 0:
                    eef_direction_annotation[i, 1] = 2
                else:
                    eef_direction_annotation[i, 1] = 3
            elif abs(row[2]) >= abs(row[0]) and abs(row[2]) >= abs(row[1]):
                if row[2] > 0:
                    eef_direction_annotation[i, 1] = 4
                else:
                    eef_direction_annotation[i, 1] = 5

        velocity_slow_threshold = self.motion_annotation_config.velocity_slow_threshold
        velocity_fast_threshold = self.motion_annotation_config.velocity_fast_threshold
        for i, row in enumerate(left_eef_velocity):
            if np.linalg.norm(row) <= velocity_slow_threshold:
                eef_velocity_annotation[i, 0] = 0
            elif (
                np.linalg.norm(row) > velocity_slow_threshold
                and np.linalg.norm(row) <= velocity_fast_threshold
            ):
                eef_velocity_annotation[i, 0] = 1
            elif np.linalg.norm(row) > velocity_fast_threshold:
                eef_velocity_annotation[i, 0] = 2

        for i, row in enumerate(right_eef_velocity):
            if np.linalg.norm(row) <= velocity_slow_threshold:
                eef_velocity_annotation[i, 1] = 0
            elif (
                np.linalg.norm(row) > velocity_slow_threshold
                and np.linalg.norm(row) <= velocity_fast_threshold
            ):
                eef_velocity_annotation[i, 1] = 1
            elif np.linalg.norm(row) > velocity_fast_threshold:
                eef_velocity_annotation[i, 1] = 2

        acc_mag_threshold = self.motion_annotation_config.acc_mag_threshold
        left_eef_speed = np.linalg.norm(left_eef_velocity, axis=1)
        right_eef_speed = np.linalg.norm(right_eef_velocity, axis=1)
        left_eef_acc_mag = (
            self._compute_difference_padded(left_eef_speed, win_size=win_size) * frame_rate
        )
        right_eef_acc_mag = (
            self._compute_difference_padded(right_eef_speed, win_size=win_size) * frame_rate
        )

        for i, acce in enumerate(left_eef_acc_mag):
            if abs(acce) <= acc_mag_threshold:
                eef_acc_mag_annotation[i, 0] = 0
            elif acce > acc_mag_threshold:
                eef_acc_mag_annotation[i, 0] = 1
            else:
                eef_acc_mag_annotation[i, 0] = 2

        for i, acce in enumerate(right_eef_acc_mag):
            if abs(acce) <= acc_mag_threshold:
                eef_acc_mag_annotation[i, 1] = 0
            elif acce > acc_mag_threshold:
                eef_acc_mag_annotation[i, 1] = 1
            else:
                eef_acc_mag_annotation[i, 1] = 2

        return {
            "eef_direction": eef_direction_annotation,
            "eef_velocity": eef_velocity_annotation,
            "eef_acc_mag": eef_acc_mag_annotation,
        }

    def _annotate_gripper_motion(
        self, sim_gripper_data: np.ndarray | None
    ) -> dict[str, np.ndarray | None]:
        win_size = self.motion_annotation_config.win_size

        if not self.sim_replay_config.has_gripper:
            return {}
        if sim_gripper_data is None:
            return {}
        gripper_mode_annotation = np.zeros((len(sim_gripper_data), 2), dtype=np.int32)
        gripper_activity_annotation = np.zeros((len(sim_gripper_data), 2), dtype=np.int32)
        left_gripper_open_scale = sim_gripper_data[:, 0]
        right_gripper_open_scale = sim_gripper_data[:, 1]

        left_gripper_activtity = self._compute_difference_padded(
            left_gripper_open_scale, win_size=win_size
        )
        right_gripper_activtity = self._compute_difference_padded(
            right_gripper_open_scale, win_size=win_size
        )

        gripper_mode_threshold = self.motion_annotation_config.gripper_mode_threshold
        gripper_activity_threshold = self.motion_annotation_config.gripper_activity_threshold

        for i, scale in enumerate(left_gripper_open_scale):
            if scale > gripper_mode_threshold:
                gripper_mode_annotation[i, 0] = 0
            else:
                gripper_mode_annotation[i, 0] = 1

        for i, scale in enumerate(right_gripper_open_scale):
            if scale > gripper_mode_threshold:
                gripper_mode_annotation[i, 1] = 0
            else:
                gripper_mode_annotation[i, 1] = 1

        for i, activity_value in enumerate(left_gripper_activtity):
            if abs(activity_value) < gripper_activity_threshold:
                gripper_activity_annotation[i, 0] = 2
            elif activity_value < 0:
                gripper_activity_annotation[i, 0] = 1
            else:
                gripper_activity_annotation[i, 0] = 0

        for i, activity_value in enumerate(right_gripper_activtity):
            if abs(activity_value) < gripper_activity_threshold:
                gripper_activity_annotation[i, 1] = 2
            elif activity_value < 0:
                gripper_activity_annotation[i, 1] = 1
            else:
                gripper_activity_annotation[i, 1] = 0

        return {
            "gripper_mode": gripper_mode_annotation,
            "gripper_activity": gripper_activity_annotation,
        }

    def process_episode_data(self, ori_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        sim_data = self._compute_eef_gripper_sim_data(ori_data)
        eef_sim_data_state = sim_data["eef_sim_pose_state"]
        eef_sim_data_action = sim_data["eef_sim_pose_action"]
        gripper_open_scale_state = sim_data.get("gripper_open_scale_state", None)
        gripper_open_scale_action = sim_data.get("gripper_open_scale_action", None)

        eef_sim_data_state_dict = {
            "eef_sim_pose_state": eef_sim_data_state,
        }
        eef_sim_data_action_dict = {
            "eef_sim_pose_action": eef_sim_data_action,
        }
        gripper_open_scale_state_dict = {
            "gripper_open_scale_state": gripper_open_scale_state,
        }

        gripper_open_scale_action_dict = {
            "gripper_open_scale_action": gripper_open_scale_action,
        }
        # gripper_open_scale_action_dict = {
        #     "gripper_open_scale_action": gripper_open_scale_action,
        # }

        eef_annotation = self._annotate_eef_motion(eef_sim_data_state)
        eef_annotation_state = {
            "eef_direction_state": eef_annotation["eef_direction"],
            "eef_velocity_state": eef_annotation["eef_velocity"],
            "eef_acc_mag_state": eef_annotation["eef_acc_mag"],
        }

        eef_annotation = self._annotate_eef_motion(eef_sim_data_action)
        eef_annotation_action = {
            "eef_direction_action": eef_annotation["eef_direction"],
            "eef_velocity_action": eef_annotation["eef_velocity"],
            "eef_acc_mag_action": eef_annotation["eef_acc_mag"],
        }
        gripper_annotation = self._annotate_gripper_motion(gripper_open_scale_state)
        if not gripper_annotation:
            gripper_annotation_state = {}
        else:
            gripper_annotation_state = {
                "gripper_mode_state": gripper_annotation["gripper_mode"],
                "gripper_activity_state": gripper_annotation["gripper_activity"],
            }
        gripper_annotation = self._annotate_gripper_motion(gripper_open_scale_action)
        if not gripper_annotation:
            gripper_annotation_action = {}
        else:
            gripper_annotation_action = {
                "gripper_mode_action": gripper_annotation["gripper_mode"],
                "gripper_activity_action": gripper_annotation["gripper_activity"],
            }

        result = (
            eef_annotation_state
            | eef_annotation_action
            | eef_sim_data_state_dict
            | eef_sim_data_action_dict
        )
        if not self.sim_replay_config.has_gripper:
            return result

        return (
            result
            | gripper_annotation_state
            | gripper_annotation_action
            | gripper_open_scale_state_dict
            | gripper_open_scale_action_dict
        )

        # return (
        #     eef_annotation_state
        #     | eef_annotation_action
        #     | gripper_annotation_state
        #     | gripper_annotation_action
        #     | eef_sim_data_state_dict
        #     | eef_sim_data_action_dict
        #     | gripper_open_scale_state_dict
        #     | gripper_open_scale_action_dict
        # )

    def get_output_feature_keys(self) -> set[str]:
        """返回输出数据的特征键（与输入可能不同）"""
        return {}

    def get_modified_feature_names(self) -> dict[str, list[str]]:
        eef_sim_pose_names = [
            "left_eef_pos_x",
            "left_eef_pos_y",
            "left_eef_pos_z",
            "left_eef_rot_x",
            "left_eef_rot_y",
            "left_eef_rot_z",
            "right_eef_pos_x",
            "right_eef_pos_y",
            "right_eef_pos_z",
            "right_eef_rot_x",
            "right_eef_rot_y",
            "right_eef_rot_z",
        ]
        eef_direction_names = [
            "left_eef_direction",
            "right_eef_direction",
        ]
        eef_velocity_names = [
            "left_eef_velocity",
            "right_eef_velocity",
        ]
        eef_acc_mag_names = [
            "left_eef_acc_mag",
            "right_eef_acc_mag",
        ]
        gripper_open_scale_names = [
            "left_gripper_open_scale",
            "right_gripper_open_scale",
        ]
        gripper_mode_names = [
            "left_gripper_mode",
            "right_gripper_mode",
        ]
        gripper_activity_names = [
            "left_gripper_activity",
            "right_gripper_activity",
        ]
        eef_feature_names = {
            "eef_sim_pose_state": eef_sim_pose_names,
            "eef_sim_pose_action": eef_sim_pose_names,
            "eef_direction_state": eef_direction_names,
            "eef_direction_action": eef_direction_names,
            "eef_velocity_state": eef_velocity_names,
            "eef_velocity_action": eef_velocity_names,
            "eef_acc_mag_state": eef_acc_mag_names,
            "eef_acc_mag_action": eef_acc_mag_names,
        }
        if self.sim_replay_config.has_gripper:
            gripper_feature_names = {
                "gripper_open_scale_state": gripper_open_scale_names,
                "gripper_open_scale_action": gripper_open_scale_names,
                "gripper_mode_state": gripper_mode_names,
                "gripper_mode_action": gripper_mode_names,
                "gripper_activity_state": gripper_activity_names,
                "gripper_activity_action": gripper_activity_names,
            }
        else:
            gripper_feature_names = {}
        return eef_feature_names | gripper_feature_names

    def write_new_info_file(self) -> None:
        """重写 write_new_info_file 方法，添加 has_gripper 信息"""
        import json

        json_dict = {}
        json_dict["features"] = {}

        for feature_key, names in self.get_modified_feature_names().items():
            if names is not None:
                if len(names) != len(set(names)):
                    raise ValueError(f"given feature names contain duplicated names: {names}")
            json_dict["features"][feature_key] = {}
            json_dict["features"][feature_key]["names"] = names

        new_info_file_path = get_meta_info_file(self.convert_path, "motion_annotation")
        with open(new_info_file_path, "w") as f:
            json.dump(json_dict, f, indent=2)

    # 该方法将ori_state_data进行后处理，返回结果为后处理后的数据
    def process_episode_state_data(self, ori_state_data: np.ndarray) -> np.ndarray:
        episode_idx = self.episode_idx
        results = self.simulator.replay_episode_background(episode_index=episode_idx, is_state=True)

        eef_data = np.array(results)

        # 计算EEF位姿数据的列数（每个site 6维：pos(3) + ori(3)）
        num_sites = len(self.sim_replay_config.mjcf_site_names)
        num_eef_cols = num_sites * 6

        # 如果没有夹爪，只返回EEF位姿数据
        if not self.sim_replay_config.state_gripper_lerobot_names:
            return eef_data

        # 有夹爪：归一化夹爪数据后返回完整数据（EEF + 夹爪）
        if eef_data.shape[1] > num_eef_cols:
            # 提取夹爪数据（EEF位姿之后的列）
            gripper_data = eef_data[:, num_eef_cols:]

            # 归一化到 [0, 1] 范围
            # 使用公式: (value - min) / (max - min)
            if self.gripper_value_open != self.gripper_value_close:
                gripper_normalized = gripper_data
                # 限制在 [0, 1] 范围内
                gripper_normalized = np.clip(gripper_normalized, 0.0, 1.0)

                # 替换原始夹爪数据
                eef_data[:, num_eef_cols:] = gripper_normalized

        return eef_data

    # 该方法将ori_action_data进行后处理，返回结果为后处理后的数据
    def process_episode_action_data(self, ori_action_data: np.ndarray) -> np.ndarray:
        episode_idx = self.episode_idx
        results = self.simulator.replay_episode_background(
            episode_index=episode_idx, is_state=False
        )

        eef_data = np.array(results)

        # 计算EEF位姿数据的列数（每个site 6维：pos(3) + ori(3)）
        num_sites = len(self.sim_replay_config.mjcf_site_names)
        num_eef_cols = num_sites * 6

        # 如果没有夹爪，只返回EEF位姿数据

        if not self.sim_replay_config.state_gripper_lerobot_names:
            return eef_data

        # 有夹爪：归一化夹爪数据后返回完整数据（EEF + 夹爪）
        if eef_data.shape[1] > num_eef_cols:
            # 提取夹爪数据（EEF位姿之后的列）
            gripper_data = eef_data[:, num_eef_cols:]

            # 归一化到 [0, 1] 范围
            # 使用公式: (value - min) / (max - min)
            if self.gripper_value_open != self.gripper_value_close:
                gripper_normalized = gripper_data
                # 限制在 [0, 1] 范围内
                gripper_normalized = np.clip(gripper_normalized, 0.0, 1.0)

                # 替换原始夹爪数据
                eef_data[:, num_eef_cols:] = gripper_normalized

        # 返回完整数据（EEF + 归一化后的夹爪），在 process_episode_data 中会分离
        return eef_data
