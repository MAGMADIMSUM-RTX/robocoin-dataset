from pathlib import Path

from robocoin_dataset.sim_replay.configs.lerobot_sim_replay_config import (
    LerobotSimReplayConfig,
)


class RuantongA2dLerobotSimReplayConfig(LerobotSimReplayConfig):
    mjcf_path = Path(__file__).parent / "mjcfs/tianqing/o2_with_sites.xml"

    left_eef_mjcf_site_name: str = "left_eef_site"
    right_eef_mjcf_site_name: str = "right_eef_site"

    # joint_names should be 1 v 1 assignment
    state_arm_joint_mjcf_names: list[str] = [
        # left arm
        # right arm
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",    
           
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
    ]
    action_arm_joint_mjcf_names = state_arm_joint_mjcf_names
    state_arm_joint_lerobot_names: list[str] = [
        # left arm
        "left_arm_joint_1_rad",
        "left_arm_joint_2_rad",
        "left_arm_joint_3_rad",
        "left_arm_joint_4_rad",
        "left_arm_joint_5_rad",
        "left_arm_joint_6_rad",
        "left_arm_joint_7_rad",
        # right arm
        "right_arm_joint_1_rad",
        "right_arm_joint_2_rad",
        "right_arm_joint_3_rad",
        "right_arm_joint_4_rad",
        "right_arm_joint_5_rad",
        "right_arm_joint_6_rad",
        "right_arm_joint_7_rad",
    ]

    action_arm_joint_lerobot_names = state_arm_joint_lerobot_names

    # eef_sim 配置
    has_gripper = True
    gripper_position_max = 1.0
    gripper_position_min = 0

    gripper_value_open = gripper_position_max
    gripper_value_close = gripper_position_min

    # gripper names allow not 1v1 assignment
    state_gripper_joint_mjcf_names: list[str] = []
    action_gripper_joint_mjcf_names = state_gripper_joint_mjcf_names

    state_gripper_lerobot_names: list[str] = [
        "left_gripper_open_scale",
        "right_gripper_open_scale",
    ]
    action_gripper_lerobot_names = state_gripper_lerobot_names

    def get_mjcf_gripper_joint_data(self, lerobot_gripper_data: list[float]) -> list[float]:
        return []
