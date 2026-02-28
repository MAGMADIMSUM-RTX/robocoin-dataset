from pathlib import Path

from robocoin_dataset.sim_replay.configs.lerobot_sim_replay_config import (
    LerobotSimReplayConfig,
)


class GalbotLerobotSimReplayConfig(LerobotSimReplayConfig):
    mjcf_path = (
        Path(__file__).parent
        / "mjcfs/galbot/robot_with_sites.xml"
    )

    # fixed
    left_eef_mjcf_site_name: str = "left_eef_site"
    right_eef_mjcf_site_name: str = "right_eef_site"

    # joint_names should be 1 v 1 assignment
    state_arm_joint_mjcf_names: list[str] = [
        # left arm
        "left_arm_joint1",
        "left_arm_joint2",
        "left_arm_joint3",
        "left_arm_joint4",
        "left_arm_joint5",
        "left_arm_joint6",
        "left_arm_joint7",
        # right arm
        "right_arm_joint1",
        "right_arm_joint2",
        "right_arm_joint3",
        "right_arm_joint4",
        "right_arm_joint5",
        "right_arm_joint6",
        "right_arm_joint7",
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


    gripper_position_max = 1.0
    gripper_position_min = 0.0
    # eef_sim 配置
    has_gripper = True
    gripper_value_open = gripper_position_max
    gripper_value_close = gripper_position_min



    action_arm_joint_lerobot_names= state_arm_joint_lerobot_names
    
     # gripper names allow not 1v1 assignment
    state_gripper_joint_mjcf_names: list[str] = []
    action_gripper_joint_mjcf_names = state_gripper_joint_mjcf_names

    state_gripper_lerobot_names: list[str] = [
        # left_hand
        "left_gripper_open_scale",
        # right hand
        "right_gripper_open_scale",
    ]
    action_gripper_lerobot_names = state_gripper_lerobot_names

    def get_mjcf_gripper_joint_data(self, lerobot_gripper_data: list[float]) -> list[float]:
        return []
    