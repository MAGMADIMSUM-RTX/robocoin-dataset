from pathlib import Path

from robocoin_dataset.sim_replay.configs.lerobot_sim_replay_config import (
    LerobotSimReplayConfig,
)


class AlphaBot2LerobotSimReplayConfig(LerobotSimReplayConfig):
    mjcf_path = (
        Path(__file__).parent
        / "mjcfs/ai2robotics_bot2/Bot2_ZPF_L2Fgripper_R2Fgripper_A_description_with_sites.xml"
    )

    # fixed
    left_eef_mjcf_site_name: str = "left_eef_site"
    right_eef_mjcf_site_name: str = "right_eef_site"

    # joint_names should be 1 v 1 assignment
    state_arm_joint_mjcf_names: list[str] = [
        # left arm
        "ZPF_left_Joint1",
        "ZPF_left_Joint2",
        "ZPF_left_Joint3",
        "ZPF_left_Joint4",
        "ZPF_left_Joint5",
        "ZPF_left_Joint6",
        "ZPF_left_Joint7",
        # right arm
        "ZPF_right_Joint1",
        "ZPF_right_Joint2",
        "ZPF_right_Joint3",
        "ZPF_right_Joint4",
        "ZPF_right_Joint5",
        "ZPF_right_Joint6",
        "ZPF_right_Joint7",
    ]
    action_arm_joint_mjcf_names = state_arm_joint_mjcf_names
    state_arm_joint_lerobot_names: list[str] = [
        # left arm
        # "left_arm_joint_0_rad",
        "left_arm_joint_1_rad",
        "left_arm_joint_2_rad",
        "left_arm_joint_3_rad",
        "left_arm_joint_4_rad",
        "left_arm_joint_5_rad",
        "left_arm_joint_6_rad",
        "left_arm_joint_7_rad",
        # right arm
        # "right_arm_joint_0_rad",
        "right_arm_joint_1_rad",
        "right_arm_joint_2_rad",
        "right_arm_joint_3_rad",
        "right_arm_joint_4_rad",
        "right_arm_joint_5_rad",
        "right_arm_joint_6_rad",
        "right_arm_joint_7_rad",
    ]

    action_arm_joint_lerobot_names: list[str] = state_arm_joint_lerobot_names

 # gripper names allow not 1v1 assignment
    state_gripper_joint_mjcf_names: list[str] = []
    action_gripper_joint_mjcf_names = state_gripper_joint_mjcf_names

    gripper_open_max = 1.0
    gripper_open_min = 0

    # eef_sim 配置
    has_gripper = True

    gripper_value_open = gripper_open_max
    gripper_value_close = gripper_open_min


    state_gripper_lerobot_names: list[str] = ["left_gripper_open_scale", "right_gripper_open_scale"]
    action_gripper_lerobot_names: list[str] = [
        "left_gripper_open_scale",
        "right_gripper_open_scale",
    ]
    def get_mjcf_gripper_joint_data(self, lerobot_gripper_data: list[float]) -> list[float]:
        return []
    
