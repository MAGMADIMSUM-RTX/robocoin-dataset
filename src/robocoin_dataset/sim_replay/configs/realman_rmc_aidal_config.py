from pathlib import Path

from robocoin_dataset.sim_replay.configs.lerobot_sim_replay_config import (
    LerobotSimReplayConfig,
)


class RealmanRmcAidalLerobotSimReplayConfig(LerobotSimReplayConfig):
    mjcf_path = (
        Path(__file__).parent
        / "mjcfs/realman_rmc_aidal/overseas_75_b_v_description_rmg24_with_sites.xml"
    )

    left_eef_mjcf_site_name: str = "left_eef_site"
    right_eef_mjcf_site_name: str = "right_eef_site"

    # joint_names should be 1 v 1 assignment
    state_arm_joint_mjcf_names: list[str] = [
        "l_joint1",
        "l_joint2",
        "l_joint3",
        "l_joint4",
        "l_joint5",
        "l_joint6",
        "l_joint7",
        "r_joint1",
        "r_joint2",
        "r_joint3",
        "r_joint4",
        "r_joint5",
        "r_joint6",
        "r_joint7",
    ]
    action_arm_joint_mjcf_names = state_arm_joint_mjcf_names
    state_arm_joint_lerobot_names: list[str] = [
        "left_arm_joint_1_rad",
        "left_arm_joint_2_rad",
        "left_arm_joint_3_rad",
        "left_arm_joint_4_rad",
        "left_arm_joint_5_rad",
        "left_arm_joint_6_rad",
        "left_arm_joint_7_rad",
        "right_arm_joint_1_rad",
        "right_arm_joint_2_rad",
        "right_arm_joint_3_rad",
        "right_arm_joint_4_rad",
        "right_arm_joint_5_rad",
        "right_arm_joint_6_rad",
        "right_arm_joint_7_rad",
    ]

    action_arm_joint_lerobot_names = state_arm_joint_lerobot_names

    # gripper names allow not 1v1 assignment
    state_gripper_joint_mjcf_names: list[str] = [
        "l_Joint_finger1",
        "l_Joint_finger2",
        "r_Joint_finger1",
        "r_Joint_finger2",
    ]
    action_gripper_joint_mjcf_names = state_gripper_joint_mjcf_names

    state_gripper_lerobot_names: list[str] = ["left_gripper_open_scale", "right_gripper_open_scale"]
    action_gripper_lerobot_names = state_gripper_lerobot_names

    gripper_open_max = 1.0
    gripper_open_min = 0
    gripper_joint_max = 0.0325
    gripper_joint_min = 0

    # eef_sim 配置
    has_gripper = True
    gripper_value_open = gripper_open_max
    gripper_value_close = gripper_open_min

    def get_mjcf_gripper_joint_data(self, lerobot_gripper_data: list[float]) -> list[float]:
        left_gripper_data = lerobot_gripper_data[0]
        right_gripper_data = lerobot_gripper_data[1]

        lglf_joint_data = (
            (self.gripper_joint_max - self.gripper_joint_min)
            * (left_gripper_data - self.gripper_open_min)
            / (self.gripper_open_max - self.gripper_open_min)
        )
        lgrf_joint_data = lglf_joint_data
        rglf_joint_data = (
            (self.gripper_joint_max - self.gripper_joint_min)
            * (right_gripper_data - self.gripper_open_min)
            / (self.gripper_open_max - self.gripper_open_min)
        )
        rgrf_joint_data = rglf_joint_data

        return [
            lglf_joint_data,
            lgrf_joint_data,
            rglf_joint_data,
            rgrf_joint_data,
        ]
