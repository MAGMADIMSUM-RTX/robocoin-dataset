from pathlib import Path

from robocoin_dataset.sim_replay.configs.lerobot_sim_replay_config import (
    LerobotSimReplayConfig,
)


class AgilexCobotMagicLerobotH5Mp4NewSimReplayConfig(LerobotSimReplayConfig):
    mjcf_path = (
        Path(__file__).parent / "mjcfs/agilex_cobot_magic/aloha_new/aloha_new_v00_with_sites.xml"
    )

    left_eef_mjcf_site_name: str = "left_eef_site"
    right_eef_mjcf_site_name: str = "right_eef_site"

    # joint_names should be 1 v 1 assignment
    state_arm_joint_mjcf_names: list[str] = [
        "fl_joint1",
        "fl_joint2",
        "fl_joint3",
        "fl_joint4",
        "fl_joint5",
        "fl_joint6",
        "fr_joint1",
        "fr_joint2",
        "fr_joint3",
        "fr_joint4",
        "fr_joint5",
        "fr_joint6",
    ]
    action_arm_joint_mjcf_names = state_arm_joint_mjcf_names
    state_arm_joint_lerobot_names: list[str] = [
        "left_arm_joint_1_rad",
        "left_arm_joint_2_rad",
        "left_arm_joint_3_rad",
        "left_arm_joint_4_rad",
        "left_arm_joint_5_rad",
        "left_arm_joint_6_rad",
        "right_arm_joint_1_rad",
        "right_arm_joint_2_rad",
        "right_arm_joint_3_rad",
        "right_arm_joint_4_rad",
        "right_arm_joint_5_rad",
        "right_arm_joint_6_rad",
    ]
    action_arm_joint_lerobot_names = state_arm_joint_lerobot_names

    # gripper names allow not 1v1 assignment
    state_gripper_joint_mjcf_names: list[str] = [
        "fl_joint7",
        "fl_joint8",
        "fr_joint7",
        "fr_joint8",
    ]
    action_gripper_joint_mjcf_names = state_gripper_joint_mjcf_names

    state_gripper_lerobot_names: list[str] = ["left_gripper_open_scale", "right_gripper_open_scale"]
    action_gripper_lerobot_names = state_gripper_lerobot_names

    gripper_open_max = 1.0
    gripper_open_min = 0
    gripper_joint_max = 0.04
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
        lgrf_joint_data = -lglf_joint_data
        rglf_joint_data = (
            (self.gripper_joint_max - self.gripper_joint_min)
            * (right_gripper_data - self.gripper_open_min)
            / (self.gripper_open_max - self.gripper_open_min)
        )
        rgrf_joint_data = -rglf_joint_data

        return [
            lglf_joint_data,
            lgrf_joint_data,
            rglf_joint_data,
            rgrf_joint_data,
        ]

class AgilexCobotMagicLerobotMultSenerSimReplayConfig(LerobotSimReplayConfig):
    mjcf_path = (
        Path(__file__).parent / "mjcfs/agilex_cobot_magic/aloha_new/aloha_new_v00_with_sites.xml"
    )

    left_eef_mjcf_site_name: str = "left_eef_site"
    right_eef_mjcf_site_name: str = "right_eef_site"

    # joint_names should be 1 v 1 assignment
    state_arm_joint_mjcf_names: list[str] = [
        "fl_joint1",
        "fl_joint2",
        "fl_joint3",
        "fl_joint4",
        "fl_joint5",
        "fl_joint6",
        "fr_joint1",
        "fr_joint2",
        "fr_joint3",
        "fr_joint4",
        "fr_joint5",
        "fr_joint6",
    ]
    action_arm_joint_mjcf_names = state_arm_joint_mjcf_names
    state_arm_joint_lerobot_names: list[str] = [
        "left_arm_joint_1_rad",
        "left_arm_joint_2_rad",
        "left_arm_joint_3_rad",
        "left_arm_joint_4_rad",
        "left_arm_joint_5_rad",
        "left_arm_joint_6_rad",
        "right_arm_joint_1_rad",
        "right_arm_joint_2_rad",
        "right_arm_joint_3_rad",
        "right_arm_joint_4_rad",
        "right_arm_joint_5_rad",
        "right_arm_joint_6_rad",
    ]
    action_arm_joint_lerobot_names = state_arm_joint_lerobot_names

    # gripper names allow not 1v1 assignment
    state_gripper_joint_mjcf_names: list[str] = [
        "fl_joint7",
        "fl_joint8",
        "fr_joint7",
        "fr_joint8",
    ]
    action_gripper_joint_mjcf_names = state_gripper_joint_mjcf_names

    state_gripper_lerobot_names: list[str] = ["left_gripper_open_scale", "right_gripper_open_scale"]
    action_gripper_lerobot_names = state_gripper_lerobot_names

    gripper_open_max = 1.0
    gripper_open_min = 0
    gripper_joint_max = 0.04
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
        lgrf_joint_data = -lglf_joint_data
        rglf_joint_data = (
            (self.gripper_joint_max - self.gripper_joint_min)
            * (right_gripper_data - self.gripper_open_min)
            / (self.gripper_open_max - self.gripper_open_min)
        )
        rgrf_joint_data = -rglf_joint_data

        return [
            lglf_joint_data,
            lgrf_joint_data,
            rglf_joint_data,
            rgrf_joint_data,
        ]

class AgilexCobotMagicLerobotSimReplayConfig(LerobotSimReplayConfig):
    mjcf_path = (
        Path(__file__).parent / "mjcfs/agilex_cobot_magic/aloha_new/aloha_new_v00_with_sites.xml"
    )

    left_eef_mjcf_site_name: str = "left_eef_site"
    right_eef_mjcf_site_name: str = "right_eef_site"
    left_gripper_mjcf_site_name: str = "left_gripper_site"
    right_gripper_mjcf_site_name: str = "right_gripper_site"

    # joint_names should be 1 v 1 assignment
    state_arm_joint_mjcf_names: list[str] = [
        "fl_joint1",
        "fl_joint2",
        "fl_joint3",
        "fl_joint4",
        "fl_joint5",
        "fl_joint6",
        "fr_joint1",
        "fr_joint2",
        "fr_joint3",
        "fr_joint4",
        "fr_joint5",
        "fr_joint6",
    ]
    action_arm_joint_mjcf_names = state_arm_joint_mjcf_names
    state_arm_joint_lerobot_names: list[str] = [
        "left_arm_joint_1_rad",
        "left_arm_joint_2_rad",
        "left_arm_joint_3_rad",
        "left_arm_joint_4_rad",
        "left_arm_joint_5_rad",
        "left_arm_joint_6_rad",
        "right_arm_joint_1_rad",
        "right_arm_joint_2_rad",
        "right_arm_joint_3_rad",
        "right_arm_joint_4_rad",
        "right_arm_joint_5_rad",
        "right_arm_joint_6_rad",
    ]
    action_arm_joint_lerobot_names = state_arm_joint_lerobot_names

    # gripper names allow not 1v1 assignment
    state_gripper_joint_mjcf_names: list[str] = [
        "fl_joint7",
        "fl_joint8",
        "fr_joint7",
        "fr_joint8",
    ]
    action_gripper_joint_mjcf_names = state_gripper_joint_mjcf_names

    state_gripper_lerobot_names: list[str] = ["left_gripper_open_scale", "right_gripper_open_scale"]
    action_gripper_lerobot_names = state_gripper_lerobot_names

    gripper_open_max = 1.0
    gripper_open_min = 0
    gripper_joint_max = 0.04
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
        lgrf_joint_data = -lglf_joint_data
        rglf_joint_data = (
            (self.gripper_joint_max - self.gripper_joint_min)
            * (right_gripper_data - self.gripper_open_min)
            / (self.gripper_open_max - self.gripper_open_min)
        )
        rgrf_joint_data = -rglf_joint_data

        return [
            lglf_joint_data,
            lgrf_joint_data,
            rglf_joint_data,
            rgrf_joint_data,
        ]

class AgilexCobotMagicLerobotH5Mp4SimReplayConfig(LerobotSimReplayConfig):
    mjcf_path = (
        Path(__file__).parent / "mjcfs/agilex_cobot_magic/aloha_new/aloha_new_v00_with_sites.xml"
    )

    left_eef_mjcf_site_name: str = "left_eef_site"
    right_eef_mjcf_site_name: str = "right_eef_site"

    # joint_names should be 1 v 1 assignment
    state_arm_joint_mjcf_names: list[str] = [
        "fl_joint1",
        "fl_joint2",
        "fl_joint3",
        "fl_joint4",
        "fl_joint5",
        "fl_joint6",
        "fr_joint1",
        "fr_joint2",
        "fr_joint3",
        "fr_joint4",
        "fr_joint5",
        "fr_joint6",
    ]
    action_arm_joint_mjcf_names = state_arm_joint_mjcf_names
    state_arm_joint_lerobot_names: list[str] = [
        "left_arm_joint_1_rad",
        "left_arm_joint_2_rad",
        "left_arm_joint_3_rad",
        "left_arm_joint_4_rad",
        "left_arm_joint_5_rad",
        "left_arm_joint_6_rad",
        "right_arm_joint_1_rad",
        "right_arm_joint_2_rad",
        "right_arm_joint_3_rad",
        "right_arm_joint_4_rad",
        "right_arm_joint_5_rad",
        "right_arm_joint_6_rad",
    ]
    action_arm_joint_lerobot_names = state_arm_joint_lerobot_names

    # gripper names allow not 1v1 assignment
    state_gripper_joint_mjcf_names: list[str] = [
        "fl_joint7",
        "fl_joint8",
        "fr_joint7",
        "fr_joint8",
    ]
    action_gripper_joint_mjcf_names = state_gripper_joint_mjcf_names

    state_gripper_lerobot_names: list[str] = ["left_gripper_open_scale", "right_gripper_open_scale"]
    action_gripper_lerobot_names = state_gripper_lerobot_names

    gripper_open_max = 1.0
    gripper_open_min = 0
    gripper_joint_max = 0.04
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
        lgrf_joint_data = -lglf_joint_data
        rglf_joint_data = (
            (self.gripper_joint_max - self.gripper_joint_min)
            * (right_gripper_data - self.gripper_open_min)
            / (self.gripper_open_max - self.gripper_open_min)
        )
        rgrf_joint_data = -rglf_joint_data

        return [
            lglf_joint_data,
            lgrf_joint_data,
            rglf_joint_data,
            rgrf_joint_data,
        ]

class AgilexCobotMagicLerobotRealsenseSimReplayConfig(LerobotSimReplayConfig):
    mjcf_path = (
        Path(__file__).parent / "mjcfs/agilex_cobot_magic/aloha_new/aloha_new_v00_with_sites.xml"
    )

    left_eef_mjcf_site_name: str = "left_eef_site"
    right_eef_mjcf_site_name: str = "right_eef_site"

    # joint_names should be 1 v 1 assignment
    state_arm_joint_mjcf_names: list[str] = [
        "fl_joint1",
        "fl_joint2",
        "fl_joint3",
        "fl_joint4",
        "fl_joint5",
        "fl_joint6",
        "fr_joint1",
        "fr_joint2",
        "fr_joint3",
        "fr_joint4",
        "fr_joint5",
        "fr_joint6",
    ]
    action_arm_joint_mjcf_names = state_arm_joint_mjcf_names
    state_arm_joint_lerobot_names: list[str] = [
        "left_arm_joint_1_rad",
        "left_arm_joint_2_rad",
        "left_arm_joint_3_rad",
        "left_arm_joint_4_rad",
        "left_arm_joint_5_rad",
        "left_arm_joint_6_rad",
        "right_arm_joint_1_rad",
        "right_arm_joint_2_rad",
        "right_arm_joint_3_rad",
        "right_arm_joint_4_rad",
        "right_arm_joint_5_rad",
        "right_arm_joint_6_rad",
    ]
    action_arm_joint_lerobot_names = state_arm_joint_lerobot_names

    # gripper names allow not 1v1 assignment
    state_gripper_joint_mjcf_names: list[str] = [
        "fl_joint7",
        "fl_joint8",
        "fr_joint7",
        "fr_joint8",
    ]
    action_gripper_joint_mjcf_names = state_gripper_joint_mjcf_names

    state_gripper_lerobot_names: list[str] = ["left_gripper_open_scale", "right_gripper_open_scale"]
    action_gripper_lerobot_names = state_gripper_lerobot_names

    gripper_open_max = 1.0
    gripper_open_min = 0
    gripper_joint_max = 0.04
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
        lgrf_joint_data = -lglf_joint_data
        rglf_joint_data = (
            (self.gripper_joint_max - self.gripper_joint_min)
            * (right_gripper_data - self.gripper_open_min)
            / (self.gripper_open_max - self.gripper_open_min)
        )
        rgrf_joint_data = -rglf_joint_data

        return [
            lglf_joint_data,
            lgrf_joint_data,
            rglf_joint_data,
            rgrf_joint_data,
        ]
