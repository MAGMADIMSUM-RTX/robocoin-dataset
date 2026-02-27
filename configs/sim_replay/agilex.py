class CustomFix:
    def __init__(self):
        # 示例参数，根据需要设置
        self.gripper_joint_max = 1
        self.gripper_joint_min = 0.0
        self.gripper_open_max = 1
        self.gripper_open_min = 0.0

    def get_mjcf_gripper_joint_data(self, lerobot_gripper_data: dict) -> dict:
        left_gripper_data = lerobot_gripper_data.get("left_gripper_open", 0.0)
        right_gripper_data = lerobot_gripper_data.get("right_gripper_open", 0.0)

        lglf_joint_data = left_gripper_data
        lgrf_joint_data = -lglf_joint_data
        rglf_joint_data = right_gripper_data
        rgrf_joint_data = -rglf_joint_data

        return {
            **lerobot_gripper_data,
            "fl_joint7": lglf_joint_data,
            "fl_joint8": lgrf_joint_data,
            "fr_joint7": rglf_joint_data,
            "fr_joint8": rgrf_joint_data,
        }
