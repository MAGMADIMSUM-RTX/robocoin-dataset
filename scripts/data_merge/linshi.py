import json
import os
from pathlib import Path

# ===================== 你给的所有目标目录 =====================
TARGET_DIRS = [
    "/mnt/nas/synnas/成功区/Agilex_Cobot_Magic_storage_towel",
    "/mnt/nas/synnas/成功区/Agilex_Cobot_Magic_fold_towel_blue",
    "/mnt/nas/synnas/成功区/Agilex_Cobot_Magic_fold_towel_purple",
    "/mnt/nas/synnas/成功区/Agilex_Cobot_Magic_fold_towel_brown",
    "/mnt/nas/synnas/成功区/Agilex_Cobot_Magic_heat_burger",
    "/mnt/nas/synnas/成功区/Agilex_Cobot_Magic_make_sandwiche",
    "/mnt/nas/synnas/成功区/Agilex_Cobot_Magic_storage_peach",
    "/mnt/nas/synnas/成功区/Agilex_Cobot_Magic_sweep_coffee_bean",
    "/mnt/nas/synnas/成功区/Agilex_Cobot_Magic_place_towel_flat",
    "/mnt/nas/synnas/成功区/Agilex_Cobot_Magic_heat_sandwich",
]

# 你要的固定 names 顺序
TARGET_NAMES = [
    "left_arm_joint_1_rad",
    "left_arm_joint_2_rad",
    "left_arm_joint_3_rad",
    "left_arm_joint_4_rad",
    "left_arm_joint_5_rad",
    "left_arm_joint_6_rad",
    "left_eef_pos_x_m",
    "left_eef_pos_y_m",
    "left_eef_pos_z_m",
    "left_eef_rot_euler_x_rad",
    "left_eef_rot_euler_y_rad",
    "left_eef_rot_euler_z_rad",
    "left_gripper_open",
    "right_arm_joint_1_rad",
    "right_arm_joint_2_rad",
    "right_arm_joint_3_rad",
    "right_arm_joint_4_rad",
    "right_arm_joint_5_rad",
    "right_arm_joint_6_rad",
    "right_eef_pos_x_m",
    "right_eef_pos_y_m",
    "right_eef_pos_z_m",
    "right_eef_rot_euler_x_rad",
    "right_eef_rot_euler_y_rad",
    "right_eef_rot_euler_z_rad",
    "right_gripper_open",
]
# ===============================================================

def process_single_info(json_path: Path):
    # 读取
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 只修改 action 和 observation.state
    if "features" in data:
        if "action" in data["features"]:
            data["features"]["action"]["names"] = TARGET_NAMES
        if "observation.state" in data["features"]:
            data["features"]["observation.state"]["names"] = TARGET_NAMES

    # 覆盖保存（原地修改，最干净）
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 已处理：{json_path}")

# 遍历所有目录
for base_dir in TARGET_DIRS:
    info_path = Path(base_dir) / "meta" / "info.json"
    if info_path.exists():
        process_single_info(info_path)
    else:
        print(f"⚠️  不存在：{info_path}")

print("\n🎉 所有目录的 info.json 处理完成！")
