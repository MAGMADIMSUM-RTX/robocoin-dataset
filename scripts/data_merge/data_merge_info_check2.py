import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import os

# 导入项目原有数据库模型和工具
from sqlalchemy.orm import Session
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB, TaskStatus

# ====================== 全局配置 ======================
# 数据库配置
DB_FILE_PATH = "/home/liuyou/Documents/robocoin-dataset/db/postgresql_config.yaml"

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 固定常量
CHUNK_ID = "000"
REQUIRED_TOP_FIELDS = [
    "codebase_version", "robot_type", "total_episodes", "total_frames",
    "total_tasks", "total_videos", "total_chunks", "chunks_size", "fps",
    "splits", "data_path", "video_path", "features"
]
CORE_FOLDERS = ["data", "meta", "videos", "annotations"]
META_REQUIRED_FILES = ["episodes.jsonl", "episodes_stats.jsonl", "info.json", "tasks.jsonl"]

# ====================== 命名校验规则配置 ======================
VALIDATION_CONFIG = {
    "cam_name": {
        "valid_positions": [
            "left", "right", "front", "rear", "upper", 
            "lower", "middle", "top", "side", "global", "env", "ego",
        ],
        "valid_parts": [
            "wrist", "head", "chest", "arm", "leg", "torso", "fisheye",
        ],
        "encoding_map": ["rgb", "depth"]
    },
    "state_action_names": {
        "pattern": "^(left|right)_(arm_joint_\\d+_rad|hand_joint_\\d+_rad|gripper_open|eef_pos_[xyz]_m|eef_rot_euler_[xyz]_rad)$",
        "valid_prefixes": ["left", "right"],
        "valid_types": [
            "arm_joint_1_rad", "arm_joint_2_rad", "arm_joint_3_rad", 
            "arm_joint_4_rad", "arm_joint_5_rad", "arm_joint_6_rad",
            "hand_joint_1_rad", "hand_joint_2_rad", "hand_joint_3_rad",
            "hand_joint_4_rad", "hand_joint_5_rad", "hand_joint_6_rad",
            "hand_joint_7_rad", "hand_joint_8_rad", "hand_joint_9_rad",
            "hand_joint_10_rad", "hand_joint_11_rad", "hand_joint_12_rad",
            "gripper_open",
            "eef_pos_x_m", "eef_pos_y_m", "eef_pos_z_m",
            "eef_rot_euler_x_rad", "eef_rot_euler_y_rad", "eef_rot_euler_z_rad"
        ],
    }
}
# 预编译正则表达式
STATE_ACTION_PATTERN = re.compile(VALIDATION_CONFIG["state_action_names"]["pattern"])

# ====================== 相机命名替换规则（核心修复规则） ======================
REPLACE_MAP = {
    "cam_third_view": "cam_front_rgb",
    "ego_view": "cam_ego_rgb",
    "image_top": "cam_head_rgb",
    "image_right": "cam_right_wrist_rgb",
    "image_left": "cam_left_wrist_rgb",
    "cam_center_down_rgb": "cam_front_chest_rgb",
}
HIGH_PATTERN = re.compile(r"high")
# ====================================================


# ====================== 核心工具函数 ======================
def rename_cam_str(text: str) -> str:
    """统一相机命名替换：先精确替换，再high→head（无备份，直接替换）"""
    if not text:
        return text
    # 1. 精确匹配替换
    for old, new in REPLACE_MAP.items():
        text = text.replace(old, new)
    # 2. 通配替换：所有high → head
    text = HIGH_PATTERN.sub("head", text)
    return text

def recursive_rename_dict_keys(data: Dict) -> Dict:
    """递归替换字典键名（适配JSON嵌套结构）"""
    new_dict = {}
    for k, v in data.items():
        new_k = rename_cam_str(k)
        if isinstance(v, dict):
            new_v = recursive_rename_dict_keys(v)
        elif isinstance(v, list):
            new_v = [recursive_rename_dict_keys(item) if isinstance(item, dict) else item for item in v]
        else:
            new_v = v
        new_dict[new_k] = new_v
    return new_dict

def stop_on_error(msg: str):
    """修改失败：打印错误并立即终止程序（核心要求）"""
    logger.error(f"❌ 修改失败，程序终止：{msg}")
    raise SystemExit(1)

# ====================== 数据集修复函数（仅修改videos文件夹） ======================
def repair_dataset(root_dir: Path):
    """
    修复单个数据集的所有命名问题
    ✅ 仅修改：videos/下相机文件夹、info.json、episodes_stats.jsonl
    ❌ 完全不修改：data文件夹、任何Parquet文件/列名
    任何步骤失败 → 立即终止程序，无备份直接修改
    """
    logger.info(f"🔧 开始修复数据集：{root_dir}")

    # -------------------------- 1. 仅重命名 videos/ 下的相机文件夹 --------------------------
    try:
        dir_path = root_dir / "videos"
        if dir_path.exists():
            # 逆序遍历，防止重命名后遍历异常
            for parent, dir_names, _ in os.walk(dir_path, topdown=False):
                for old_name in dir_names:
                    new_name = rename_cam_str(old_name)
                    if old_name != new_name:
                        old_path = Path(parent) / old_name
                        new_path = Path(parent) / new_name
                        logger.info(f"📂 重命名文件夹：{old_name} → {new_name}")
                        old_path.rename(new_path)
    except Exception as e:
        stop_on_error(f"videos文件夹重命名失败：{str(e)}")

    # -------------------------- 2. 修复 meta/info.json --------------------------
    info_path = root_dir / "meta" / "info.json"
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info_data = json.load(f)
        new_info = recursive_rename_dict_keys(info_data)
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(new_info, f, ensure_ascii=False, indent=2)
        logger.info(f"📄 修复 info.json 完成")
    except Exception as e:
        stop_on_error(f"info.json 修改失败：{str(e)}")

    # -------------------------- 3. 修复 meta/episodes_stats.jsonl --------------------------
    stats_path = root_dir / "meta" / "episodes_stats.jsonl"
    try:
        if stats_path.exists():
            new_lines = []
            with open(stats_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        new_lines.append(line)
                        continue
                    data = json.loads(line)
                    new_data = recursive_rename_dict_keys(data)
                    new_lines.append(json.dumps(new_data, ensure_ascii=False))
            with open(stats_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")
            logger.info(f"📄 修复 episodes_stats.jsonl 完成")
    except Exception as e:
        stop_on_error(f"episodes_stats.jsonl 修改失败：{str(e)}")

    logger.info(f"✅ 数据集修复完成：{root_dir}\n")

# ====================== 核心命名校验函数 ======================
def validate_cam_name(cam_name: str) -> Tuple[bool, str]:
    """校验摄像头名称格式（复用你提供的规则）"""
    cam_config = VALIDATION_CONFIG["cam_name"]
    parts = cam_name.split("_")
    
    if len(parts) < 3 or parts[0] != "cam":
        return False, f"格式错误：必须以 cam_xxx 开头，示例：cam_left_wrist_rgb"
    
    encoding = parts[-1]
    if encoding not in cam_config["encoding_map"]:
        return False, f"无效模态：{encoding}，支持：{cam_config['encoding_map']}"
    
    position_and_part_parts = parts[1:-1]
    if not position_and_part_parts:
        return False, "缺少位置信息"
    
    part = None
    position_parts = position_and_part_parts
    for i in range(len(position_and_part_parts)-1, -1, -1):
        if position_and_part_parts[i] in cam_config["valid_parts"]:
            part = position_and_part_parts[i]
            position_parts = position_and_part_parts[:i]
            break
    
    # 已修复：删除行尾多余冒号
    invalid_positions = [p for p in position_parts if p not in cam_config["valid_positions"]]
    if invalid_positions:
        return False, f"无效方位词：{invalid_positions}，支持：{cam_config['valid_positions']}"
    
    if part and part not in cam_config["valid_parts"]:
        return False, f"无效部位：{part}，支持：{cam_config['valid_parts']}"
    
    return True, "校验通过"

def validate_state_action_names(name_list: List[str], field_name: str) -> Tuple[bool, List[str]]:
    """校验 state/action 命名规范"""
    invalid_names = []
    for name in name_list:
        if not STATE_ACTION_PATTERN.match(name):
            invalid_names.append(name)
    return len(invalid_names) == 0, invalid_names

# ====================== 基础校验函数 ======================
def check_core_folders(root_dir: Path) -> Tuple[bool, List[str]]:
    missing_folders = [f for f in CORE_FOLDERS if not (root_dir / f).is_dir()]
    return not missing_folders, missing_folders

def check_meta_files(meta_dir: Path) -> Tuple[bool, List[str]]:
    missing_files = [f for f in META_REQUIRED_FILES if not (meta_dir / f).is_file()]
    return not missing_files, missing_files

def check_info_json_compliance(info_path: Path) -> Tuple[bool, List[str], Dict]:
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info_data = json.load(f)
    except Exception as e:
        return False, ["读取失败："+str(e)], {}
    
    missing_top = [f for f in REQUIRED_TOP_FIELDS if f not in info_data]
    return not missing_top, missing_top, info_data

def check_video_folder_naming(videos_dir: Path, info_features: Dict) -> Tuple[bool, List[str], List[str]]:
    video_chunk_dir = videos_dir / f"chunk-{CHUNK_ID}"
    if not video_chunk_dir.exists():
        return False, ["视频文件夹不存在"], []
    
    expected = [k for k in info_features if k.startswith("observation.images.")]
    actual = [item.name for item in video_chunk_dir.iterdir() if item.is_dir()]
    
    missing = [f for f in expected if f not in actual]
    extra = [f for f in actual if f not in expected]
    return not (missing or extra), missing, extra

# ====================== 数据集单例校验 ======================
def validate_single_dataset(dataset_uuid: str, data_path: str) -> Dict:
    result = {
        "uuid": dataset_uuid, "data_path": data_path,
        "overall_status": "PASS", "details": {}
    }
    root_dir = Path(data_path).expanduser().absolute()
    logger.info(f"\n开始校验数据集: {dataset_uuid}")

    # 1. 核心文件夹校验
    ok, missing = check_core_folders(root_dir)
    result["details"]["core_folders"] = {"status": "PASS" if ok else "FAIL", "missing": missing}

    # 2. Meta文件校验
    meta_dir = root_dir / "meta"
    ok, missing = check_meta_files(meta_dir)
    result["details"]["meta_files"] = {"status": "PASS" if ok else "FAIL", "missing": missing}

    # 3. Info.json 基础校验
    info_path = meta_dir / "info.json"
    ok, missing_top, info_data = check_info_json_compliance(info_path)
    result["details"]["info_json"] = {"status": "PASS" if ok else "FAIL", "missing": missing_top}
    if not ok:
        return result

    features = info_data.get("features", {})

    # 4. 摄像头名称校验
    cam_errors = {}
    for feat_key in features:
        if feat_key.startswith("observation.images."):
            cam_name = feat_key.replace("observation.images.", "")
            ok, msg = validate_cam_name(cam_name)
            if not ok:
                cam_errors[cam_name] = msg
    result["details"]["camera_naming"] = {
        "status": "PASS" if not cam_errors else "FAIL",
        "errors": cam_errors
    }

    # 5. observation.state 命名校验
    state_ok, state_invalid = True, []
    if "observation.state" in features:
        state_names = features["observation.state"].get("names", [])
        state_ok, state_invalid = validate_state_action_names(state_names, "state")
    result["details"]["state_naming"] = {
        "status": "PASS" if state_ok else "FAIL",
        "invalid_names": state_invalid
    }

    # 6. action 命名校验
    action_ok, action_invalid = True, []
    if "action" in features:
        action_names = features["action"].get("names", [])
        action_ok, action_invalid = validate_state_action_names(action_names, "action")
    result["details"]["action_naming"] = {
        "status": "PASS" if action_ok else "FAIL",
        "invalid_names": action_invalid
    }

    # 7. 视频文件夹命名匹配校验
    ok, missing, extra = check_video_folder_naming(root_dir / "videos", features)
    result["details"]["video_folders"] = {"status": "PASS" if ok else "FAIL", "missing": missing, "extra": extra}

    # 最终状态判定
    for item in result["details"].values():
        if item["status"] == "FAIL":
            result["overall_status"] = "FAIL"
            break

    return result

# ====================== 批量校验 + 自动修复 + 失败即停 ======================
def validate_and_repair_datasets():
    db = DatasetDatabase(Path(DB_FILE_PATH).expanduser().absolute())

    with db.with_session() as session:
        datasets = session.query(DatasetDB)\
            .filter(DatasetDB.data_merge_status == TaskStatus.COMPLETED)\
            .filter(DatasetDB.is_ignore == False).all()

        if not datasets:
            logger.info("无已完成合并的数据集")
            return
        
        logger.info(f"找到 {len(datasets)} 个待校验数据集")
        
        # 遍历校验+修复，失败立即终止
        for ds in datasets:
            if not ds.qced_repo_gen_path:
                stop_on_error(f"数据集 {ds.dataset_uuid} 路径为空")
            
            # 1. 校验数据集
            res = validate_single_dataset(ds.dataset_uuid, ds.qced_repo_gen_path)
            
            # 2. 校验失败 → 执行修复
            if res["overall_status"] == "FAIL":
                repair_dataset(Path(ds.qced_repo_gen_path))
                # 修复后重新校验
                logger.info("🔍 修复后重新校验...")
                recheck_res = validate_single_dataset(ds.dataset_uuid, ds.qced_repo_gen_path)
                if recheck_res["overall_status"] == "FAIL":
                    stop_on_error(f"数据集 {ds.dataset_uuid} 修复后仍不规范")

    logger.info("\n🎉 所有数据集校验+修复完成，无异常！")

if __name__ == "__main__":
    validate_and_repair_datasets()
