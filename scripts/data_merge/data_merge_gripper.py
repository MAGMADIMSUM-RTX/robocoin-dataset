import logging
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# 数据库依赖（仅读取，无任何修改）
from sqlalchemy.orm import Session
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB, TaskStatus
import json
import numpy as np

# ====================== 核心配置 ======================
DB_FILE_PATH = "/home/liuyou/Documents/robocoin-dataset/db/postgresql_config.yaml"
# 归一化合法范围：0.0 ~ 1.0
GRIPPER_VALUE_MIN = 0.0
GRIPPER_VALUE_MAX = 1.0
# 正确校验字段
TARGET_FIELDS = ["gripper_open_scale_state", "gripper_open_scale_action"]
# 正确的字段结构
REQUIRED_NAMES = ["left_gripper_open_scale", "right_gripper_open_scale"]
# 🔥 已取消 shape 强制校验
# REQUIRED_SHAPE = [2]

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gripper_scale_check_report.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ====================== 核心校验逻辑 ======================
def validate_gripper_normalization(dataset_path: str) -> tuple[bool, str]:
    """
    仅校验：夹爪开合度字段是否正确归一化到 0~1
    返回：(校验通过, 错误信息/成功信息)
    """
    root_path = Path(dataset_path).expanduser().absolute()

    # 1. 查找 info.json
    info_path = root_path / "meta" / "info.json"
    if not info_path.exists():
        info_path = root_path / "info.json"
    if not info_path.exists():
        return False, "缺失 info.json"

    # 2. 读取并校验 info.json 结构
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info_data = json.load(f)
        features = info_data.get("features", {})
    except Exception as e:
        return False, f"info.json 解析失败：{str(e)}"

    # 🔥 已移除 shape 校验，仅校验：字段存在 + 名称匹配
    for field in TARGET_FIELDS:
        if field not in features:
            return False, f"缺失字段：{field}"
        field_conf = features[field]
        # 仅校验 names 匹配，不再校验 shape
        if field_conf.get("names") != REQUIRED_NAMES:
            return False, f"{field} 名称不匹配标准夹爪开合度字段"

    # 3. 查找统计文件校验数值范围
    stats_path = root_path / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        return False, "缺失 episodes_stats.jsonl"

    try:
        with open(stats_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                return False, "统计文件为空"
            stats_data = json.loads(first_line)
            stats = stats_data.get("stats", {})
    except:
        return False, "统计文件解析失败"

    # 4. 核心校验：数值必须在 0~1 之间（保留不变）
    for field in TARGET_FIELDS:
        if field not in stats:
            return False, f"统计数据无{field}"
        
        field_stats = stats[field]
        min_val = field_stats["min"][0] if isinstance(field_stats["min"], list) else field_stats["min"]
        max_val = field_stats["max"][0] if isinstance(field_stats["max"], list) else field_stats["max"]

        # 超出 0~1 判定为归一化异常
        if min_val < GRIPPER_VALUE_MIN - 1e-4 or max_val > GRIPPER_VALUE_MAX + 1e-4:
            return False, f"{field} 归一化异常！值范围 [{min_val:.3f}, {max_val:.3f}]，应在 [0,1]"

    return True, "✅ 夹爪开合度字段归一化校验通过（0~1）"

# ====================== 批量校验（仅读取，无数据库修改） ======================
def run_gripper_check():
    db = DatasetDatabase(Path(DB_FILE_PATH).expanduser().absolute())
    # 存储校验结果
    valid_list: List[Dict] = []
    invalid_list: List[Dict] = []

    with db.with_session() as session:
        # 🔥 已增加过滤条件：is_ignore == False
        datasets = session.query(DatasetDB)\
            .filter(DatasetDB.data_merge_status == TaskStatus.COMPLETED)\
            .filter(DatasetDB.is_ignore == False)\
            .all()

        if not datasets:
            logger.info("📊 数据库中无待校验数据集")
            return

        total = len(datasets)
        logger.info(f"🚀 开始校验夹爪开合度归一化 | 总计数据集：{total}\n")

        for idx, ds in enumerate(datasets, 1):
            uuid = ds.dataset_uuid
            path = ds.qced_repo_gen_path
            device = ds.device_model.strip() if ds.device_model else "未知设备"

            logger.info(f"===== {idx}/{total} | {device} | UUID: {uuid} =====")

            # 路径检查
            if not path or not Path(path).exists():
                invalid_list.append({"uuid": uuid, "device": device, "reason": "数据集路径不存在"})
                logger.error("❌ 路径无效\n")
                continue

            # 执行校验
            is_ok, msg = validate_gripper_normalization(path)
            if is_ok:
                valid_list.append({"uuid": uuid, "device": device})
                logger.info(f"{msg}\n")
            else:
                invalid_list.append({"uuid": uuid, "device": device, "reason": msg})
                logger.error(f"❌ {msg}\n")

    # 输出最终统计报告
    print_check_report(valid_list, invalid_list)

# ====================== 输出详细统计报告 ======================
def print_check_report(valid, invalid):
    total = len(valid) + len(invalid)
    logger.info("=" * 100)
    logger.info("📋 夹爪开合度(gripper_open_scale)归一化校验报告 | 0~1 | 无数据库修改")
    logger.info("=" * 100)
    logger.info(f"📊 总计校验：{total} 个")
    logger.info(f"✅ 归一化正常：{len(valid)} 个")
    logger.info(f"❌ 归一化异常：{len(invalid)} 个")
    logger.info("-" * 100)

    # 按设备型号统计
    valid_dev = defaultdict(int)
    invalid_dev = defaultdict(int)
    for v in valid: valid_dev[v["device"]] += 1
    for i in invalid: invalid_dev[i["device"]] += 1

    logger.info("\n📡 正常数据集设备统计：")
    for dev, cnt in valid_dev.items():
        logger.info(f"   {dev}：{cnt} 个")

    logger.info("\n📡 异常数据集设备统计：")
    for dev, cnt in invalid_dev.items():
        logger.info(f"   {dev}：{cnt} 个")

    logger.info("\n🔴 所有异常数据集详情（UUID+异常原因）：")
    for item in invalid:
        logger.info(f"[{item['device']}] {item['uuid']} | 原因：{item['reason']}")

    logger.info("=" * 100)

if __name__ == "__main__":
    run_gripper_check()
