import logging
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# 数据库依赖
from sqlalchemy.orm import Session
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB, TaskStatus
import json
import numpy as np
import pandas as pd

# ====================== 日志配置（终端+文件） ======================
DB_FILE_PATH = "/home/liuyou/Documents/robocoin-dataset/db/postgresql_config.yaml"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dataset_validation_report5.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ====================== 核心调度逻辑 ======================
def run_validation_on_all_datasets():
    db = DatasetDatabase(Path(DB_FILE_PATH).expanduser().absolute())
    # 🔥 存储：正常/异常数据集的 UUID + 设备型号
    valid_datasets: List[Dict] = []   # {"uuid": str, "device_model": str}
    invalid_datasets: List[Dict] = [] # {"uuid": str, "device_model": str}

    with db.with_session() as session:
        datasets: List[DatasetDB] = (
            session.query(DatasetDB)
            .filter(DatasetDB.data_merge_status == TaskStatus.COMPLETED)
            .filter(DatasetDB.is_ignore == False)  # 只校验未被忽略的数据
            .all()
        )

        if not datasets:
            logger.info("✅ 数据库中无待校验的数据集")
            return

        logger.info(f"📊 共找到 {len(datasets)} 个转换完成的数据集，开始批量校验...\n")

        for idx, ds in enumerate(datasets, 1):
            dataset_uuid = ds.dataset_uuid
            convert_path = ds.qced_repo_gen_path
            device_model = ds.device_model.strip() if ds.device_model else "未知设备"  # 读取设备型号

            logger.info(f"===== 正在校验第 {idx}/{len(datasets)} 个数据集 =====")
            logger.info(f"UUID: {dataset_uuid}")
            logger.info(f"设备型号: {device_model}")
            logger.info(f"路径: {convert_path}")

            # ====================== 标记异常数据：自动设置 is_ignore=True ======================
            def mark_as_invalid_and_ignore(error_desc: str):
                """封装异常处理：记录+更新数据库is_ignore=True"""
                invalid_datasets.append({"uuid": dataset_uuid, "device_model": device_model})
                ds.is_ignore = True  # 核心修改：标记为忽略
                session.commit()      # 提交数据库更新
                logger.error(f"❌ {error_desc}")
                logger.info(f"✅ 已自动标记该数据集 is_ignore = True，后续流程将自动跳过\n")

            # 路径无效 → 标记异常+忽略
            if not convert_path or not Path(convert_path).exists():
                mark_as_invalid_and_ignore("数据集异常：路径不存在/为空，已跳过")
                continue

            try:
                # 🔥 快速校验：发现1个异常立即返回
                validator = LerobotDatasetValidator(convert_path)
                is_valid, error_msg = validator.run()

                if is_valid:
                    # 正常数据集：不修改任何字段
                    valid_datasets.append({"uuid": dataset_uuid, "device_model": device_model})
                    logger.info(f"✅ 数据集校验通过\n")
                else:
                    # 校验失败 → 标记异常+忽略
                    mark_as_invalid_and_ignore(f"数据集存在异常，已跳过该数据集！异常信息：{error_msg[:200]}...")

            except Exception as e:
                # 程序异常 → 标记异常+忽略
                mark_as_invalid_and_ignore(f"校验程序异常，已跳过该数据集：{str(e)}")

    # 🔥 生成最终报告（UUID + 设备型号占比）
    generate_final_report(valid_datasets, invalid_datasets)


def calculate_device_ratio(dataset_list: List[Dict]) -> Dict:
    """
    统计设备型号的数量和占比
    :return: {设备型号: {"count": 数量, "ratio": 占比}}
    """
    total = len(dataset_list)
    device_count = defaultdict(int)
    for item in dataset_list:
        device_count[item["device_model"]] += 1

    # 计算占比
    device_stats = {}
    for device, count in device_count.items():
        ratio = (count / total * 100) if total > 0 else 0.0
        device_stats[device] = {"count": count, "ratio": round(ratio, 2)}
    return device_stats


def generate_final_report(valid_ds: List[Dict], invalid_ds: List[Dict]):
    """生成最终汇总报告：UUID + 设备型号占比统计"""
    total_count = len(valid_ds) + len(invalid_ds)
    valid_count = len(valid_ds)
    invalid_count = len(invalid_ds)

    # 统计设备占比
    valid_device_stats = calculate_device_ratio(valid_ds)
    invalid_device_stats = calculate_device_ratio(invalid_ds)

    # ====================== 基础统计 ======================
    logger.info("=" * 120)
    logger.info("📋 数据集校验最终汇总报告（含设备型号占比 + 已标记异常数据is_ignore=True）")
    logger.info("=" * 120)
    logger.info(f"📊 总计校验数据集：{total_count} 个")
    logger.info(f"✅ 正常数据集：{valid_count} 个")
    logger.info(f"❌ 异常数据集（已标记is_ignore=True）：{invalid_count} 个")
    logger.info("-" * 120)

    # ====================== 正常数据集 设备型号统计 ======================
    logger.info("\n📡 【正常数据集】设备型号统计：")
    if valid_device_stats:
        for device, stats in valid_device_stats.items():
            logger.info(f"   {device}：{stats['count']} 个，占比 {stats['ratio']}%")
    else:
        logger.info("   无正常数据集")

    # ====================== 异常数据集 设备型号统计 ======================
    logger.info("\n📡 【异常数据集】设备型号统计：")
    if invalid_device_stats:
        for device, stats in invalid_device_stats.items():
            logger.info(f"   {device}：{stats['count']} 个，占比 {stats['ratio']}%")
    else:
        logger.info("   无异常数据集")

    # ====================== UUID 列表 ======================
    logger.info("\n" + "-" * 120)
    logger.info("\n✅ 正常数据集 UUID 列表：")
    for item in valid_ds:
        logger.info(f"   [{item['device_model']}] {item['uuid']}")

    logger.info("\n❌ 异常数据集 UUID 列表（已标记is_ignore=True）：")
    for item in invalid_ds:
        logger.info(f"   [{item['device_model']}] {item['uuid']}")

    logger.info("=" * 120)


# ==================== 快速失败版校验器（发现1个异常立即停止） ====================
class LerobotDatasetValidator:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path).expanduser().absolute()
        self.info_path = self.dataset_path / "meta" / "info.json"
        self.data_dir = self.dataset_path / "data"
        if not self.info_path.exists():
            self.info_path = self.dataset_path / "info.json"
        self._validate_paths()
        self.info_data = self._load_info_json()
        self.expected_features = self.info_data.get('features', {})
        self.feature_names_map = self._extract_feature_names()

        # 物理限制
        self.FINE_GRAINED_BOUNDS = {
            "_rad": {"min": -3.1415926, "max": 3.1415926},
            "_m": {"min": -2.0, "max": 2.0},
            "gripper_open_scale": {"min": 0.0, "max": 1.0},
            "gripper_open": {"min": 0.0, "max": 1010.0},
        }
        self.GLOBAL_FALLBACK_BOUNDS = {"min": -10.0, "max": 10.0}

    def _validate_paths(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
        if not self.info_path.exists():
            raise FileNotFoundError(f"找不到 info.json: {self.info_path}")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

    def _load_info_json(self) -> dict:
        with open(self.info_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_feature_names(self) -> Dict[str, List[str]]:
        names_map = {}
        for feat_name, feat_info in self.expected_features.items():
            if "names" in feat_info and isinstance(feat_info["names"], list):
                names_map[feat_name] = feat_info["names"]
        return names_map

    def _check_single_parquet_fast(self, file_path: Path) -> str:
        """🔥 快速检查：发现第一个错误立即返回错误信息，不再继续"""
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            return f"Parquet读取失败: {str(e)}"

        for feature_name, feature_info in self.expected_features.items():
            if not (feature_name.startswith("action") or feature_name.startswith("observation.state")):
                continue
            if feature_name not in df.columns:
                return f"缺失特征列: {feature_name}"

            expected_shape = tuple(feature_info.get('shape', []))
            dtype = feature_info.get('dtype')
            if not expected_shape or dtype in ['video', 'image']:
                continue

            try:
                np_data = np.vstack(df[feature_name].values)
            except ValueError:
                return f"特征 {feature_name} 数据格式错乱"

            # 维度检查
            actual_shape = np_data.shape[1:]
            if actual_shape != expected_shape:
                return f"{feature_name} 维度不匹配: 预期{expected_shape} 实际{actual_shape}"

            # NaN/Inf检查
            if np_data.dtype.kind in 'fc':
                if np.isnan(np_data).any() or np.isinf(np_data).any():
                    return f"{feature_name} 存在NaN/Inf异常值"

            # 物理极值检查（快速失败）
            if feature_name in self.feature_names_map:
                sub_names = self.feature_names_map[feature_name]
                for col_idx, sub_name in enumerate(sub_names):
                    col_data_np = np_data[:, col_idx]
                    matched_bounds = None
                    for keyword, bounds in self.FINE_GRAINED_BOUNDS.items():
                        if keyword in sub_name:
                            matched_bounds = bounds
                            break
                    if matched_bounds:
                        hard_min, hard_max = matched_bounds["min"], matched_bounds["max"]
                        if np.any(col_data_np < hard_min - 1e-4) or np.any(col_data_np > hard_max + 1e-4):
                            return f"{feature_name}.{sub_name} 超出物理极值范围"
            else:
                # 兜底检查
                actual_min, actual_max = np.min(np_data), np.max(np_data)
                if actual_min < self.GLOBAL_FALLBACK_BOUNDS["min"] or actual_max > self.GLOBAL_FALLBACK_BOUNDS["max"]:
                    return f"{feature_name} 存在飞点数据"

        return ""

    def run(self) -> tuple[bool, str]:
        """🔥 快速失败模式：检查到第一个异常立即返回结果"""
        parquet_files = list(self.data_dir.rglob("*.parquet"))
        if not parquet_files:
            return False, "无Parquet文件"

        # 遍历文件，发现错误立即返回
        for file in parquet_files:
            error = self._check_single_parquet_fast(file)
            if error:
                return False, f"文件{file.name}: {error}"

        return True, ""

if __name__ == "__main__":
    run_validation_on_all_datasets()
