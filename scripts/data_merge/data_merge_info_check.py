import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

# 导入项目原有数据库模型和工具
from sqlalchemy.orm import Session
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB, TaskStatus

# ====================== 配置项 ======================
# 数据库文件路径（请根据实际情况修改）
DB_FILE_PATH = "/home/liuyou/Documents/robocoin-dataset/db/postgresql_config.yaml"
# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 复用原有代码的必填字段定义
REQUIRED_TOP_FIELDS = [
    "codebase_version", "robot_type", "total_episodes", "total_frames",
    "total_tasks", "total_videos", "total_chunks", "chunks_size", "fps",
    "splits", "data_path", "video_path", "features"
]

# 核心文件夹检查列表
CORE_FOLDERS = ["data", "meta", "videos", "annotations"]
# meta文件夹下核心文件检查列表
META_REQUIRED_FILES = ["episodes.jsonl", "episodes_stats.jsonl", "info.json", "tasks.jsonl"]
# 固定Chunk ID
CHUNK_ID = "000"
# ====================================================


def check_core_folders(root_dir: Path) -> Tuple[bool, List[str]]:
    """检查核心文件夹是否存在"""
    missing_folders = []
    for folder in CORE_FOLDERS:
        folder_path = root_dir / folder
        if not folder_path.exists() or not folder_path.is_dir():
            missing_folders.append(folder)
    return len(missing_folders) == 0, missing_folders


def check_meta_files(meta_dir: Path) -> Tuple[bool, List[str]]:
    """检查meta文件夹下核心文件是否存在"""
    missing_files = []
    for file in META_REQUIRED_FILES:
        file_path = meta_dir / file
        if not file_path.exists() or not file_path.is_file():
            missing_files.append(file)
    return len(missing_files) == 0, missing_files


def check_info_json_compliance(info_path: Path) -> Tuple[bool, List[str], List[str], Dict]:
    """检查info.json的字段合规性，返回：是否合规、缺失顶级字段、缺失features字段、info数据"""
    missing_top = []
    info_data = {}

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info_data = json.load(f)
    except Exception as e:
        logger.error(f"读取info.json失败: {info_path}, 错误: {str(e)}")
        return False, ["读取失败"], [], info_data

    # 检查顶级字段
    missing_top = [f for f in REQUIRED_TOP_FIELDS if f not in info_data]

    return len(missing_top) == 0, missing_top, [], info_data


def check_first_parquet_columns(data_dir: Path, expected_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    【动态校验】校验第一个parquet文件的列
    校验规则：必须包含 info.json -> features 中 非图像类 的所有列名
    """
    parquet_path = data_dir / f"chunk-{CHUNK_ID}" / "episode_000000.parquet"
    missing_cols = []

    if not parquet_path.exists():
        return False, ["文件不存在"]

    try:
        # 仅读取列名，不加载数据，速度极快
        df = pd.read_parquet(parquet_path, columns=None)
        parquet_cols = list(df.columns)

        # 动态校验：必须包含过滤后的所有列
        missing_cols = [col for col in expected_columns if col not in parquet_cols]

    except Exception as e:
        logger.error(f"读取parquet失败: {parquet_path}, 错误: {str(e)}")
        return False, ["文件读取失败"]

    return len(missing_cols) == 0, missing_cols


def check_video_folder_naming(videos_dir: Path, info_features: Dict) -> Tuple[bool, List[str], List[str]]:
    """
    校验：videos/chunk-000/ 下的文件夹命名
    必须与 info.json 中 observation.images.xxx 完全匹配
    """
    video_chunk_dir = videos_dir / f"chunk-{CHUNK_ID}"
    missing_folders = []
    extra_folders = []

    if not video_chunk_dir.exists():
        return False, ["视频chunk文件夹不存在"], []

    # 1. 从info.json提取所有图像key（observation.images.xxx → 文件夹名）
    expected_folder_names = []
    for key in info_features.keys():
        if key.startswith("observation.images."):
            expected_folder_names.append(key)

    # 2. 获取实际文件夹名
    actual_folders = []
    for item in video_chunk_dir.iterdir():
        if item.is_dir():
            actual_folders.append(item.name)

    # 3. 对比校验
    missing_folders = [f for f in expected_folder_names if f not in actual_folders]
    extra_folders = [f for f in actual_folders if f not in expected_folder_names]

    return len(missing_folders) == 0 and len(extra_folders) == 0, missing_folders, extra_folders


def validate_single_dataset(dataset_uuid: str, data_path: str) -> Dict:
    """校验单个数据集的完整性，返回校验结果"""
    result = {
        "uuid": dataset_uuid,
        "data_path": data_path,
        "overall_status": "PASS",
        "details": {}
    }
    root_dir = Path(data_path).expanduser().absolute()
    logger.info(f"\n开始校验数据集: {dataset_uuid}, 路径: {root_dir}")

    # 1. 检查核心文件夹
    folder_pass, missing_folders = check_core_folders(root_dir)
    result["details"]["core_folders"] = {
        "status": "PASS" if folder_pass else "FAIL",
        "missing": missing_folders
    }

    # 2. 检查meta文件夹下文件
    meta_dir = root_dir / "meta"
    file_pass, missing_files = check_meta_files(meta_dir)
    result["details"]["meta_files"] = {
        "status": "PASS" if file_pass else "FAIL",
        "missing": missing_files
    }

    # 3. 检查info.json字段
    info_path = meta_dir / "info.json"
    info_pass, missing_top, _, info_data = check_info_json_compliance(info_path)
    result["details"]["info_json"] = {
        "status": "PASS" if info_pass else "FAIL",
        "missing_top_fields": missing_top
    }

    # ====================== 核心修改 ======================
    # 动态获取features列 + 【自动跳过所有 observation.images. 开头的视频列】
    all_features = list(info_data.get("features", {}).keys())
    expected_parquet_columns = [
        col for col in all_features
        if not col.startswith("observation.images.")
    ]
    # ======================================================

    # 4. 【动态校验】parquet文件列名（已过滤图像列）
    parquet_pass, missing_parquet_cols = check_first_parquet_columns(
        root_dir / "data",
        expected_parquet_columns
    )
    result["details"]["parquet_columns_check"] = {
        "status": "PASS" if parquet_pass else "FAIL",
        "missing_columns": missing_parquet_cols,
        "validated_columns": expected_parquet_columns  # 实际校验的列（非图像）
    }

    # 5. 校验videos文件夹命名（只校验图像列）
    video_pass, missing_video_folders, extra_video_folders = check_video_folder_naming(
        root_dir / "videos",
        info_data.get("features", {})
    )
    result["details"]["video_folder_naming"] = {
        "status": "PASS" if video_pass else "FAIL",
        "missing_folders": missing_video_folders,
        "extra_folders": extra_video_folders
    }

    # 判定整体状态
    for check_item in result["details"].values():
        if check_item["status"] == "FAIL":
            result["overall_status"] = "FAIL"
            break

    logger.info(f"数据集 {dataset_uuid} 校验完成, 整体状态: {result['overall_status']}")
    return result


def validate_completed_datasets(db_file_path: str) -> List[Dict]:
    """主函数：查询所有合并完成的数据集并执行校验"""
    # 初始化数据库连接
    db = DatasetDatabase(Path(db_file_path).expanduser().absolute())
    validation_results = []

    with db.with_session() as session:
        # 查询 data_merge_status == COMPLETED 的数据集
        completed_datasets: List[DatasetDB] = (
            session.query(DatasetDB)
            .filter(DatasetDB.data_merge_status == TaskStatus.COMPLETED)
            .filter(DatasetDB.is_ignore == False)
            .all()
        )

        if not completed_datasets:
            logger.info("数据库中未找到【合并完成】的数据集")
            return validation_results

        logger.info(f"找到 {len(completed_datasets)} 个合并完成的数据集，开始批量校验...")

        # 遍历校验每个数据集
        for ds in completed_datasets:
            ds_uuid = ds.dataset_uuid
            # 获取数据路径（合并完成后的路径：qced_repo_gen_path）
            data_path = ds.qced_repo_gen_path

            if not data_path:
                logger.error(f"数据集 {ds_uuid} 数据路径为空，跳过校验")
                validation_results.append({
                    "uuid": ds_uuid,
                    "data_path": "",
                    "overall_status": "FAIL",
                    "details": {"error": "数据路径为空"}
                })
                continue

            # 执行校验
            result = validate_single_dataset(ds_uuid, data_path)
            validation_results.append(result)

    return validation_results


def generate_validation_report(results: List[Dict]):
    """生成可视化校验报告"""
    logger.info("\n" + "="*80)
    logger.info("数据集合并完成完整性校验报告")
    logger.info("="*80)

    total = len(results)
    pass_num = sum(1 for r in results if r["overall_status"] == "PASS")
    fail_num = total - pass_num

    logger.info(f"\n汇总：总数据集 {total} 个 | 校验通过 {pass_num} 个 | 校验失败 {fail_num} 个\n")

    # 输出失败详情
    for res in results:
        if res["overall_status"] == "FAIL":
            logger.info(f"❌ 失败数据集: {res['uuid']} | 路径: {res['data_path']}")
            for check_name, check_res in res["details"].items():
                if check_res["status"] == "FAIL":
                    logger.info(f"   - {check_name}: {check_res}")
            logger.info("-"*60)
        else:
            logger.info(f"✅ 通过数据集: {res['uuid']}")

    logger.info("\n" + "="*80)


if __name__ == "__main__":
    # 执行校验
    validation_results = validate_completed_datasets(DB_FILE_PATH)
    # 生成报告
    generate_validation_report(validation_results)
