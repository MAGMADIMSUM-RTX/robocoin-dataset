import json
import logging
from pathlib import Path
from typing import Dict, Optional

from sqlalchemy import select

# 导入你提供的数据库相关模块
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB

# 配置日志（和原脚本保持一致）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("convert_path_check.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ConvertPathChecker")

class ConvertPathChecker:
    def __init__(self, db_file: str | Path):
        """
        初始化路径检查器
        :param db_file: 数据库文件路径
        """
        self.db_file = Path(db_file).expanduser().absolute()
        self.db = DatasetDatabase(db_file=self.db_file)
        self.check_results = {
            "total": 0,                  # 总检查数据集数量
            "exists_count": 0,           # 路径存在的数量
            "not_exists_count": 0,       # 路径不存在的数量
            "empty_path_count": 0,       # convert_path为空的数量
            "errors": []                 # 详细错误信息（含ID和路径）
        }

    def check_convert_path_exists(self, dataset_id: int, convert_path: str) -> bool:
        """
        检查单个convert_path是否真实存在
        :param dataset_id: 数据集ID（用于日志和报告）
        :param convert_path: 要检查的路径
        :return: 路径是否存在
        """
        self.check_results["total"] += 1
        
        # 处理空路径情况
        if not convert_path or convert_path.strip() == "":
            self.check_results["empty_path_count"] += 1
            error_detail = {
                "dataset_id": dataset_id,
                "convert_path": convert_path,
                "error_type": "空路径",
                "error_msg": "convert_path字段为空或空白字符"
            }
            self.check_results["errors"].append(error_detail)
            logger.warning(f"⚠️  数据集ID={dataset_id} | convert_path为空")
            return False
        
        # 检查路径是否存在
        path_obj = Path(convert_path)
        if path_obj.exists():
            self.check_results["exists_count"] += 1
            logger.info(f"✅ 数据集ID={dataset_id} | convert_path存在: {convert_path}")
            return True
        else:
            self.check_results["not_exists_count"] += 1
            error_detail = {
                "dataset_id": dataset_id,
                "convert_path": convert_path,
                "error_type": "路径不存在",
                "error_msg": f"convert_path路径不存在: {convert_path}"
            }
            self.check_results["errors"].append(error_detail)
            logger.error(f"❌ 数据集ID={dataset_id} | convert_path不存在: {convert_path}")
            return False

    def check_all_datasets(self) -> Dict:
        """
        检查所有is_ignore=false的数据集的convert_path路径
        :return: 检查结果汇总
        """
        logger.info("开始检查所有is_ignore=false的数据集的convert_path路径...")
        
        with self.db.with_session() as session:
            # 查询数据集ID和convert_path字段（仅is_ignore=false）
            stmt = select(DatasetDB.id, DatasetDB.convert_path).where(
                DatasetDB.is_ignore == False
            )
            result = session.execute(stmt).all()
            
            # 整理数据集信息（含ID，方便定位问题）
            datasets = [{"id": row.id, "convert_path": row.convert_path} for row in result]
            logger.info(f"共找到 {len(datasets)} 个is_ignore=false的数据集")
            
            # 逐个检查convert_path
            for dataset in datasets:
                dataset_id = dataset["id"]
                convert_path = dataset["convert_path"]
                self.check_convert_path_exists(dataset_id, convert_path)
        
        # 生成检查报告
        self.generate_report()
        return self.check_results

    def generate_report(self):
        """
        生成并打印检查报告（和原脚本风格一致）
        """
        logger.info("=" * 80)
        logger.info("Convert_Path 路径检查报告")
        logger.info("=" * 80)
        logger.info(f"总检查数据集数量: {self.check_results['total']}")
        logger.info(f"路径存在的数量: {self.check_results['exists_count']}")
        logger.info(f"路径不存在的数量: {self.check_results['not_exists_count']}")
        logger.info(f"convert_path为空的数量: {self.check_results['empty_path_count']}")
        logger.info(f"路径不存在率: {self.check_results['not_exists_count']/max(self.check_results['total'], 1)*100:.2f}%")
        
        # 打印详细错误信息
        if self.check_results["errors"]:
            logger.info("\n详细错误信息:")
            for i, error in enumerate(self.check_results["errors"], 1):
                logger.info(f"{i}. 数据集ID: {error['dataset_id']}")
                logger.info(f"   路径: {error['convert_path']}")
                logger.info(f"   错误类型: {error['error_type']}")
                logger.info(f"   错误信息: {error['error_msg']}")
                logger.info("-" * 40)
        
        # 保存报告到JSON文件（方便后续处理）
        with open("convert_path_check_report.json", "w", encoding="utf-8") as f:
            json.dump(self.check_results, f, ensure_ascii=False, indent=2)
        logger.info(f"\n检查报告已保存到: convert_path_check_report.json")


# 执行示例（和原脚本保持一致）
if __name__ == "__main__":
    # 替换为你的数据库文件路径
    DB_FILE_PATH = "/home/liuyou/Documents/robocoin-dataset/db/postgresql_config.yaml"
    
    # 初始化检查器
    checker = ConvertPathChecker(db_file=DB_FILE_PATH)
    
    # 执行全量检查
    results = checker.check_all_datasets()
