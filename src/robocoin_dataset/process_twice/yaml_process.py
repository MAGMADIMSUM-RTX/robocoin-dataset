import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from sqlalchemy import select

# 导入你提供的数据库相关模块
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("yaml_validation.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("YAMLValidator")

# Device Model 校验规则（你提供的JSON格式）
DEVICE_MODEL_RULES = {
    "key": "device_model",
    "label": "本体型号",
    "type": "multiselect",
    "options": {
        "Galbot_G1": "银河通用",
        "Galaxea_R1_Lite": "星海图",
        "Leju_Kuavo_4_LB": "乐聚",
        "Leju_Kuavo_4_Pro": "乐聚",
        "Agilex_Split_Aloha": "松灵合体",
        "Agilex_Cobot_Magic": "松灵分体",
        "Unitree_G1edu-u3": "宇树",
        "Airbot_MMK2": "求知",
        "Tianqing_A2": "软通天擎",
        "Tianqing_A2D": "软通天擎",
        "AI2_Alphabot_1s": "智平方",
        "AI2_Alphabot_2": "智平方",
        "Realman_RMC-AIDA-L": "睿尔曼",
        "Realman_Rs-01": "睿尔曼",
        "Realman_Rs-02": "睿尔曼",
        "Agilex_Pika_Touch": "pika触觉",
        "Robotera_Q5": "星动纪元"
    }
}

class YAMLValidator:
    def __init__(self, db_file: str | Path):
        """
        初始化YAML校验器
        :param db_file: 数据库文件路径
        """
        self.db_file = Path(db_file).expanduser().absolute()
        self.db = DatasetDatabase(db_file=self.db_file)
        self.valid_device_models = set(DEVICE_MODEL_RULES["options"].keys())
        self.validation_results = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "errors": []
        }

    def load_yaml_file(self, yaml_path: str | Path) -> Optional[Dict]:
        """
        加载YAML文件并处理异常
        :param yaml_path: YAML文件路径
        :return: YAML内容字典，加载失败返回None
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            error_msg = f"YAML文件不存在: {yaml_path}"
            logger.error(error_msg)
            return None
        
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            error_msg = f"YAML文件解析失败 {yaml_path}: {str(e)}"
            logger.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"读取YAML文件失败 {yaml_path}: {str(e)}"
            logger.error(error_msg)
            return None

    def validate_device_model(self, yaml_content: Dict, yaml_path: str) -> Tuple[bool, str]:
        """
        校验device_model字段是否在可选列表中
        :param yaml_content: YAML内容字典
        :param yaml_path: YAML文件路径（用于日志）
        :return: (是否通过, 错误信息)
        """
        # 检查device_model字段是否存在
        if "device_model" not in yaml_content:
            return False, "缺少device_model字段"
        
        device_models = yaml_content["device_model"]
        # 处理单个值和列表值两种情况
        if not isinstance(device_models, list):
            device_models = [device_models]
        
        # 检查每个值是否在可选列表中
        invalid_models = [dm for dm in device_models if dm not in self.valid_device_models]
        if invalid_models:
            return False, f"无效的device_model值: {invalid_models}，可选值: {sorted(self.valid_device_models)}"
        
        return True, ""

    def validate_dataset_name(self, yaml_content: Dict, yaml_path: str) -> Tuple[bool, str, Optional[str]]:
        """
        校验并返回修正后的值（新增返回修正值）
        :return: (是否通过, 提示信息, 修正后的值/None)
        """
        if "dataset_name" not in yaml_content:
            return False, "缺少dataset_name字段", None
        
        dataset_name = yaml_content["dataset_name"]
        if not isinstance(dataset_name, str):
            return False, f"dataset_name必须是字符串类型，当前类型: {type(dataset_name)}", None
        
        corrected_name = None  # 新增：存储修正后的值
        if "device_model" in yaml_content:
            device_models = yaml_content["device_model"]
            if not isinstance(device_models, list):
                device_models = [device_models]
            
            matched_device_model = None
            for dm in device_models:
                if dataset_name.startswith(f"{dm}_"):
                    matched_device_model = dm
                    break
            
            if matched_device_model:
                corrected_name = dataset_name[len(matched_device_model) + 1:]
                if not corrected_name:
                    return False, f"dataset_name去除device_model前缀后为空，原始值: {dataset_name}", None
                return True, f"检测到device_model前缀，建议修正值: {corrected_name}", corrected_name
        
        # 无前缀时，修正值为原名称
        return True, "未检测到device_model前缀，校验通过（无需修正）", dataset_name

    def validate_single_yaml(self, yaml_path: str, auto_fix: bool = True) -> bool:
        """
        新增auto_fix参数：是否自动修正并保存（修改前强制备份）
        """
        self.validation_results["total"] += 1
        logger.info(f"开始校验YAML文件: {yaml_path}")
        
        yaml_path = Path(yaml_path)  # 转为Path对象，方便备份操作
        yaml_content = self.load_yaml_file(yaml_path)
        if not yaml_content:
            error_detail = {
                "yaml_path": str(yaml_path),
                "error_type": "文件加载失败",
                "error_msg": "文件不存在或解析失败"
            }
            self.validation_results["errors"].append(error_detail)
            self.validation_results["failed"] += 1
            return False
        
        # 校验device_model
        dm_valid, dm_msg = self.validate_device_model(yaml_content, str(yaml_path))
        # 校验dataset_name（接收修正后的值）
        dn_valid, dn_msg, corrected_name = self.validate_dataset_name(yaml_content, str(yaml_path))
        
        # 新增：自动修正逻辑（核心：先备份，再修改）
        if dm_valid and dn_valid and auto_fix and corrected_name is not None and corrected_name != yaml_content.get("dataset_name"):
            # ========== 步骤1：强制备份原文件（优先级最高） ==========
            backup_success = True
            # 备份文件命名：原文件名 + .bak + 时间戳 + 原后缀（避免重复）
            backup_path = yaml_path.with_suffix(f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}{yaml_path.suffix}")
            try:
                # copy2保留文件元信息（权限、创建时间等），比普通copy更安全
                shutil.copy2(yaml_path, backup_path)
                logger.info(f"✅ 已备份原文件到: {backup_path}")
            except Exception as e:
                backup_success = False
                logger.error(f"❌ 备份原文件失败 {yaml_path}: {str(e)}")
            
            # 只有备份成功，才执行修改操作
            if backup_success:
                # ========== 步骤2：修改并写入文件 ==========
                try:
                    # 修改内存中的dataset_name
                    yaml_content["dataset_name"] = corrected_name
                    # 写入原文件（保持格式和原文件一致）
                    with open(yaml_path, "w", encoding="utf-8") as f:
                        yaml.dump(
                            yaml_content,
                            f,
                            indent=2,           # 匹配原文件2个空格缩进
                            sort_keys=False,    # 不打乱原字段顺序
                            default_flow_style=False  # 保持块式格式，避免流式错乱
                        )
                    logger.info(f"✅ 已自动修正并保存dataset_name: {yaml_path} -> {corrected_name}")
                except Exception as e:
                    logger.error(f"❌ 自动保存修正后的YAML文件失败 {yaml_path}: {str(e)}")
            else:
                # 备份失败则跳过修改，绝对不碰原文件
                logger.warning(f"⚠️  备份失败，跳过文件修正: {yaml_path}")
        
        # 后续汇总逻辑不变
        if dm_valid and dn_valid:
            logger.info(f"✅ YAML文件校验通过: {yaml_path}")
            logger.info(f"  - device_model: 校验通过")
            logger.info(f"  - dataset_name: {dn_msg}")
            self.validation_results["success"] += 1
            return True
        else:
            # 失败逻辑不变
            error_detail = {
                "yaml_path": str(yaml_path),
                "error_type": "字段校验失败",
                "device_model_error": dm_msg if not dm_valid else "无",
                "dataset_name_error": dn_msg if not dn_valid else "无"
            }
            self.validation_results["errors"].append(error_detail)
            self.validation_results["failed"] += 1
            
            logger.error(f"❌ YAML文件校验失败: {yaml_path}")
            if not dm_valid:
                logger.error(f"  - device_model: {dm_msg}")
            if not dn_valid:
                logger.error(f"  - dataset_name: {dn_msg}")
            return False

    def validate_all_datasets(self) -> Dict:
        """
        校验数据库中所有is_ignore=false且有YAML路径的数据集的YAML文件
        :return: 校验结果汇总
        """
        logger.info("开始校验所有is_ignore=false的数据集的YAML文件...")
        
        with self.db.with_session() as session:
            # 修复数据库查询报错：只查询需要的字段，避免触达不存在的列
            stmt = select(DatasetDB.id, DatasetDB.yaml_file_path).where(
                DatasetDB.yaml_file_path.isnot(None),
                DatasetDB.is_ignore == False
            )
            result = session.execute(stmt).all()
            
            # 整理数据集路径
            datasets = [{"yaml_file_path": row.yaml_file_path} for row in result]
            logger.info(f"共找到 {len(datasets)} 个is_ignore=false且有YAML路径的数据集")
            
            # 逐个校验
            for dataset in datasets:
                yaml_path = dataset["yaml_file_path"]
                if not yaml_path:
                    continue
                self.validate_single_yaml(yaml_path)
        
        # 生成汇总报告
        self._generate_report()
        return self.validation_results

    def _generate_report(self):
        """
        生成并打印校验报告
        """
        logger.info("=" * 80)
        logger.info("YAML文件校验报告")
        logger.info("=" * 80)
        logger.info(f"总校验文件数: {self.validation_results['total']}")
        logger.info(f"通过数: {self.validation_results['success']}")
        logger.info(f"失败数: {self.validation_results['failed']}")
        logger.info(f"失败率: {self.validation_results['failed']/max(self.validation_results['total'], 1)*100:.2f}%")
        
        if self.validation_results["errors"]:
            logger.info("\n失败详情:")
            for i, error in enumerate(self.validation_results["errors"], 1):
                logger.info(f"{i}. 文件: {error['yaml_path']}")
                if error["error_type"] == "文件加载失败":
                    logger.info(f"   错误: {error['error_msg']}")
                else:
                    logger.info(f"   device_model错误: {error['device_model_error']}")
                    logger.info(f"   dataset_name错误: {error['dataset_name_error']}")
                logger.info("-" * 40)
        
        # 保存报告到文件
        with open("yaml_validation_report.json", "w", encoding="utf-8") as f:
            json.dump(self.validation_results, f, ensure_ascii=False, indent=2)
        logger.info(f"\n详细报告已保存到: yaml_validation_report.json")


# 执行示例
if __name__ == "__main__":
    # 替换为你的数据库文件路径
    DB_FILE_PATH = "/home/liuyou/Documents/robocoin-dataset/db/postgresql_config.yaml"
    
    # 初始化校验器
    validator = YAMLValidator(db_file=DB_FILE_PATH)
    
    # 执行全量校验
    results = validator.validate_all_datasets()