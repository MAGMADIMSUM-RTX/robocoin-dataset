import logging
from pathlib import Path
import json
from sqlalchemy.orm import Session
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB

# ====================== 配置 ======================
DB_FILE_PATH = "/home/liuyou/Documents/robocoin-dataset/db/postgresql_config.yaml"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ====================== 核心脚本 ======================
def update_dataset_from_info_json():
    db = DatasetDatabase(Path(DB_FILE_PATH).expanduser().absolute())
    
    with db.with_session() as session:
        # 1. 仅查询 【未被忽略】 的数据集 (is_ignore != True)
        all_datasets: list[DatasetDB] = session.query(DatasetDB).filter(
            DatasetDB.is_ignore == False,
        ).all()
        
        if not all_datasets:
            logger.info("数据库中无【未被忽略】的数据集")
            return

        logger.info("=" * 120)
        logger.info(f"开始遍历更新 {len(all_datasets)} 个【有效数据集】，从info.json同步字段到数据库")
        logger.info("更新字段：converted_episodes | fps | total_time")
        logger.info("=" * 120)

        update_count = 0  # 统计成功更新条数
        total_valid_seconds = 0.0  # 累计：未忽略数据集的总时长(秒)
        total_converted_episodes = 0  # 新增：累计总converted_episodes

        # 2. 遍历每个有效数据集
        for ds in all_datasets:
            dataset_uuid = ds.dataset_uuid
            convert_path = ds.convert_path

            # ------------------------------
            # 第一步：校验路径和info.json
            # ------------------------------
            if not convert_path:
                logger.info(f"[跳过] {dataset_uuid} | convert_path 为空")
                continue

            dataset_root = Path(convert_path)
            if not dataset_root.exists():
                logger.info(f"[跳过] {dataset_uuid} | 路径不存在: {convert_path}")
                continue

            info_path = dataset_root / "meta" / "info.json"
            if not info_path.exists():
                logger.info(f"[跳过] {dataset_uuid} | info.json 不存在")
                continue

            # ------------------------------
            # 第二步：解析info.json
            # ------------------------------
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info_data = json.load(f)

                # 提取关键字段
                total_episodes = info_data.get("total_episodes")
                total_frames = info_data.get("total_frames")
                fps = info_data.get("fps")

                # 校验必须字段
                if not all(isinstance(v, (int, float)) for v in [total_episodes, total_frames, fps]):
                    logger.info(f"[跳过] {dataset_uuid} | info.json 字段格式错误")
                    continue
                if fps <= 0:
                    logger.info(f"[跳过] {dataset_uuid} | fps 无效: {fps}")
                    continue

                # 计算总时长（秒）
                total_time = round(total_frames / fps, 2)

            except Exception as e:
                logger.info(f"[跳过] {dataset_uuid} | 解析json失败: {str(e)[:50]}")
                continue

            # ------------------------------
            # 第三步：更新数据库字段
            # ------------------------------
            try:
                ds.converted_episodes = int(total_episodes)
                ds.fps = int(fps)
                ds.total_time = float(total_time)

                session.commit()
                update_count += 1
                # 累加有效时长（仅统计未忽略+更新成功的数据）
                total_valid_seconds += total_time
                total_converted_episodes += total_episodes  # 新增：累加总片段数
                logger.info(f"[更新成功] {dataset_uuid} | episodes={total_episodes} | fps={fps} | 时长={total_time}s")

            except Exception as e:
                session.rollback()
                logger.info(f"[更新失败] {dataset_uuid} | 数据库写入错误: {str(e)[:50]}")

        # ------------------------------
        # 最终统计：总时长 + 总片段数
        # ------------------------------
        total_valid_hours = round(total_valid_seconds / 3600, 2)
        logger.info("=" * 120)
        logger.info(f"更新完成！总计处理：{len(all_datasets)} 条 | 成功更新：{update_count} 条")
        logger.info(f"📊 【未忽略数据集】统计汇总：")
        logger.info(f"   总片段数(converted_episodes)：{total_converted_episodes} 个")  # 新增输出
        logger.info(f"   总秒数：{round(total_valid_seconds, 2)} 秒")
        logger.info(f"   总小时数：{total_valid_hours} 小时")
        logger.info("=" * 120)


if __name__ == "__main__":
    update_dataset_from_info_json()
