import argparse
import asyncio
import logging
from pathlib import Path
import yaml 

from robocoin_dataset.quality_check.qced_repo_generator import QualityCheckedRepoGeneratorServer
from robocoin_dataset.utils.logger import setup_logger


def load_yaml_config(config_path: str | Path) -> dict:
    """读取YAML配置文件，返回字典"""
    config_path = Path(config_path).expanduser().absolute()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 确保返回的是字典（防止YAML为空）
    return config if isinstance(config, dict) else {}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_file_path",
        type=str,
        default="",
        help="Path to the database file",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Path to the log directory",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8768,
        help="Port to run the server",
    )
    parser.add_argument(
        "--qc_config_path",
        type=str,
        default="./configs/qc_repo_config.yaml",
        help="Path to the quality check YAML config file",
    )
    parser.add_argument(
        "--target_dataset_uuid",
        type=str,
        default=None,
        help="Specify the target dataset UUID to generate qced repo (optional).",
    )

    parser.add_argument("--ds_api_key", type=str, default="sk-a3c8736391cf43809957329f28cac287")

    args = parser.parse_args()
    db_file_path = Path(args.db_file_path).expanduser().absolute()

    try:
        qc_config = load_yaml_config(args.qc_config_path)
        logging.info(f"成功读取配置文件: {args.qc_config_path}")
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        exit(1)

    # 3. 合并配置：命令行参数 > YAML配置（用get方法保证安全）
    final_config = qc_config.copy()

    if not db_file_path.exists():
        print(f"{db_file_path} does not exist")
        exit(1)

    logger = setup_logger(
        name="quality checked repo generator server",
        log_dir=Path(args.log_dir),
        level=logging.INFO,
    )

    server = QualityCheckedRepoGeneratorServer(
        db_file_path=db_file_path,
        host=args.host,
        port=args.port,
        logger=logger,
        qc_config=final_config,
        ds_api_key=args.ds_api_key,
        target_dataset_uuid=args.target_dataset_uuid,
    )

    await server.start()


if __name__ == "__main__":
    asyncio.run(main())

"""usage:
python scripts/quality_check/qced_repo_gen_server.py \
    --db_file_path ./db/datasets_new.db \
    --host 0.0.0.0 \
    --port 2110 \
    --min_episodes_num 5 \
    --log_dir ./logs/quality_checked_repo_generator_server 
"""
