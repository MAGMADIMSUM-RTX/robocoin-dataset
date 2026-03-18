import argparse
import logging
from pathlib import Path

# 导入你的模块
from robocoin_dataset.utils.logger import setup_logger
from robocoin_dataset.data_merge.data_merger import DataMerger, DataMergeConfig

def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description='指定UUID测试数据合并功能')
    parser.add_argument('--db_file_path', type=str, required=True, help='数据库文件路径')
    parser.add_argument('--uuid', type=str, required=True, help='要测试的数据集UUID')
    parser.add_argument('--log_dir', type=str, default='./logs/test_merge', help='日志目录')
    
    args = parser.parse_args()
    
    # 2. 初始化日志
    logger = setup_logger(
        name="test_merge_by_uuid",
        log_dir=Path(args.log_dir),
        level=logging.DEBUG  # 调试模式输出详细日志
    )
    
    # 3. 初始化DataMerger
    data_merger = DataMerger(
        db_file_path=args.db_file_path,
        logger=logger,
        data_merge_config=DataMergeConfig()
    )
    
    # 4. 执行指定UUID的合并
    success = data_merger.merge_data_by_uuid(args.uuid)
    
    # 5. 输出结果
    if success:
        logger.info(f"✅ UUID {args.uuid} 合并测试成功")
        exit(0)
    else:
        logger.error(f"❌ UUID {args.uuid} 合并测试失败")
        exit(1)

if __name__ == "__main__":
    main()
