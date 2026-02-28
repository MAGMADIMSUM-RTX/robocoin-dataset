#!/usr/bin/env python3
# scripts/db/separate.py

import os
import sys
import yaml
from pathlib import Path
import argparse
import logging
import shutil

# 自定义 Dumper：将 None 输出为 dataset_uuid:
class NullSafeDumper(yaml.SafeDumper):
    def represent_none(self, data):
        return self.represent_scalar('tag:yaml.org,2002:null', '')

# 注册 None 的表示方式
yaml.add_representer(type(None), NullSafeDumper.represent_none, Dumper=NullSafeDumper)


def setup_logging(log_dir: Path, quiet: bool = False) -> None:
    """设置日志系统"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "separate.log"

    # 清空旧日志
    if log_file.exists():
        log_file.unlink()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台输出
    if not quiet:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(ch)

    # 文件输出
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logging.info("separate.py 脚本启动")


def process_yaml_files(root_dir: Path, dry_run: bool = False) -> dict:
    """
    处理指定目录下所有 local_dataset_info*.y*ml 文件
    返回统计信息
    """
    stats = {
        "processed": 0,
        "with_uuid": 0,
        "created_uuid_file": 0,
        "failed": 0,
        "skipped_no_uuid": 0,
        "backed_up": 0
    }

    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        logging.critical(f"根目录不存在: {root_path}")
        return stats
    if not root_path.is_dir():
        logging.critical(f"路径不是目录: {root_path}")
        return stats

    logging.info(f"开始处理目录: {root_path}")

    for dirpath_str, dirnames, filenames in os.walk(root_path):
        dirpath = Path(dirpath_str)

        # 查找匹配的 YAML 文件
        matching_files = list(dirpath.glob('local_dataset_info*.y*ml'))
        if not matching_files:
            continue

        local_info_file = matching_files[0]
        logging.info(f"处理文件: {local_info_file}")
        stats["processed"] += 1

        try:
            with open(local_info_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if not data:
                logging.warning(f"YAML 文件为空: {local_info_file}")
                continue
        except yaml.YAMLError as e:
            logging.error(f"YAML 解析失败 {local_info_file}: {e}")
            stats["failed"] += 1
            continue
        except Exception as e:
            logging.error(f"读取文件失败 {local_info_file}: {e}")
            stats["failed"] += 1
            continue

        # ======== 1. 提取原始 dataset_uuid ========
        raw_uuid = data.get('dataset_uuid')
        if not raw_uuid or not str(raw_uuid).strip():
            logging.warning(f"跳过：未找到有效 dataset_uuid → {local_info_file}")
            stats["skipped_no_uuid"] += 1
        else:
            logging.info(f"提取到 UUID: {raw_uuid}")
            stats["with_uuid"] += 1

            # 创建 dataset_uuid.yaml（如果不存在）
            uuid_output_file = dirpath / "dataset_uuid.yaml"
            if uuid_output_file.exists():
                logging.debug(f"跳过创建：{uuid_output_file} 已存在")
            else:
                if not dry_run:
                    try:
                        with open(uuid_output_file, 'w', encoding='utf-8') as f:
                            yaml.dump({'dataset_uuid': raw_uuid}, f, allow_unicode=True, indent=2)
                        logging.info(f"已创建 UUID 文件: {uuid_output_file}")
                        stats["created_uuid_file"] += 1
                    except Exception as e:
                        logging.error(f"无法创建 UUID 文件 {uuid_output_file}: {e}")
                        stats["failed"] += 1
                else:
                    logging.info(f"[dry-run] 将创建 UUID 文件: {uuid_output_file}")

        # ======== 2. 替换 task_descriptions 中的 '_' 为空格 ========
        # if 'task_descriptions' in data and isinstance(data['task_descriptions'], list):
        #     original = data['task_descriptions']
        #     data['task_descriptions'] = [
        #         item.replace('_', ' ') if isinstance(item, str) else item
        #         for item in original
        #     ]
        #     changed = [f"'{a}'→'{b}'" for a, b in zip(original, data['task_descriptions']) if a != b]
        #     if changed:
        #         logging.info(f"替换 task_descriptions: {', '.join(changed[:3])}{'...' if len(changed)>3 else ''}")

        # ======== 3. 将 dataset_uuid 设为空（用于留空输出）========
        # if 'dataset_uuid' in data:
        #     data['dataset_uuid'] = None
        #     logging.debug(f"已设 dataset_uuid: None（将输出为 dataset_uuid:）")

        # ======== 4. 写回原文件（带备份）========
        try:
        #    backup_file = local_info_file.with_suffix(local_info_file.suffix + ".bak")
        #    if not backup_file.exists() and not dry_run:
        #        shutil.copy2(local_info_file, backup_file)
        #        logging.debug(f"已创建备份: {backup_file}")
        #        stats["backed_up"] += 1

            if not dry_run:
                with open(local_info_file, 'w', encoding='utf-8') as f:
                    yaml.dump(
                        data,
                        f,
                        allow_unicode=True,
                        sort_keys=False,
                        Dumper=NullSafeDumper,
                        indent=2
                    )
                logging.info(f"已保存修改: {local_info_file}")
            else:
                logging.info(f"[dry-run] 将修改文件: {local_info_file}")

        except Exception as e:
            logging.error(f"写入文件失败 {local_info_file}: {e}")
            stats["failed"] += 1

        # 裁枝：不进入子目录
        dirnames.clear()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="""
        处理 local_dataset_info.yml/.yaml 文件：
        1. 替换 task_descriptions 中的 '_' 为空格
        2. 将 dataset_uuid 设为空（输出为 dataset_uuid:）
        3. 提取原始 UUID 到同目录下的 dataset_uuid.yaml
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "root_dirs",
        nargs="*",
        default=["./datasets"],
        help="要处理的一个或多个根目录（默认: ./datasets）"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="日志输出目录（默认: ./logs）"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="静默模式，只输出错误"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="模拟运行，不修改任何文件"
    )

    args = parser.parse_args()

    root_paths = [Path(p) for p in args.root_dirs] if args.root_dirs else [Path("./datasets")]
    log_dir = Path(args.log_dir)

    # 设置日志
    setup_logging(log_dir, quiet=args.quiet)

    if args.dry_run:
        logging.info("【DRY RUN】模拟运行，不会修改任何文件")

    total_stats = {
        "processed": 0,
        "with_uuid": 0,
        "created_uuid_file": 0,
        "failed": 0,
        "skipped_no_uuid": 0,
        "backed_up": 0
    }

    for root_path in root_paths:
        if not root_path.exists():
            logging.critical(f"指定的目录不存在: {root_path}")
            continue
        if not root_path.is_dir():
            logging.critical(f"指定的路径不是一个目录: {root_path}")
            continue

        logging.info(f"正在处理目录: {root_path}")
        stats = process_yaml_files(root_path, dry_run=args.dry_run)
        for k in total_stats:
            total_stats[k] += stats[k]

    # 输出总统计
    logging.info("   处理完成，总统计结果:")
    logging.info(f"  已处理文件: {total_stats['processed']}")
    logging.info(f"  包含 UUID: {total_stats['with_uuid']}")
    logging.info(f"  新建 dataset_uuid.yaml: {total_stats['created_uuid_file']}")
    logging.info(f"  跳过（无 UUID）: {total_stats['skipped_no_uuid']}")
    logging.info(f"  失败: {total_stats['failed']}")
    logging.info("   脚本执行完毕。")


if __name__ == "__main__":
    main()