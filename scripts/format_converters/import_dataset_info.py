#!/usr/bin/env python3
# scripts/db/import_dataset_info_from_yaml.py
import argparse
import concurrent.futures
import logging
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import func  # 关键导入

from robocoin_dataset.database.check_duplicates import (
    get_or_create_uuid,
)
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB  # 新增导入
from robocoin_dataset.database.services.dataset_info import upsert_dataset_info

# 把项目根目录塞进 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 全局变量（用于本次运行去重）
used_uuids_global = set()

# 支持的 YAML 文件名
SUPPORTED_YAML_NAMES = {"local_dataset_info.yaml", "local_dataset_info.yml"}


def write_dataset_uuid_yaml(folder_path: Path, dataset_uuid: str, dry_run: bool = False) -> None:
    """
    在指定文件夹下生成/更新 dataset_uuid.yaml 文件（移除冗余mkdir）
    """
    uuid_yaml_path = folder_path / "dataset_uuid.yaml"
    uuid_data = {"dataset_uuid": dataset_uuid}
    
    if dry_run:
        logging.info(f"[dry-run] 将生成 dataset_uuid.yaml: {uuid_yaml_path} → {dataset_uuid}")
        return
    
    try:
        with uuid_yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                uuid_data, f, allow_unicode=True, default_flow_style=False, indent=2, sort_keys=False
            )
        logging.info(f"已生成 dataset_uuid.yaml: {uuid_yaml_path} → {dataset_uuid}")
    except Exception as e:
        logging.error(f"生成 dataset_uuid.yaml 失败 {uuid_yaml_path}: {e}")


# 新增：数据库会话工具函数
def get_max_dataset_name_id(db_path: str, dataset_name: str) -> int:
    """查询数据库中指定dataset_name的最大dataset_name_id，不存在则返回-1"""
    db = DatasetDatabase(db_path)
    with db.with_session() as session:
        result = session.query(func.max(DatasetDB.dataset_name_id)).filter(
            DatasetDB.dataset_name == dataset_name
        ).scalar()
        return result if result is not None else -1

def rename_folder_and_yaml(yaml_path: Path, old_folder_path: Path, new_folder_path: Path, dry_run: bool = False) -> bool:
    """
    仅重命名文件夹（不修改dataset_name），同步复制YAML文件到新路径（移除冗余exists检查）
    """
    try:
        new_yaml_path = new_folder_path / yaml_path.name
        if not dry_run:
            new_folder_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(yaml_path, new_yaml_path)
            logging.info(f"已复制YAML文件（保留原名称）: {yaml_path} → {new_yaml_path}")
        
        if not dry_run:
            if new_folder_path.exists():
                shutil.rmtree(new_folder_path)
            old_folder_path.rename(new_folder_path)
            logging.info(f"已重命名文件夹: {old_folder_path} → {new_folder_path}")
            
            # 移除旧文件夹（可选）
            if old_folder_path.exists() and old_folder_path != new_folder_path:
                shutil.rmtree(old_folder_path)
                logging.info(f"已删除旧文件夹: {old_folder_path}")
        else:
            logging.info(f"[dry-run] 将重命名文件夹: {old_folder_path} → {new_folder_path}")
        
        return True
    except Exception as e:
        logging.error(f"重命名失败 {old_folder_path}: {e}")
        if new_folder_path.exists() and not dry_run:
            shutil.rmtree(new_folder_path)
        return False


# ------------------------------------------------------------------
# 配置日志
# ------------------------------------------------------------------
def setup_logging(log_dir: Path, dry_run: bool = False) -> None:
    """设置日志输出到文件和控制台"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "import_dataset_info.log"

    if log_file.exists() and not dry_run:
        log_file.unlink()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 控制台 Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(ch)

    # 文件 Handler
    if not dry_run:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

    logging.info(f"日志已启动，日志文件：{log_file}")

# ------------------------------------------------------------------
# 使用 os.walk 查找 YAML 文件（跳过子目录）
# ------------------------------------------------------------------
def find_local_yaml_files(root: Path) -> list[Path]:
    found = []
    for current, dirnames, files in os.walk(root):
        if SUPPORTED_YAML_NAMES & set(files):
            filename = next(iter(SUPPORTED_YAML_NAMES & set(files)))
            found.append(Path(current) / filename)
            dirnames.clear()
    return found

# ------------------------------------------------------------------
# 数据清理函数
# ------------------------------------------------------------------
def clean_data_value(value: Any) -> Any:
    """清理数据值，确保数据库兼容"""
    if isinstance(value, list):
        return str(value[0]) if len(value) == 1 and value else str(value) if value else None
    if isinstance(value, dict):
        return str(value)
    return value

# ------------------------------------------------------------------
# 读取 + 检查 dataset_uuid + 重命名文件夹（不修改dataset_name）
# ------------------------------------------------------------------
def load_and_patch(yaml_path: Path, dry_run: bool = False, db_path: str = "./db/datasets.db") -> dict[str, Any]:
    try:
        with yaml_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logging.warning(f"解析失败 {yaml_path}: {e}")
        return {}

    if "dataset_name" not in data:
        logging.warning(f"跳过无效文件（缺少 dataset_name）：{yaml_path}")
        return {}

    # 保留原始dataset_name，仅生成序号
    original_dataset_name = data["dataset_name"].strip()
    max_id = get_max_dataset_name_id(db_path, original_dataset_name)
    new_name_id = max_id + 1
    data["dataset_name_id"] = new_name_id
    
    # 重命名文件夹
    old_folder_path = yaml_path.parent
    new_folder_path = old_folder_path.parent / f"{old_folder_path.name}_{new_name_id}"
    new_yaml_path = new_folder_path / yaml_path.name
    
    # 更新路径字段
    data["yaml_file_path"] = str(new_yaml_path.resolve())
    data["data_path"] = str(new_folder_path.resolve())
    
    # 调用重命名函数
    rename_success = rename_folder_and_yaml(yaml_path, old_folder_path, new_folder_path, dry_run)
    if not rename_success and not dry_run:
        logging.error(f"重命名失败，放弃处理该文件: {yaml_path}")
        return {}

    # UUID 处理逻辑
    existing_uuid = data.get("dataset_uuid")
    if not existing_uuid:
        uuid_yaml_path = yaml_path.parent / "dataset_uuid.yaml" if dry_run else new_folder_path / "dataset_uuid.yaml"
        try:
            if uuid_yaml_path.exists():
                with uuid_yaml_path.open(encoding="utf-8") as f:
                    uuid_data = yaml.safe_load(f)
                    external_uuid = uuid_data.get("uuid") or uuid_data.get("dataset_uuid")
                    if external_uuid:
                        logging.info(f"从 {uuid_yaml_path.name} 获取 UUID: {external_uuid}")
                        existing_uuid = external_uuid
        except Exception as e:
            logging.warning(f"读取 {uuid_yaml_path} 失败: {e}")

    if existing_uuid:
        data["dataset_uuid"] = existing_uuid
        used_uuids_global.add(existing_uuid)
        logging.info(f"[{original_dataset_name}] (ID: {new_name_id}) 使用已有 UUID: {existing_uuid}")
        
        # 补生成 dataset_uuid.yaml
        if not dry_run:
            write_dataset_uuid_yaml(new_folder_path, existing_uuid, dry_run)
    else:
        task_desc = data.get("task_instruction")
        device_model = data.get("device_model") or "unknown_device"

        try:
            registry_file = PROJECT_ROOT / "dataset_registry.yaml"
            new_uuid = get_or_create_uuid(
                task=task_desc, device=device_model, yaml_path=str(data["yaml_file_path"]),
                registry_file=str(registry_file), used_uuids=used_uuids_global,
            )
            data["dataset_uuid"] = new_uuid
            used_uuids_global.add(new_uuid)

            if not dry_run:
                with new_yaml_path.open("w", encoding="utf-8") as f:
                    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, indent=2, sort_keys=False)
                logging.info(f"已写入新 UUID: {new_yaml_path} → {new_uuid}")
                write_dataset_uuid_yaml(new_folder_path, new_uuid, dry_run)
            else:
                logging.info(f"[dry-run] 将生成 UUID: {new_yaml_path} → {new_uuid}")
                logging.info(f"[dry-run] 将生成 dataset_uuid.yaml: {new_folder_path}/dataset_uuid.yaml → {new_uuid}")

        except Exception as e:
            logging.error(f"自动生成 UUID 失败 {new_yaml_path}: {e}")
            return {}

    # 清理字段
    for key in ["device_model", "end_effector_type", "operation_platform_height"]:
        if key in data:
            data[key] = clean_data_value(data[key])

    logging.info(f"[{original_dataset_name}] (ID: {new_name_id}) -> {data['dataset_uuid']}")
    logging.info(f"数据集路径已设置为: {data['data_path']}")
    logging.info(f"YAML文件路径已更新为: {data['yaml_file_path']}")
    assert "dataset_name_id" in data, "dataset_name_id 字段缺失！"
    return data

# ---------------- 新增：提取 yaml 文件 ----------------
def collect_yaml_files(root_dirs: list[Path], output_dir: Path, dry_run: bool = False, skip_log_setup: bool = False) -> None:
    """收集所有 local_dataset_info.yml/.yaml 到 output_dir，并记录日志（支持跳过日志初始化）"""
    log_dir = output_dir / "logs"
    if not skip_log_setup:
        setup_logging(log_dir, dry_run=dry_run)

    if not dry_run:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        status = "将被清空" if output_dir.exists() else "将被创建"
        logging.info(f"[dry-run] 输出目录: {output_dir} ({status})")

    hub = {}
    idx = 0
    for r in root_dirs:
        for dirpath, dirnames, files in os.walk(r):
            if SUPPORTED_YAML_NAMES & set(files):
                filename = next(iter(SUPPORTED_YAML_NAMES & set(files)))
                src = Path(dirpath) / filename
                dst = output_dir / f"local_dataset_info_{idx}.yml"
                if not dry_run:
                    shutil.copy2(src, dst)
                hub[str(dst.resolve())] = str(src.resolve())
                logging.info(f"已复制: {src} → {dst}")
                idx += 1
                dirnames.clear()

    if not dry_run:
        hub_file = output_dir / "local_dataset_info_hub.yml"
        hub_file.write_text(yaml.dump(hub, sort_keys=True, allow_unicode=True, indent=2), encoding="utf-8")
        logging.info(f"已生成 hub 文件: {hub_file}")

    logging.info(f"已收集 {idx} 个文件到 {output_dir}")

# ------------------------------------------------------------------
# 批量处理函数
# ------------------------------------------------------------------
def process_files_batch(
    yaml_files: list[Path], max_workers: int = 8, dry_run: bool = False, db_path: str = "./db/datasets.db"
) -> tuple[list[dict[str, Any]], int]:
    datasets = []
    invalid_count = 0
    seen_dataset_combo = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(load_and_patch, yml, dry_run, db_path): yml for yml in yaml_files}

        for future in concurrent.futures.as_completed(future_to_file):
            yml = future_to_file[future]
            try:
                data = future.result()
                if not data:
                    invalid_count += 1
                    continue

                # 组合键去重
                combo_key = (
                    data.get("dataset_name"),
                    data.get("device_model", "unknown_device"),
                    data.get("dataset_name_id", 0)
                )
                if combo_key in seen_dataset_combo:
                    logging.warning(f"跳过重复的数据集组合: {combo_key[0]} (ID: {combo_key[2]}, device_model: {combo_key[1]}) - 文件: {yml}")
                    invalid_count += 1
                    continue

                seen_dataset_combo.add(combo_key)
                datasets.append(data)

            except Exception as e:
                logging.error(f"处理文件失败 {yml}: {e}")
                invalid_count += 1

    return datasets, invalid_count

# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def main() -> None:
    """主函数：解析参数 → 扫描文件 → 处理数据 → 导入数据库"""
    global used_uuids_global
    used_uuids_global = set()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="递归查找 local_dataset_info.yaml/.yml，检查 dataset_uuid 并批量入库")
    parser.add_argument("scan_root", type=str, help="要扫描的根目录（支持多路径分隔）")
    parser.add_argument("--db-path", type=str, default="./db/datasets.db", help="数据库文件路径")
    parser.add_argument("--workers", type=int, default=8, help="并行工作线程数")
    parser.add_argument("--collect-only", action="store_true", help="仅收集 yaml 文件，不做数据库导入")
    parser.add_argument("--collect-output", type=Path, default=Path("./collected_yamls"), help="收集模式输出目录")
    parser.add_argument("--dry-run", action="store_true", help="只扫描和模拟，不写入文件或数据库")
    args = parser.parse_args()

    # 解析多路径
    root_paths = [Path(p.strip()) for p in re.split(r"[\s;,:\n]+", args.scan_root.strip()) if p.strip()]
    
    # 统一初始化日志
    log_dir = args.collect_output / "logs"
    setup_logging(log_dir, dry_run=args.dry_run)

    if args.collect_only:
        logging.info(f"收集模式：从 {len(root_paths)} 个根目录收集 YAML 文件")
        collect_yaml_files(root_paths, args.collect_output, dry_run=args.dry_run, skip_log_setup=True)
        return

    # 扫描所有YAML文件
    all_yaml_files = []
    for root in root_paths:
        if not root.exists():
            logging.error(f"路径不存在：{root}")
            continue
        logging.info(f"扫描目录: {root}")
        all_yaml_files.extend(find_local_yaml_files(root))

    if not all_yaml_files:
        logging.warning("未找到任何 local_dataset_info.yaml/.yml，退出。")
        sys.exit(0)

    logging.info(f"找到 {len(all_yaml_files)} 个 YAML 文件")

    # 并行处理
    datasets, invalid_count = process_files_batch(all_yaml_files, args.workers, args.dry_run, args.db_path)

    if invalid_count > 0:
        logging.warning(f"跳过 {invalid_count} 个无效文件")
    if not datasets:
        logging.warning("没有有效数据集，退出。")
        sys.exit(0)

    logging.info(f"准备处理 {len(datasets)} 个数据集")
    if args.dry_run:
        logging.info("[dry-run] 模拟结束，未写入数据库或文件。")
        return

    # 写入数据库（复用session，避免重复创建连接）
    try:
        db_path = Path(args.db_path).expanduser().absolute()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = DatasetDatabase(db_path)
        logging.info(f"数据库连接已创建: {db_path}")

        with db.with_session() as session:
            for record in datasets:
                # 传入已有session，提升性能
                upsert_dataset_info(yaml_data=record, session=session)
            session.commit()
        
        success_count = len(datasets)
        logging.info(f"成功导入 {success_count} 个数据集到数据库")

        # 执行 separate.py
        logging.info("正在执行 separate.py 脚本...")
        separate_script = PROJECT_ROOT / "scripts" / "format_converters" / "separate.py"
        if not separate_script.exists():
            logging.critical(f"无法找到 separate.py: {separate_script}")
            sys.exit(1)

        cmd = [sys.executable, str(separate_script), *[str(p) for p in root_paths], "--log-dir", str(log_dir / "separate_logs")]
        if args.dry_run:
            cmd.append("--dry-run")

        result = subprocess.run(
            cmd, check=True, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8"
        )
        logging.info("separate.py 脚本执行成功。")
        for line in result.stdout.strip().splitlines():
            if any(kw in line for kw in ["DRY RUN", "处理完成", "统计"]):
                logging.info(f"[separate] {line}")

    except subprocess.CalledProcessError as e:
        logging.critical(f"separate.py 脚本执行失败: {e}")
        if e.stdout:
            logging.critical(f"标准输出:\n{e.stdout}")
        if e.stderr:
            logging.critical(f"错误输出:\n{e.stderr}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"批量导入失败: {e}")
        logging.critical(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()