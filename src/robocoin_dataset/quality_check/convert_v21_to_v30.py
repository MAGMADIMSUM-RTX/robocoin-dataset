#!/usr/bin/env python
"""
LeRobot v2.1 -> v3.0 数据集转换
自动保留额外文件夹 + meta额外文件
重名时提醒，不覆盖、不替换
"""

import sys
import logging
from pathlib import Path
import shutil

# 导入官方转换函数
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    convert_dataset as convert_dataset_func,
    init_logging
)

def copy_extra_folders(src_root: Path, dst_root: Path):
    """复制不是 data/meta/videos 的额外文件夹，重名则提醒不覆盖"""
    standard = {"data", "meta", "videos"}
    for f in src_root.iterdir():
        if f.is_dir() and f.name not in standard:
            dst = dst_root / f.name
            if dst.exists():
                print(f"⚠️  目标已存在，跳过不覆盖文件夹: {dst}")
                continue
            shutil.copytree(f, dst)
            print(f"📂 已复制额外文件夹: {f.name}")

def copy_extra_meta_files(src_meta: Path, dst_meta: Path):
    """复制meta中非标准文件，重名则提醒不覆盖"""
    standard_meta_files = {
        "info.json",
        "episodes.jsonl",
        "tasks.jsonl",
        "episodes_stats.jsonl"
    }
    for f in src_meta.iterdir():
        if f.is_file() and f.name not in standard_meta_files:
            dst_file = dst_meta / f.name
            if dst_file.exists():
                print(f"⚠️  目标已存在，跳过不覆盖文件: {dst_file}")
                continue
            shutil.copy2(f, dst_file)
            print(f"📄 已复制meta额外文件: {f.name}")

def convert_v21_to_v30(
    input_path: str | Path,
    output_path: str | Path | None = None,
    repo_id: str | None = None,
    push_to_hub: bool = False,
    branch: str | None = None,
    force_conversion: bool = False,
    data_file_size_in_mb: int | None = None,
    video_file_size_in_mb: int | None = None,
):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    is_single_dataset = (input_path / "meta/info.json").exists()

    if is_single_dataset:
        _convert_single_dataset(
            root=input_path,
            output=output_path,
            repo_id=repo_id or input_path.name,
            push_to_hub=push_to_hub,
            branch=branch,
            force_conversion=force_conversion,
            data_file_size_in_mb=data_file_size_in_mb,
            video_file_size_in_mb=video_file_size_in_mb
        )
    else:
        _convert_batch_datasets(
            root_dir=input_path,
            output_dir=output_path,
            push_to_hub=push_to_hub,
            branch=branch,
            force_conversion=force_conversion,
            data_file_size_in_mb=data_file_size_in_mb,
            video_file_size_in_mb=video_file_size_in_mb
        )

def _convert_single_dataset(root, output, repo_id, push_to_hub, **kwargs):
    print(f"\n正在处理: {root.name}")

    if output:
        output_path = Path(output)
    else:
        output_path = root.parent / f"{root.name}_v3"

    # 输出目录已存在，直接提醒并退出，不覆盖
    if output_path.exists():
        print(f"⚠️  输出目录已存在，不执行转换与覆盖: {output_path}")
        return

    if push_to_hub:
        print("ℹ️  本地转换已自动关闭 push_to_hub")
        push_to_hub = False

    try:
        official_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['branch', 'data_file_size_in_mb', 'video_file_size_in_mb', 'force_conversion']
        }

        convert_dataset_func(
            repo_id=repo_id,
            root=root,
            push_to_hub=push_to_hub,
            **official_kwargs
        )

        old_root_backup = root.parent / f"{root.name}_old"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), str(output_path))
        print(f"✅ 转换完成 → {output_path}")

        # 复制额外文件夹与meta文件（重名不覆盖）
        copy_extra_folders(old_root_backup, output_path)
        copy_extra_meta_files(old_root_backup / "meta", output_path / "meta")

        # 恢复原始数据
        if old_root_backup.exists():
            shutil.move(str(old_root_backup), str(root))
            print(f"✅ 原始数据已恢复 → {root}")

        print("✅ 所有额外内容已复制完成（重名已跳过）\n")

    except Exception as e:
        print(f"❌ 转换失败 {root.name}: {str(e)}")
        old_root_backup = root.parent / f"{root.name}_old"
        if old_root_backup.exists():
            shutil.move(str(old_root_backup), str(root))
            print(f"✅ 原始数据已恢复 → {root}")

def _convert_batch_datasets(root_dir, output_dir, **kwargs):
    subdirs = [d for d in root_dir.iterdir() if d.is_dir() and (d / "meta/info.json").exists()]

    if not subdirs:
        print("未找到任何合法数据集")
        return

    print(f"找到 {len(subdirs)} 个数据集，开始批量转换")
    for subdir in subdirs:
        subdir_output = Path(output_dir) / f"{subdir.name}_v3" if output_dir else None
        _convert_single_dataset(
            root=subdir,
            output=subdir_output,
            repo_id=subdir.name,** kwargs
        )

# ====================== 测试加载v3.0数据集 ======================
def test_load_dataset(output_path):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(repo_id=Path(output_path).name, root=output_path)
        print("="*50)
        print("✅ 数据集加载成功")
        print(f"总帧数: {len(dataset)}")
        print(f"相机 keys: {dataset.meta.camera_keys}")
        print("="*50)
        return dataset
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

if __name__ == "__main__":

    INPUT = "/home/liuyou/Documents/data/Agilex_Cobot_Magic_pour_water_into_cup_0_qced_hardlink"
    OUTPUT = "/home/liuyou/Documents/lerobotv3.0/Agilex_Cobot_Magic_pour_water_into_cup_0_qced_hardlink_v3"

    convert_v21_to_v30(
        input_path=INPUT,
        output_path=OUTPUT,
        force_conversion=True
    )

    test_load_dataset(OUTPUT)
