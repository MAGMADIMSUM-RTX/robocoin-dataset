#!/usr/bin/env python
"""
LeRobot v2.1 -> v3.0 数据集转换 - 纯函数调用版
直接调用 convert_v21_to_v30() 即可使用
"""

import sys
import logging
from pathlib import Path

# 导入官方转换函数
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    convert_dataset as convert_dataset_func,
    init_logging
)

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
    """
    把 LeRobot 数据集从 v2.1 转换成 v3.0
    支持：单个数据集 / 批量文件夹转换
    不会覆盖原始数据，安全转换
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    # 判断是单个数据集还是批量目录
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
    print(f"正在处理: {root.name}")
    
    # 输出路径（不指定则自动生成 xxx_v3）
    if output:
        output_path = Path(output)
    else:
        output_path = root.parent / f"{root.name}_v3"
    
    # 本地转换禁用推送
    if push_to_hub:
        print("本地转换已自动关闭 push_to_hub")
        push_to_hub = False

    try:
        # 调用官方转换
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
        
        # 安全移动文件 + 恢复原始数据
        import shutil
        old_root_backup = root.parent / f"{root.name}_old"
        
        # 移动转换结果到目标目录
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            shutil.rmtree(output_path)
            
        shutil.move(str(root), str(output_path))
        print(f"转换完成 → {output_path}")
        
        # 恢复原始数据集
        if old_root_backup.exists():
            shutil.move(str(old_root_backup), str(root))
            print(f"原始数据已恢复 → {root}")

    except Exception as e:
        print(f"转换失败 {root.name}: {str(e)}")


def _convert_batch_datasets(root_dir, output_dir, **kwargs):
    # 批量扫描所有数据集
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
            repo_id=subdir.name,
            **kwargs
        )


# ====================== 你只需要改这里 ======================
if __name__ == "__main__":
    init_logging()
    
    # 1. 转换单个数据集
    convert_v21_to_v30(
        input_path="/home/liuyou/Documents/robocoin-dataset/你的数据集文件夹",
        output_path="/home/liuyou/Documents/robocoin-dataset/转换后_v3",
        force_conversion=True
    )

    # 2. 批量转换整个目录（二选一即可）
    # convert_v21_to_v30(
    #     input_path="/home/liuyou/Documents/robocoin-dataset/所有数据集目录",
    #     output_path="/home/liuyou/Documents/robocoin-dataset/批量输出目录",
    #     force_conversion=True
    # )
