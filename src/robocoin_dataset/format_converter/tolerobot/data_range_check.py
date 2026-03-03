import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ==========================================
# 🌟 细粒度物理绝对极限定义 (Hard Bounds)
# ==========================================
# 通过特征名称的关键字（后缀）来动态匹配物理限制
FINE_GRAINED_BOUNDS = {
    "_rad": {"min": -3.14, "max": 3.14},          # 关节角度：通常在 -pi 到 pi 之间 (加一点容差)
    "_m": {"min": -2.0, "max": 2.0},              # 空间坐标：米，假设机器人在两米范围内活动
    "gripper_open_scale": {"min": 0.0, "max": 1.0}, # 归一化夹爪：必须严格在 0 到 1 之间
    "gripper_open": {"min": 0.0, "max": 100.0},     # 未归一化的原始夹爪：根据实际硬件调整 (假设最大100)
}

# 针对没有细粒度 names 的数组特征进行通用兜底防御（防飞点）
GLOBAL_FALLBACK_BOUNDS = {"min": -10.0, "max": 10.0}


class LerobotDatasetValidator:
    """
    LeRobot 数据集后置校验器
    用于在转换完成后，基于 info.json 和 细粒度物理约束，
    交叉验证 Parquet 数据文件的完整性与一致性。
    """
    def __init__(self, dataset_path: str):
        """
        初始化数据集校验器
        
        Args:
            dataset_path: 转换后数据集的根路径（包含meta/和data/目录）
        """
        self.dataset_path = Path(dataset_path).expanduser().absolute()
        self.info_path = self.dataset_path / "meta" / "info.json"
        self.data_dir = self.dataset_path / "data"
        
        # 兼容不同的LeRobot版本路径
        if not self.info_path.exists():
            self.info_path = self.dataset_path / "info.json"
            
        self._validate_paths()
        
        # 加载元数据法典
        self.info_data = self._load_info_json()
        self.expected_features = self.info_data.get('features', {})
        self.expected_stats = self.info_data.get('stats', {})
        
        # 提取特征名映射字典 (Feature -> List of sub-names)
        self.feature_names_map = self._extract_feature_names()
        

    def _validate_paths(self):
        """验证必要文件/目录是否存在"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
        if not self.info_path.exists():
            raise FileNotFoundError(f"找不到 info.json 文件: {self.info_path}")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

    def _load_info_json(self) -> dict:
        """加载info.json"""
        with open(self.info_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_feature_names(self) -> Dict[str, List[str]]:
        """提取特征中的 names 列表，用于细粒度校验"""
        names_map = {}
        for feat_name, feat_info in self.expected_features.items():
            if "names" in feat_info and isinstance(feat_info["names"], list):
                names_map[feat_name] = feat_info["names"]
                logger.debug(f"📐 提取到 '{feat_name}' 的子维度列表，共 {len(feat_info['names'])} 维")
        return names_map

    def _validate_single_parquet(self, file_path: Path) -> List[str]:
        """
        深度校验单个 parquet 文件
        返回: 错误信息列表 (如果为空说明校验完美通过)
        """
        errors = []
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            return [f"无法读取 Parquet 文件: {e}"]

        for feature_name, feature_info in self.expected_features.items():
            # 1. 检查列是否存在
            if not (feature_name.startswith("action") or feature_name.startswith("observation.state")):
                continue
            if feature_name not in df.columns:
                errors.append(f"缺失特征列: '{feature_name}'")
                continue

            expected_shape = tuple(feature_info.get('shape', []))
            dtype = feature_info.get('dtype')
            
            # 如果是视频或图像元数据索引，直接跳过数值范围和维度的硬校验
            if not expected_shape or dtype in ['video', 'image']:
                continue

            col_data = df[feature_name]

            # 将这列数据堆叠成高效的 Numpy 矩阵: shape = (N_frames, Dim)
            try:
                np_data = np.vstack(col_data.values)
            except ValueError:
                errors.append(f"特征 '{feature_name}' 数据格式错乱，无法解析为连续数组")
                continue

            # 2. 检查维度 (严格对齐 info.json)
            actual_shape = np_data.shape[1:]  # 去掉第一维(帧数)
            if actual_shape != expected_shape:
                errors.append(f"特征 '{feature_name}' 维度错误: info.json 预期 {expected_shape}，但实际读取为 {actual_shape}")
                continue # 维度错乱后续切片会崩溃，直接跳过该特征

            # 3. 检查是否有致命脏数据 (NaN 或 Inf)
            if np_data.dtype.kind in 'fc':  # 针对浮点型数据
                if np.isnan(np_data).any():
                    errors.append(f"☠️ 致命错误: 特征 '{feature_name}' 中检测到 NaN (缺失值)！")
                if np.isinf(np_data).any():
                    errors.append(f"☠️ 致命错误: 特征 '{feature_name}' 中检测到 Inf (无限大值)！")

            # ==========================================
            # 🌟 4. 细粒度物理绝对极限检查 (Hard Bounds)
            # ==========================================
            if feature_name in self.feature_names_map:
                sub_names = self.feature_names_map[feature_name]
                for col_idx, sub_name in enumerate(sub_names):
                    # 精准切出这一子维度的数据 (例如仅提取 left_arm_joint_1_rad)
                    col_data_np = np_data[:, col_idx]
                    
                    # 动态匹配极限规则
                    matched_bounds = None
                    for keyword, bounds in FINE_GRAINED_BOUNDS.items():
                        if keyword in sub_name:
                            matched_bounds = bounds
                            break
                    
                    if matched_bounds:
                        hard_min = matched_bounds["min"]
                        hard_max = matched_bounds["max"]
                        
                        # 找出低于下限的异常帧索引
                        under_limit_indices = np.where(col_data_np < hard_min - 1e-4)[0]
                        if len(under_limit_indices) > 0:
                            # 提取最严重的一个极值
                            extreme_min_val = np.min(col_data_np[under_limit_indices])
                            # 取前 3 个帧号作为示例展示
                            sample_frames = under_limit_indices[:3].tolist()
                            errors.append(
                                f"🔴 物理极限报错: '{feature_name}.{sub_name}' 出现非法小值 (极小值 {extreme_min_val:.4f}，下限 {hard_min})。共 {len(under_limit_indices)} 帧异常，典型帧号: {sample_frames}"
                            )

                        # 找出高于上限的异常帧索引
                        over_limit_indices = np.where(col_data_np > hard_max + 1e-4)[0]
                        if len(over_limit_indices) > 0:
                            extreme_max_val = np.max(col_data_np[over_limit_indices])
                            sample_frames = over_limit_indices[:3].tolist()
                            errors.append(
                                f"🔴 物理极限报错: '{feature_name}.{sub_name}' 出现非法大值 (极大值 {extreme_max_val:.4f}，上限 {hard_max})。共 {len(over_limit_indices)} 帧异常，典型帧号: {sample_frames}"
                            )
            else:
                # 针对没有 names 的连续数组，使用全局兜底校验防止严重飞点
                actual_min_val = np.min(np_data)
                actual_max_val = np.max(np_data)
                if actual_min_val < GLOBAL_FALLBACK_BOUNDS["min"] or actual_max_val > GLOBAL_FALLBACK_BOUNDS["max"]:
                    errors.append(f"🟠 全局飞点警告: '{feature_name}' 整体范围异常 [{actual_min_val:.2f}, {actual_max_val:.2f}]")

            # 5. 越界检查 (与 info.json 里的动态 stats 极值核对 - Soft Bounds)
            feature_stats = self.expected_stats.get(feature_name)
            if feature_stats and 'min' in feature_stats and 'max' in feature_stats:
                exp_min = np.array(feature_stats['min'])
                exp_max = np.array(feature_stats['max'])
                
                actual_min_axis0 = np.min(np_data, axis=0)
                actual_max_axis0 = np.max(np_data, axis=0)

                # 引入微小容差，防止浮点数精度漂移
                tol = 1e-5
                if np.any(actual_min_axis0 < exp_min - tol):
                    errors.append(f"🟡 数据漂移警告: 特征 '{feature_name}' 存在低于 info.json `min` 限值的异常数据")
                if np.any(actual_max_axis0 > exp_max + tol):
                    errors.append(f"🟡 数据漂移警告: 特征 '{feature_name}' 存在高于 info.json `max` 限值的异常数据")

        return errors

    def run(self) -> tuple[bool, str]:
        """
        执行全量盘查流程
        🌟 修改返回值: 返回 (是否通过校验, 错误信息汇总)
        """
        logger.info(f"🔍 开始基于 info.json 校验数据集: {self.dataset_path.name}")
        
        parquet_files = list(self.data_dir.rglob("*.parquet"))
        if not parquet_files:
            return False, "找不到任何 Parquet 数据文件"

        total_errors = 0
        all_error_messages = [] # 用来收集所有报错文本

        for file in parquet_files:
            errors = self._validate_single_parquet(file)
            if errors:
                total_errors += len(errors)
                # 把文件名和对应的错误拼起来
                file_err_str = f"[{file.name}] " + " | ".join(errors)
                all_error_messages.append(file_err_str)
                
                for err in errors:
                    logger.warning(f"  ⚠️ [{file.name}] {err}")

        if total_errors == 0:
            logger.info("✅ 数据集校验完美通过！")
            return True, ""
        else:
            logger.error(f"❌ 校验结束，共拦截到 {total_errors} 处异常。")
            # 将所有错误拼接成一个长字符串返回（如果太长可以做截断保护数据库）
            final_err_msg = "\n".join(all_error_messages)
            return False, f"物理极值校验失败，共 {total_errors} 处异常: {final_err_msg}"

# ==================== 使用示例 ====================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # 替换为你刚才生成的数据集绝对/相对路径
    DATASET_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/user/robocoin-dataset/outputs/converted_datasets/Agilex_Split_Aloha_erase _blackboard_0"
    
    try:
        validator = LerobotDatasetValidator(DATASET_PATH)
        validator.run()
    except Exception as e:
        logger.error(f"校验程序异常中断: {e}")