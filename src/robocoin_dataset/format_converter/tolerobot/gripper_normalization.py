import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class GripperOpenNormalizer:
    def __init__(self, dataset_path: str):
        """
        初始化夹爪数据归一化器
        
        Args:
            dataset_path: 转换后数据集的根路径（包含meta/和data/目录）
        """
        self.dataset_path = Path(dataset_path).expanduser().absolute()
        self.info_path = self.dataset_path / "meta" / "info.json"
        self.stats_path = self.dataset_path / "meta" / "episodes_stats.jsonl"
        self.data_dir = self.dataset_path / "data"
        
        # 验证路径
        self._validate_paths()
        
        # 加载info.json
        self.info_data = self._load_info_json()
        
        # 定位gripper_open字段的索引
        self.gripper_indices = self._find_gripper_indices()
        logger.info(f"找到gripper_open字段索引: {self.gripper_indices}")
        
        # 存储归一化参数
        self.gripper_stats = {}
        # 检测有效夹爪类型（单臂/双臂）
        self.valid_gripper_types = self._get_valid_gripper_types()
        logger.info(f"检测到有效夹爪类型: {self.valid_gripper_types}")
        # 量程信息
        self.gripper_ranges = {}  # 存储每个夹爪的原始量程

    def _validate_paths(self):
        """验证必要文件/目录是否存在"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
        if not self.info_path.exists():
            raise FileNotFoundError(f"info.json不存在: {self.info_path}")
        if not self.stats_path.exists():
            raise FileNotFoundError(f"episodes_stats.jsonl不存在: {self.stats_path}")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

    def _load_info_json(self) -> dict:
        """加载info.json"""
        with open(self.info_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _find_gripper_indices(self) -> Dict[str, Dict[str, Optional[int]]]:
        """
        查找gripper_open字段在observation.state和action中的索引
        支持单臂场景（不存在的夹爪返回None）
        返回格式: 
        {
            "observation.state": {"left_gripper_open": 6, "right_gripper_open": None},
            "action": {"left_gripper_open": 6, "right_gripper_open": None}
        }
        """
        indices = {}
        
        # 检查observation.state
        state_names = self.info_data['features']['observation.state']['names']
        obs_indices = {}
        try:
            obs_indices['left_gripper_open'] = state_names.index('left_gripper_open')
        except ValueError:
            obs_indices['left_gripper_open'] = None
        try:
            obs_indices['right_gripper_open'] = state_names.index('right_gripper_open')
        except ValueError:
            obs_indices['right_gripper_open'] = None
        indices['observation.state'] = obs_indices
        
        # 检查action
        action_names = self.info_data['features']['action']['names']
        act_indices = {}
        try:
            act_indices['left_gripper_open'] = action_names.index('left_gripper_open')
        except ValueError:
            act_indices['left_gripper_open'] = None
        try:
            act_indices['right_gripper_open'] = action_names.index('right_gripper_open')
        except ValueError:
            act_indices['right_gripper_open'] = None
        indices['action'] = act_indices
        
        return indices

    def _get_valid_gripper_types(self) -> List[str]:
        """检测有效的夹爪类型（存在索引的）"""
        valid_types = []
        # 检查observation.state中的索引
        for gripper_type in ['left_gripper_open', 'right_gripper_open']:
            if (self.gripper_indices['observation.state'][gripper_type] is not None or
                self.gripper_indices['action'][gripper_type] is not None):
                valid_types.append(gripper_type)
        return valid_types

    def _detect_gripper_range(self, min_val: float, max_val: float) -> int:
        """
        根据数值范围判断夹爪原始量程
        Args:
            min_val: 最小值
            max_val: 最大值
        Returns:
            量程（1/10/100/1000）
        """
        # 定义量程区间阈值（允许少量超出）
        ranges = [
            (1.1, 1),    # 0-1 量程
            (10.5, 10),  # 0-10 量程
            (105.0, 100),# 0-100 量程
            (1050.0, 1000)# 0-1000 量程
        ]
        
        max_abs = max(abs(min_val), abs(max_val))
        for threshold, range_val in ranges:
            if max_abs <= threshold:
                return range_val
        
        # 默认返回1（0-1量程）
        logger.warning(f"无法识别量程范围 (min={min_val}, max={max_val})，默认使用0-1量程")
        return 1

    def collect_gripper_stats(self) -> Dict[str, Dict[str, float]]:
        """
        收集所有episode的gripper_open全局min/max，并检测量程
        返回格式:
        {
            "left_gripper_open": {"min": 0.05, "max": 0.09, "range": 1},
            "right_gripper_open": {"min": 0.03, "max": 0.09, "range": 1}
        }
        """
        # 初始化统计值
        stats = {}
        for gripper_type in self.valid_gripper_types:
            stats[gripper_type] = {
                'min': float('inf'),
                'max': -float('inf')
            }
        
        # 先从episodes_stats.jsonl获取初始统计
        with open(self.stats_path, 'r', encoding='utf-8') as f:
            for line in f:
                episode_stats = json.loads(line)
                episode_stats_data = episode_stats['stats']
                
                # 处理observation.state
                state_stats = episode_stats_data.get('observation.state', {})
                if state_stats:
                    for gripper_type in self.valid_gripper_types:
                        idx = self.gripper_indices['observation.state'][gripper_type]
                        if idx is not None and idx < len(state_stats.get('min', [])):
                            min_val = state_stats['min'][idx]
                            max_val = state_stats['max'][idx]
                            stats[gripper_type]['min'] = min(stats[gripper_type]['min'], min_val)
                            stats[gripper_type]['max'] = max(stats[gripper_type]['max'], max_val)
                
                # 处理action
                action_stats = episode_stats_data.get('action', {})
                if action_stats:
                    for gripper_type in self.valid_gripper_types:
                        idx = self.gripper_indices['action'][gripper_type]
                        if idx is not None and idx < len(action_stats.get('min', [])):
                            min_val = action_stats['min'][idx]
                            max_val = action_stats['max'][idx]
                            stats[gripper_type]['min'] = min(stats[gripper_type]['min'], min_val)
                            stats[gripper_type]['max'] = max(stats[gripper_type]['max'], max_val)
        
        # 验证统计值，异常则从原始数据重新计算
        need_recollect = False
        for gripper_type in self.valid_gripper_types:
            if stats[gripper_type]['min'] >= stats[gripper_type]['max']:
                need_recollect = True
                break
        
        if need_recollect:
            logger.warning("从stats文件获取的范围异常，将从原始数据重新计算")
            stats = self._collect_stats_from_parquet()
        
        # 检测每个夹爪的量程
        for gripper_type in self.valid_gripper_types:
            self.gripper_ranges[gripper_type] = self._detect_gripper_range(
                stats[gripper_type]['min'],
                stats[gripper_type]['max']
            )
        
        self.gripper_stats = stats
        logger.info(f"收集到gripper_open统计: {self.gripper_stats}")
        logger.info(f"检测到夹爪量程: {self.gripper_ranges}")
        
        return self.gripper_stats

    def _collect_stats_from_parquet(self) -> Dict[str, Dict[str, float]]:
        """从parquet文件重新计算gripper_open的全局min/max"""
        stats = {}
        values = {}
        for gripper_type in self.valid_gripper_types:
            stats[gripper_type] = {'min': float('inf'), 'max': -float('inf')}
            values[gripper_type] = []
        
        # 遍历所有parquet文件
        parquet_files = list(self.data_dir.rglob("*.parquet"))
        logger.info(f"找到 {len(parquet_files)} 个parquet文件，开始读取...")
        
        for file in parquet_files:
            df = pd.read_parquet(file)
            
            # 处理observation.state
            if 'observation.state' in df.columns:
                state_data = np.vstack(df['observation.state'].values)
                for gripper_type in self.valid_gripper_types:
                    idx = self.gripper_indices['observation.state'][gripper_type]
                    if idx is not None and idx < state_data.shape[1]:
                        values[gripper_type].extend(state_data[:, idx])
            
            # 处理action
            if 'action' in df.columns:
                action_data = np.vstack(df['action'].values)
                for gripper_type in self.valid_gripper_types:
                    idx = self.gripper_indices['action'][gripper_type]
                    if idx is not None and idx < action_data.shape[1]:
                        values[gripper_type].extend(action_data[:, idx])
        
        # 计算统计
        for gripper_type in self.valid_gripper_types:
            if values[gripper_type]:
                stats[gripper_type]['min'] = np.min(values[gripper_type])
                stats[gripper_type]['max'] = np.max(values[gripper_type])
        
        return stats

    def normalize_gripper_value(self, value: float, gripper_type: str) -> float:
        """
        根据检测到的量程归一化单个夹爪值到[0,1]范围
        
        Args:
            value: 原始值
            gripper_type: 'left_gripper_open' 或 'right_gripper_open'
        
        Returns:
            归一化后的值
        """
        if gripper_type not in self.gripper_stats:
            raise ValueError(f"不支持的夹爪类型: {gripper_type}")
        
        # 获取量程
        gripper_range = self.gripper_ranges.get(gripper_type, 1)
        
        # 按量程归一化（0-量程 -> 0-1）
        normalized = value / gripper_range
        # 限制范围在[0,1]
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return float(normalized)

    def process_parquet_files(self) -> bool:
        """
        处理所有parquet文件：
        1. 提取gripper_open字段
        2. 根据检测到的量程归一化得到gripper_open_scale
        3. 保存新字段到parquet文件，支持单臂/双臂场景
        
        Returns:
            是否成功处理（有文件被修改）
        """
        if not self.gripper_stats:
            self.collect_gripper_stats()
        
        # 遍历所有parquet文件
        parquet_files = list(self.data_dir.rglob("*.parquet"))
        logger.info(f"开始处理 {len(parquet_files)} 个parquet文件...")
        
        has_processed = False
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                modified = False
                
                # 处理observation.state -> 添加gripper_open_scale_state
                if 'observation.state' in df.columns:
                    state_data = np.vstack(df['observation.state'].values)
                    normalized_values = []
                    
                    # 为每个有效夹爪归一化
                    for gripper_type in self.valid_gripper_types:
                        idx = self.gripper_indices['observation.state'][gripper_type]
                        if idx is not None and idx < state_data.shape[1]:
                            gripper_vals = state_data[:, idx]
                            norm_vals = [self.normalize_gripper_value(v, gripper_type) for v in gripper_vals]
                            normalized_values.append(norm_vals)
                    
                    # 组合成新字段
                    if normalized_values:
                        gripper_scale = np.column_stack(normalized_values)
                        # 转换为列表形式保存
                        if gripper_scale.shape[1] == 1:
                            df['gripper_open_scale_state'] = gripper_scale.flatten().tolist()
                        else:
                            df['gripper_open_scale_state'] = list(gripper_scale)
                        modified = True
                
                # 处理action -> 添加gripper_open_scale_action
                if 'action' in df.columns:
                    action_data = np.vstack(df['action'].values)
                    normalized_values = []
                    
                    # 为每个有效夹爪归一化
                    for gripper_type in self.valid_gripper_types:
                        idx = self.gripper_indices['action'][gripper_type]
                        if idx is not None and idx < action_data.shape[1]:
                            gripper_vals = action_data[:, idx]
                            norm_vals = [self.normalize_gripper_value(v, gripper_type) for v in gripper_vals]
                            normalized_values.append(norm_vals)
                    
                    # 组合成新字段
                    if normalized_values:
                        gripper_scale = np.column_stack(normalized_values)
                        # 转换为列表形式保存
                        if gripper_scale.shape[1] == 1:
                            df['gripper_open_scale_action'] = gripper_scale.flatten().tolist()
                        else:
                            df['gripper_open_scale_action'] = list(gripper_scale)
                        modified = True
                
                # 保存修改后的文件
                if modified:
                    df.to_parquet(file)
                    logger.info(f"已处理文件: {file}")
                    has_processed = True
                else:
                    logger.info(f"文件无需要处理的字段: {file}")
                    
            except Exception as e:
                logger.error(f"处理文件 {file} 失败: {e}")
                continue
        
        return has_processed

    def update_info_json(self) -> bool:
        """
        更新info.json，根据有效夹爪数量动态添加字段描述
        
        Returns:
            是否成功更新
        """
        try:
            # 移除旧的字段（如果存在）
            for field in ['gripper_open_scale_state', 'gripper_open_scale_action']:
                if field in self.info_data['features']:
                    del self.info_data['features'][field]
            
            # 生成字段名称列表
            scale_names = [f"{gt.replace('_open', '_open_scale')}" for gt in self.valid_gripper_types]
            # 确定shape
            field_shape = [len(self.valid_gripper_types)] if len(self.valid_gripper_types) > 1 else [1]
            
            # 添加gripper_open_scale_state
            self.info_data['features']['gripper_open_scale_state'] = {
                "names": scale_names,
                "dtype": "float32",
                "shape": field_shape
            }
            
            # 添加gripper_open_scale_action
            self.info_data['features']['gripper_open_scale_action'] = {
                "names": scale_names,
                "dtype": "float32",
                "shape": field_shape
            }
            
            # 保存修改后的info.json
            with open(self.info_path, 'w', encoding='utf-8') as f:
                json.dump(self.info_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"已更新info.json: {self.info_path}")
            logger.info(f"新字段配置 - names: {scale_names}, shape: {field_shape}")
            return True
        except Exception as e:
            logger.error(f"更新info.json失败: {e}")
            return False

    def update_episodes_stats(self) -> bool:
        """
        更新episodes_stats.jsonl，添加gripper_open_scale的统计信息，支持单臂/双臂
        
        Returns:
            是否成功更新
        """
        try:
            new_stats_lines = []
            
            # 读取原始stats文件
            with open(self.stats_path, 'r', encoding='utf-8') as f:
                for line in f:
                    episode_stats = json.loads(line)
                    episode_idx = episode_stats['episode_index']
                    
                    # 找到对应的parquet文件
                    parquet_file = None
                    for file in self.data_dir.rglob(f"episode_{episode_idx:06d}.parquet"):
                        parquet_file = file
                        break
                    
                    if not parquet_file:
                        logger.warning(f"找不到episode {episode_idx} 的parquet文件")
                        new_stats_lines.append(json.dumps(episode_stats))
                        continue
                    
                    # 读取parquet文件并计算统计
                    df = pd.read_parquet(parquet_file)
                    
                    # 处理gripper_open_scale_state
                    if 'gripper_open_scale_state' in df.columns:
                        scale_data = np.vstack(df['gripper_open_scale_state'].values)
                        if len(scale_data.shape) == 1:
                            scale_data = scale_data.reshape(-1, 1)
                        episode_stats['stats']['gripper_open_scale_state'] = {
                            "min": scale_data.min(axis=0).tolist(),
                            "max": scale_data.max(axis=0).tolist(),
                            "mean": scale_data.mean(axis=0).tolist(),
                            "std": scale_data.std(axis=0).tolist(),
                            "count": [len(scale_data)]
                        }
                    
                    # 处理gripper_open_scale_action
                    if 'gripper_open_scale_action' in df.columns:
                        scale_data = np.vstack(df['gripper_open_scale_action'].values)
                        if len(scale_data.shape) == 1:
                            scale_data = scale_data.reshape(-1, 1)
                        episode_stats['stats']['gripper_open_scale_action'] = {
                            "min": scale_data.min(axis=0).tolist(),
                            "max": scale_data.max(axis=0).tolist(),
                            "mean": scale_data.mean(axis=0).tolist(),
                            "std": scale_data.std(axis=0).tolist(),
                            "count": [len(scale_data)]
                        }
                    
                    new_stats_lines.append(json.dumps(episode_stats))
            
            # 保存新的stats文件
            with open(self.stats_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_stats_lines))
            
            logger.info(f"已更新episodes_stats.jsonl: {self.stats_path}")
            return True
        except Exception as e:
            logger.error(f"更新episodes_stats.jsonl失败: {e}")
            return False

    def run(self) -> bool:
        """
        执行完整的归一化流程
        
        Returns:
            是否成功完成所有步骤
        """
        try:
            logger.info("开始执行gripper_open归一化流程...")
            
            # 1. 收集统计信息并检测量程
            self.collect_gripper_stats()
            
            # 2. 处理parquet文件，添加归一化字段
            processed_files = self.process_parquet_files()
            
            # 3. 更新info.json
            updated_info = self.update_info_json()
            
            # 4. 更新episodes_stats.jsonl
            updated_stats = self.update_episodes_stats()
            
            # 验证所有关键步骤是否成功
            all_success = processed_files and updated_info and updated_stats
            
            if all_success:
                logger.info("归一化流程执行完成！")
            else:
                logger.warning("归一化流程部分步骤执行失败")
            
            return all_success
            
        except Exception as e:
            logger.error(f"归一化流程执行失败: {e}", exc_info=True)
            return False


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 替换为你的数据集路径
    DATASET_PATH = "/path/to/your/converted_dataset"
    
    # 初始化并运行
    normalizer = GripperOpenNormalizer(DATASET_PATH)
    success = normalizer.run()
    if success:
        logger.info("夹爪归一化完全执行成功！")
    else:
        logger.error("夹爪归一化执行失败！")