import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple

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
        
        # 存储归一化参数（全局min/max）
        self.gripper_stats = {}

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

    def _find_gripper_indices(self) -> Dict[str, Dict[str, int]]:
        """
        查找gripper_open字段在observation.state和action中的索引
        返回格式: 
        {
            "observation.state": {"left_gripper_open": 6, "right_gripper_open": 19},
            "action": {"left_gripper_open": 6, "right_gripper_open": 19}
        }
        """
        indices = {}
        
        # 检查observation.state
        state_names = self.info_data['features']['observation.state']['names']
        indices['observation.state'] = {
            'left_gripper_open': state_names.index('left_gripper_open'),
            'right_gripper_open': state_names.index('right_gripper_open')
        }
        
        # 检查action
        action_names = self.info_data['features']['action']['names']
        indices['action'] = {
            'left_gripper_open': action_names.index('left_gripper_open'),
            'right_gripper_open': action_names.index('right_gripper_open')
        }
        
        return indices

    def collect_gripper_stats(self) -> Dict[str, Dict[str, float]]:
        """
        收集所有episode的gripper_open全局min/max（用于归一化）
        返回格式:
        {
            "left_gripper_open": {"min": 0.05, "max": 0.09},
            "right_gripper_open": {"min": 0.03, "max": 0.09}
        }
        """
        # 先从episodes_stats.jsonl获取初始统计
        left_min, left_max = float('inf'), -float('inf')
        right_min, right_max = float('inf'), -float('inf')
        
        # 读取stats文件
        with open(self.stats_path, 'r', encoding='utf-8') as f:
            for line in f:
                episode_stats = json.loads(line)
                stats = episode_stats['stats']
                
                # 处理observation.state
                state_stats = stats.get('observation.state', {})
                if state_stats:
                    # left_gripper_open (第7个元素，索引6)
                    left_val_min = state_stats['min'][self.gripper_indices['observation.state']['left_gripper_open']]
                    left_val_max = state_stats['max'][self.gripper_indices['observation.state']['left_gripper_open']]
                    # right_gripper_open (第20个元素，索引19)
                    right_val_min = state_stats['min'][self.gripper_indices['observation.state']['right_gripper_open']]
                    right_val_max = state_stats['max'][self.gripper_indices['observation.state']['right_gripper_open']]
                    
                    left_min = min(left_min, left_val_min)
                    left_max = max(left_max, left_val_max)
                    right_min = min(right_min, right_val_min)
                    right_max = max(right_max, right_val_max)
                
                # 处理action（通常和state范围一致）
                action_stats = stats.get('action', {})
                if action_stats:
                    left_val_min = action_stats['min'][self.gripper_indices['action']['left_gripper_open']]
                    left_val_max = action_stats['max'][self.gripper_indices['action']['left_gripper_open']]
                    right_val_min = action_stats['min'][self.gripper_indices['action']['right_gripper_open']]
                    right_val_max = action_stats['max'][self.gripper_indices['action']['right_gripper_open']]
                    
                    left_min = min(left_min, left_val_min)
                    left_max = max(left_max, left_val_max)
                    right_min = min(right_min, right_val_min)
                    right_max = max(right_max, right_val_max)
        
        # 验证统计值
        if left_min >= left_max or right_min >= right_max:
            logger.warning("从stats文件获取的范围异常，将从原始数据重新计算")
            return self._collect_stats_from_parquet()
        
        self.gripper_stats = {
            'left_gripper_open': {'min': left_min, 'max': left_max},
            'right_gripper_open': {'min': right_min, 'max': right_max}
        }
        
        logger.info(f"收集到gripper_open统计: {self.gripper_stats}")
        return self.gripper_stats

    def _collect_stats_from_parquet(self) -> Dict[str, Dict[str, float]]:
        """从parquet文件重新计算gripper_open的全局min/max"""
        left_vals = []
        right_vals = []
        
        # 遍历所有parquet文件
        parquet_files = list(self.data_dir.rglob("*.parquet"))
        logger.info(f"找到 {len(parquet_files)} 个parquet文件，开始读取...")
        
        for file in parquet_files:
            df = pd.read_parquet(file)
            
            # 处理observation.state
            if 'observation.state' in df.columns:
                state_data = np.vstack(df['observation.state'].values)
                left_vals.extend(state_data[:, self.gripper_indices['observation.state']['left_gripper_open']])
                right_vals.extend(state_data[:, self.gripper_indices['observation.state']['right_gripper_open']])
            
            # 处理action
            if 'action' in df.columns:
                action_data = np.vstack(df['action'].values)
                left_vals.extend(action_data[:, self.gripper_indices['action']['left_gripper_open']])
                right_vals.extend(action_data[:, self.gripper_indices['action']['right_gripper_open']])
        
        # 计算统计
        left_min, left_max = np.min(left_vals), np.max(left_vals)
        right_min, right_max = np.min(right_vals), np.max(right_vals)
        
        self.gripper_stats = {
            'left_gripper_open': {'min': left_min.item(), 'max': left_max.item()},
            'right_gripper_open': {'min': right_min.item(), 'max': right_max.item()}
        }
        
        logger.info(f"从原始数据收集到gripper_open统计: {self.gripper_stats}")
        return self.gripper_stats

    def normalize_gripper_value(self, value: float, gripper_type: str) -> float:
        """
        归一化单个夹爪值到[0,1]范围
        
        Args:
            value: 原始值
            gripper_type: 'left_gripper_open' 或 'right_gripper_open'
        
        Returns:
            归一化后的值
        """
        if gripper_type not in self.gripper_stats:
            raise ValueError(f"不支持的夹爪类型: {gripper_type}")
        
        stats = self.gripper_stats[gripper_type]
        if stats['max'] == stats['min']:
            return 0.0  # 避免除零
        
        # 归一化并限制范围
        normalized = (value - stats['min']) / (stats['max'] - stats['min'])
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return float(normalized)

    def process_parquet_files(self):
        """
        处理所有parquet文件：
        1. 提取gripper_open字段
        2. 归一化得到gripper_open_scale
        3. 保存新字段到parquet文件（字段名调整为gripper_open_scale_state/action）
        """
        if not self.gripper_stats:
            self.collect_gripper_stats()
        
        # 遍历所有parquet文件
        parquet_files = list(self.data_dir.rglob("*.parquet"))
        logger.info(f"开始处理 {len(parquet_files)} 个parquet文件...")
        
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                modified = False
                
                # 处理observation.state -> 添加gripper_open_scale_state
                if 'observation.state' in df.columns:
                    state_data = np.vstack(df['observation.state'].values)
                    
                    # 提取并归一化左右夹爪
                    left_gripper = state_data[:, self.gripper_indices['observation.state']['left_gripper_open']]
                    right_gripper = state_data[:, self.gripper_indices['observation.state']['right_gripper_open']]
                    
                    left_normalized = [self.normalize_gripper_value(v, 'left_gripper_open') for v in left_gripper]
                    right_normalized = [self.normalize_gripper_value(v, 'right_gripper_open') for v in right_gripper]
                    
                    # 组合成新字段 (left, right)
                    gripper_scale = np.column_stack([left_normalized, right_normalized])
                    df['gripper_open_scale_state'] = list(gripper_scale)  # 调整字段名
                    modified = True
                
                # 处理action -> 添加gripper_open_scale_action
                if 'action' in df.columns:
                    action_data = np.vstack(df['action'].values)
                    
                    # 提取并归一化左右夹爪
                    left_gripper = action_data[:, self.gripper_indices['action']['left_gripper_open']]
                    right_gripper = action_data[:, self.gripper_indices['action']['right_gripper_open']]
                    
                    left_normalized = [self.normalize_gripper_value(v, 'left_gripper_open') for v in left_gripper]
                    right_normalized = [self.normalize_gripper_value(v, 'right_gripper_open') for v in right_gripper]
                    
                    # 组合成新字段 (left, right)
                    gripper_scale = np.column_stack([left_normalized, right_normalized])
                    df['gripper_open_scale_action'] = list(gripper_scale)  # 调整字段名
                    modified = True
                
                # 保存修改后的文件
                if modified:
                    # 备份原文件
                    # backup_file = file.with_suffix('.parquet.bak')
                    # if not backup_file.exists():
                    #     file.rename(backup_file)
                    
                    # 保存新文件
                    df.to_parquet(file)
                    logger.info(f"已处理文件: {file}")
                else:
                    logger.info(f"文件无需要处理的字段: {file}")
                    
            except Exception as e:
                logger.error(f"处理文件 {file} 失败: {e}")
                continue

    def update_info_json(self):
        """更新info.json，添加顶级的gripper_open_scale_state/action字段描述"""
        # 移除旧的嵌套字段（如果存在）
        if 'observation.gripper_open_scale' in self.info_data['features']:
            del self.info_data['features']['observation.gripper_open_scale']
        if 'action.gripper_open_scale' in self.info_data['features']:
            del self.info_data['features']['action.gripper_open_scale']
        
        # 添加顶级的gripper_open_scale_state
        self.info_data['features']['gripper_open_scale_state'] = {
            "names": ["left_gripper_open_scale", "right_gripper_open_scale"],
            "dtype": "float32",
            "shape": [2]
        }
        
        # 添加顶级的gripper_open_scale_action
        self.info_data['features']['gripper_open_scale_action'] = {
            "names": ["left_gripper_open_scale", "right_gripper_open_scale"],
            "dtype": "float32",
            "shape": [2]
        }
        
        # 保存修改后的info.json
        with open(self.info_path, 'w', encoding='utf-8') as f:
            json.dump(self.info_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"已更新info.json: {self.info_path}")

    def update_episodes_stats(self):
        """更新episodes_stats.jsonl，添加顶级gripper_open_scale的统计信息"""
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
                    episode_stats['stats']['gripper_open_scale_action'] = {
                        "min": scale_data.min(axis=0).tolist(),
                        "max": scale_data.max(axis=0).tolist(),
                        "mean": scale_data.mean(axis=0).tolist(),
                        "std": scale_data.std(axis=0).tolist(),
                        "count": [len(scale_data)]
                    }
                
                new_stats_lines.append(json.dumps(episode_stats))
        
        # 备份原stats文件
        # backup_stats = self.stats_path.with_suffix('.jsonl.bak')
        # if not backup_stats.exists():
        #     self.stats_path.rename(backup_stats)
        
        # 保存新的stats文件
        with open(self.stats_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_stats_lines))
        
        logger.info(f"已更新episodes_stats.jsonl: {self.stats_path}")

    def run(self):
        """执行完整的归一化流程"""
        logger.info("开始执行gripper_open归一化流程...")
        
        # 1. 收集统计信息
        self.collect_gripper_stats()
        
        # 2. 处理parquet文件，添加归一化字段
        self.process_parquet_files()
        
        # 3. 更新info.json
        self.update_info_json()
        
        # 4. 更新episodes_stats.jsonl
        self.update_episodes_stats()
        
        logger.info("归一化流程执行完成！")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 替换为你的数据集路径
    DATASET_PATH = "/path/to/your/converted_dataset"
    
    # 初始化并运行
    normalizer = GripperOpenNormalizer(DATASET_PATH)
    normalizer.run()