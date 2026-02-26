"""Local simulation replay script
参数说明：
- repo_path: 数据集路径
- config_name: 配置文件名称（例如 agilex）
- data_source: 数据来源文件夹（默认为 data，可选 sa_dpp）
- data_type: 加载state还是action（默认为 all）
- episode_idx: episode 索引（默认为 0）
"""

""" Usage example:
python scripts/sim_replay_new/sim_replay_local.py \
    --repo_path /home/user/process_symmetry/processed_dataset \
    --config_name realman \
    --data_source data \
    --data_type all \
    --episode_idx 0

python scripts/sim_replay_new/sim_replay_local.py \
    --repo_path /home/user/process_symmetry/Cobot_Magic_move_plate_qced_hardlink/ \
    --config_name agilex \
    --data_source data \
    --data_type all \
    --episode_idx 0
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root and src to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2] # scripts/sim_replay_new -> scripts -> root
src_path = project_root / "src"

# Add root to path so that 'configs' module can be found if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src to path so that 'robocoin_dataset' package can be found
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from robocoin_dataset.sim_replay_new.sim_replay import run_replay
except ImportError:
    # Fallback if robocoin_dataset is not a package or structure is different
    # Try adding src/robocoin_dataset/sim_replay_new to path
    sys.path.append(str(src_path / "robocoin_dataset" / "sim_replay_new"))
    from sim_replay import run_replay

def main():
    parser = argparse.ArgumentParser(description="Local Replay Script")
    parser.add_argument("--repo_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--config_name", type=str, required=True, help="Name of the config file (e.g. agilex)")
    parser.add_argument("--data_source", type=str, default="data", choices=["data", "sa_dpp"], help="Data source folder (default: data)")
    parser.add_argument("--data_type", type=str, default="all", help="Data columns to load (default: all)")
    parser.add_argument("--episode_idx", type=int, default=0, help="Episode index (default: 0)")
    
    args = parser.parse_args()
    
    run_replay(
        repo_path=args.repo_path,
        config_name=args.config_name,
        data_source=args.data_source,
        data_type=args.data_type,
        episode_idx=args.episode_idx
    )

if __name__ == "__main__":
    main()
