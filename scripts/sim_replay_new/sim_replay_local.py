"""Local simulation replay script
参数说明：
- repo_path: 数据集路径
- config_name: 配置文件名称（例如 agilex）
- data_source: 数据来源文件夹（默认为 data，可选 sa_dpp）
- data_type: 加载state还是action（默认为 all）
- episode_idx: episode 索引（默认为 0）
- version: 机器人版本（默认为 default_version）
"""

""" Usage example:
python scripts/sim_replay_new/sim_replay_local.py \
    --repo_path /home/user/process_symmetry/processed_dataset \
    --config_name realman \
    --data_source data \
    --data_type all \
    --episode_idx 0

python scripts/sim_replay_new/sim_replay_local.py \
    --repo_path /home/user/robocoin-dataset_new/Agilex_Cobot_Magic_pour_water_into_cup_0 \
    --config_name agilex \
    --data_source data \
    --data_type all \
    --episode_idx 0 \
    --version default_version
"""

import argparse
import sys
import os
import json
import random
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
    try:
        # Try importing from local directory (scripts/sim_replay_new)
        sys.path.append(str(Path(__file__).parent))
        from sim_replay import run_replay
    except ImportError:
        # Try adding src/robocoin_dataset/sim_replay_new to path
        sys.path.append(str(src_path / "robocoin_dataset" / "sim_replay_new"))
        from sim_replay import run_replay

def main():
    parser = argparse.ArgumentParser(description="Local Replay Script")
    parser.add_argument("--repo_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--config_name", type=str, required=True, help="Name of the config file (e.g. agilex)")
    parser.add_argument("--data_source", type=str, default="data", choices=["data", "sa_dpp"], help="Data source folder (default: data)")
    parser.add_argument("--data_type", type=str, default="all", help="Data columns to load (default: all)")
    parser.add_argument("--episode_idx", type=int, default=None, help="Episode index (default: Random)")
    parser.add_argument("--version", type=str, default="default_version", help="Version key in the config file (e.g. default_version)")
    
    args = parser.parse_args()
    
    episode_idx = args.episode_idx
    if episode_idx is None:
        try:
            repo_path = Path(args.repo_path)
            # 1. Try meta/info.json
            info_path = repo_path / "meta" / "info.json"
            total_episodes = 0
            if info_path.exists():
                with open(info_path, "r") as f:
                    info = json.load(f)
                    total_episodes = info.get("total_episodes", 0)
                    if not total_episodes and "splits" in info:
                        total_episodes = sum(s.get("num_episodes", 0) for s in info["splits"].values())
            
            # 2. If still 0, try scanning videos (heuristic: count files in first camera of first chunk)
            if total_episodes == 0:
                 video_dir = repo_path / "videos"
                 if video_dir.exists():
                     # Check chunk-000
                     chunk_dir = video_dir / "chunk-000"
                     if chunk_dir.exists():
                         # Get first camera dir
                         cam_dirs = [d for d in chunk_dir.iterdir() if d.is_dir()]
                         if cam_dirs:
                             # Count mp4 files
                             total_episodes = len(list(cam_dirs[0].glob("episode_*.mp4")))
            
            if total_episodes > 0:
                episode_idx = random.randint(0, total_episodes - 1)
                print(f"Randomly selected episode: {episode_idx} (Total: {total_episodes})")
            else:
                print("Warning: Could not determine total episodes. Defaulting to 0.")
                episode_idx = 0
        except Exception as e:
            print(f"Error determining random episode: {e}")
            episode_idx = 0
            
    print(f"Running replay for episode {episode_idx}...")
    print(f"Config: {args.config_name}, Version: {args.version}")
    
    run_replay(
        repo_path=args.repo_path,
        config_name=args.config_name,
        data_source=args.data_source, 
        data_type=args.data_type,
        episode_idx=episode_idx,
        auto_close=False, # Keep charts open for local testing
        version=args.version
    )
    
    print("\nReplay finished. Charts should remain open.")
    input("Type any key to exit this script (charts may close depending on backend).")
    try:
        # Keep main thread alive if needed by some backends, though rerun usually runs in separate process/thread
        # But auto_close=False in run_replay should handle the blocking/input wait.
        # However, run_replay implementation of auto_close=False might just skip the close call.
        # If run_replay doesn't block, we need to block here.
        # Let's check run_replay implementation again.
        pass
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
