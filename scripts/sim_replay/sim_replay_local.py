#!/usr/bin/env python3
'''
python scripts/sim_replay/sim_replay_local.py \
  --repo_path /mnt/nas/synnas/成功区/Airbot_MMK2_Airbot_MMK2_storage_potato_right \
  --config_module robocoin_dataset.sim_replay.configs.mmk2_config \
  --config_class Mmk2LerobotSimReplayConfig \
  --replay_source sa_dpp

python scripts/sim_replay/sim_replay_local.py \
  --repo_path /mnt/nas/synnas/docker2/robocoin-datasets/Cobot_Magic_move_beverage \
  --config_module robocoin_dataset.sim_replay.configs.agilex_cobot_magic_config \
  --config_class AgilexCobotMagicLerobotSimReplayConfig \
  --replay_source sa_dpp
  --episode_idx 94
'''

import argparse
from pathlib import Path
from importlib import import_module
import sys

# Ensure project root is on sys.path so running this script directly
# can import the local `robocoin_dataset` package.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Prefer `src/` layout if present (e.g. src/robocoin_dataset)
_SRC_ROOT = _PROJECT_ROOT / "src"
if _SRC_ROOT.exists():
    root_to_add = _SRC_ROOT
else:
    root_to_add = _PROJECT_ROOT
if str(root_to_add) not in sys.path:
    sys.path.insert(0, str(root_to_add))

from robocoin_dataset.sim_replay.sim_replay import _sim_replay_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_path", help="Path to the converted dataset (leformat)")
    p.add_argument("--config_module", required=True, help="Python module path for replay config, e.g. mypkg.configs.my_cfg_module")
    p.add_argument("--config_class", required=True, help="Class name inside module, e.g. MySimReplayConfig")
    # p.add_argument("--episode_path", help="Optional path to a single episode parquet file to replay")
    p.add_argument("--replay_source", default="sa_dpp", help="Index of the episode to replay within the dataset")
    p.add_argument("--episode_idx", type=int, default=0, help="Index of the episode to replay within the dataset")
    args = p.parse_args()

    mod = import_module(args.config_module)
    cfg_cls = getattr(mod, args.config_class)
    cfg = cfg_cls()
    if not args.repo_path:
        p.error("--repo_path is required")
    repo = Path(args.repo_path).expanduser().absolute()
    _sim_replay_dataset(repo, cfg,replay_source=args.replay_source,episode_idx=args.episode_idx)

if __name__ == '__main__':
    main()