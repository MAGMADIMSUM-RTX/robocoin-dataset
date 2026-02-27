import argparse
import logging
import sys
from pathlib import Path

# Add project root and src to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2] # scripts/sim_replay -> scripts -> root
src_path = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


from robocoin_dataset.sim_replay.sim_replay import SimReplay
from robocoin_dataset.utils.logger import setup_logger



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_file_path",
        type=str,
        default="",
        help="Path to the database file",
    )

    parser.add_argument(
        "--sim_replay_config_path",
        type=str,
        default="",
        help="Path to the annotation config classes",
    )

    parser.add_argument(
        "--device_model",
        type=str,
        default="",
        help="Device model to simulate",
    )

    parser.add_argument(
        "--device_model_version",
        type=str,
        default="",
        help="Device model version to simulate",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Path to the log directory",
    )

    args = parser.parse_args()
    db_file_path = Path(args.db_file_path).expanduser().absolute()
    sim_replay_config_path = Path(args.sim_replay_config_path).expanduser().absolute()
    device_model = args.device_model
    device_model_version = args.device_model_version

    if not db_file_path.exists():
        print(f"{db_file_path} does not exist")
        exit(1)

    logger = setup_logger(
        name="sim_replay",
        log_dir=Path(args.log_dir),
        level=logging.ERROR,
    )

    sim_replayer = SimReplay(
        db_file_path=db_file_path,
        sim_replay_config_path=sim_replay_config_path,
        logger=logger,
    )

    sim_replayer.sim_replay_datasets(device_model, device_model_version=device_model_version)


"""usage:
# realman_rmc_aidal
python scripts/sim_replay/sim_replay.py \
    --db_file_path db/postgresql_config.yaml \
    --sim_replay_config_path ./scripts/sim_replay/configs/sim_replay_config_path.yaml \
    --device_model agilex_cobot_decoupled_magic \
    --device_model_version default_version \
    --log_dir ./logs/sim_replay
"""
# galaxea
"""
python scripts/sim_replay/sim_replay.py \
    --db_file_path /mnt/db/datasets_new.db \
    --converter_factory_config_path ./scripts/sim_replay/configs/sim_replay_factory_config.yaml \
    --device_model yinhe \
    --device_model_version lite_version \
    --log_dir ./logs/sim_replay
"""