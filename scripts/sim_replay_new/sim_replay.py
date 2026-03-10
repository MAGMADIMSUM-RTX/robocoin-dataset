"""Database Interactive Replay Script
Usage:
python scripts/sim_replay_new/sim_replay.py \
    --db_file_path db/my_config.yaml \
    --device_model agilex_cobot_decoupled_magic \
    --device_model_version default_version \
    --log_dir ./logs/sim_replay
"""

import argparse
import logging
import sys
import time
import random
import subprocess
import yaml
from pathlib import Path
import json

# Add project root and src to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2] # scripts/sim_replay_new -> scripts -> root
src_path = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB, TaskStatus
from robocoin_dataset.utils.logger import setup_logger
from robocoin_dataset.sim_replay.sim_replay import _sync_sim_replay_tasks, _gen_one_sim_replay_task
from robocoin_dataset.sim_replay_new.sim_replay import run_replay

class SimReplayNew:
    def __init__(self, db_file_path: str | Path, sim_replay_config_path: str | Path, logger: logging.Logger | None = None) -> None:
        self.db_file_path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.logger = logger or logging.getLogger(__name__)
        self.sim_replay_config_path = Path(sim_replay_config_path).expanduser().absolute()
        self.config_mapping = self._load_config_mapping()

    def _load_config_mapping(self) -> dict:
        if not self.sim_replay_config_path.exists():
            self.logger.warning(f"Config mapping file {self.sim_replay_config_path} not found.")
            return {}
        try:
            with open(self.sim_replay_config_path, "r") as f:
                data = yaml.safe_load(f)
                # Ensure it's a dict of strings (device_model -> config_name)
                mapping = {}
                if data:
                    for k, v in data.items():
                        if isinstance(v, str):
                            mapping[k] = v
                return mapping
        except Exception as e:
            self.logger.error(f"Error loading config mapping: {e}")
            return {}


    def sim_replay_datasets(self, device_model: str, device_model_version: str = "", target_dataset_uuid: str | None = None) -> None:
        self.logger.info(f"Syncing tasks for device_model={device_model}, version={device_model_version}")
        with self.db.with_session() as session:
            _sync_sim_replay_tasks(session, device_model, device_model_version=device_model_version, target_dataset_uuid=target_dataset_uuid)

        while True:
            try:
                with self.db.with_session() as session:
                    dataset_uuid, qced_repo_gen_path, current_device_model, current_device_model_version = _gen_one_sim_replay_task(
                        session=session, 
                        device_model=device_model, 
                        device_model_version=device_model_version, 
                        target_dataset_uuid=target_dataset_uuid
                    )

                if not dataset_uuid:
                    self.logger.info("No more tasks to process.")
                    break
                
                self.logger.info(f"Processing dataset {dataset_uuid} (Model: {current_device_model}, Version: {current_device_model_version})")
                
                # Map device model to config name
                config_name = self.config_mapping.get(current_device_model)
                
                if not config_name:
                    # Heuristic: use first part of name
                    config_name = current_device_model.split("_")[0].lower()
                    self.logger.warning(f"No config mapping found for {current_device_model}, trying {config_name}")

                # Get total episodes and pick random
                episode_idx = 0
                with self.db.with_session() as session:
                    item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                    if item:
                        total_episodes = item.total_episodes
                        self.logger.info(f"Dataset {dataset_uuid} total_episodes from DB: {total_episodes}")
                        if qced_repo_gen_path:
                            # 1. 将路径转为 Path 对象，方便拼接和判断
                            repo_path = Path(qced_repo_gen_path)
                            # 2. 拼接 meta/info.json 路径
                            info_json_path = repo_path / "meta" / "info.json"
                            
                            # 3. 检查文件是否存在
                            if info_json_path.exists():
                                # 4. 读取并解析 JSON 文件
                                with open(info_json_path, "r", encoding="utf-8") as f:
                                    info_data = json.load(f)
                                
                                # 5. 提取 total_episodes 字段（做容错处理）
                                if "total_episodes" in info_data:
                                    total_episodes = info_data["total_episodes"]
                                    print(f"成功获取 total_episodes: {total_episodes}")
                                else:
                                    print(f"警告: {info_json_path} 中未找到 total_episodes 字段")
                                        
                        # If total_episodes is invalid in DB, try to count files
                        # if not total_episodes or total_episodes <= 0:
                        #     self.logger.info(f"DB has invalid total_episodes, checking files in {qced_repo_gen_path}...")
                        #     try:
                        #         data_path = Path(qced_repo_gen_path) / "data"
                        #         if data_path.exists():
                        #             count = 0
                        #             # Count parquet files in all chunk-* directories
                        #             for chunk_dir in data_path.glob("chunk-*"):
                        #                 if chunk_dir.is_dir():
                        #                     # Fast count using iterator
                        #                     count += sum(1 for _ in chunk_dir.glob("episode_*.parquet"))
                                    
                        #             if count > 0:
                        #                 total_episodes = count
                        #                 self.logger.info(f"Counted {total_episodes} episodes from filesystem.")
                        #     except Exception as e:
                        #         self.logger.error(f"Error counting episodes: {e}")
                        
                        if total_episodes and total_episodes > 0:
                            episode_idx = random.randint(0, total_episodes - 1)
                            self.logger.info(f"Selected random episode {episode_idx} from total {total_episodes}")
                        else:
                             self.logger.warning(f"Could not determine total episodes for {dataset_uuid}, defaulting to 0")
                    else:
                        self.logger.warning(f"Dataset {dataset_uuid} not found in DB when selecting episode")

                # Close previous Rerun viewer if any
                try:
                    subprocess.run(["pkill", "rerun"], check=False)
                    time.sleep(0.5) 
                except Exception:
                    pass

                # Run replay
                print(f"\n--- Replaying Dataset {dataset_uuid} Episode {episode_idx} ---")
                print("Press Ctrl+C in the terminal to stop replay and provide feedback.")
                
                fail_reason = None
                status = TaskStatus.PENDING # Default if something goes wrong before status set
                
                try:
                    run_replay(
                        repo_path=qced_repo_gen_path,
                        config_name=config_name,
                        data_source="sa_dpp", 
                        data_type="all",
                        episode_idx=episode_idx,
                        auto_close=True,
                        version=current_device_model_version or "default_version"
                    )
                except Exception as e:
                    self.logger.error(f"Error during replay: {e}")
                
                # Interactive status check
                exit_after_update = False
                while True:
                    user_input = input(f"Dataset {dataset_uuid}: 通过 (p) / 失败 (f) / 通过并退出 (c)? [p]: ").strip().lower()
                    if user_input in ["", "p", "pass"]:
                        status = TaskStatus.COMPLETED
                        break
                    elif user_input in ["c", "close", "exit"]:
                        status = TaskStatus.COMPLETED
                        exit_after_update = True
                        break
                    elif user_input in ["f", "fail"]:
                        status = TaskStatus.FAILED
                        while True:
                            reason = input("输入错误原因: ").strip()
                            if reason:
                                fail_reason = f"episode {episode_idx} error: {reason}"
                                break
                            print("原因不能为空")
                        break
                    else:
                        print("无效输入. Please enter 'p', 'f', or 'c'.")
                
                # Update status in DB
                with self.db.with_session() as session:
                    item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                    if item:
                        item.sim_replay_status = status
                        if status == TaskStatus.FAILED and fail_reason:
                            item.sim_replay_error_msg = fail_reason
                            self.logger.info(f"Updated status to FAILED: {fail_reason}")
                        else:
                            self.logger.info(f"Updated status to {status}")
                
                if exit_after_update:
                    self.logger.info("Exiting as requested.")
                    break

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received, exiting...")
                break
            except Exception as e:
                self.logger.error(f"Error in replay loop: {e}")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database Interactive Replay Script")
    parser.add_argument("--db_file_path", type=str, required=True, help="Path to the database file")
    parser.add_argument("--sim_replay_config_path", type=str, default="scripts/sim_replay_new/configs/sim_replay_map.yaml", help="Path to the sim replay map config")
    parser.add_argument("--device_model", type=str, required=True, help="Device model filter")
    parser.add_argument("--device_model_version", type=str, default="", help="Device model version filter")
    parser.add_argument("--target_dataset_uuid", type=str, default=None, help="Target dataset UUID")
    parser.add_argument("--log_dir", type=str, default="./logs/sim_replay", help="Log directory")

    args = parser.parse_args()

    logger = setup_logger(
        name="sim_replay_interactive",
        log_dir=Path(args.log_dir),
        level=logging.INFO,
    )

    replayer = SimReplayNew(
        db_file_path=args.db_file_path,
        sim_replay_config_path=args.sim_replay_config_path,
        logger=logger
    )

    replayer.sim_replay_datasets(
        device_model=args.device_model,
        device_model_version=args.device_model_version,
        target_dataset_uuid=args.target_dataset_uuid
    )
