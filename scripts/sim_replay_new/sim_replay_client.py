import argparse
import asyncio
import logging
import os

# Suppress Rerun warnings
os.environ["RERUN_LOG"] = "error"
logging.getLogger("rerun").setLevel(logging.ERROR)

import sys
import random
import subprocess
import time
import traceback
from pathlib import Path

# Add project root and src to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from robocoin_dataset.distribution_computation.task_client import TaskClient
from robocoin_dataset.distribution_computation.constant import (
    DATASET_UUID,
    TASK_ID,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
    TASK_FAILED,
    ERR_MSG,
    TASK_RESULT_CONTENT
)
from robocoin_dataset.utils.logger import setup_logger
from robocoin_dataset.sim_replay_new.sim_replay import run_replay

# Constants matching Server
LEFORMAT_PATH = "leformat_path"
DEVICE_MODEL_VERSION = "device_model_version"
CONFIG_NAME = "config_name"
TOTAL_EPISODES = "total_episodes"

class SimReplayNewClient(TaskClient):
    def __init__(
        self,
        server_uri: str = "ws://localhost:8767",
        heartbeat_interval: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            logger=logger,
        )

    def get_task_category(self) -> str:
        return "simulation_replay_new"

    def generate_task_request_desc(self) -> dict:
        return {}

    async def process_task(self, task_data: dict) -> dict:
        # Run sync task in executor to avoid blocking heartbeat
        loop = asyncio.get_event_loop()
        task_id = task_data.get(TASK_ID)
        
        try:
            task_result = await loop.run_in_executor(None, self._sync_process_task, task_data)
            return task_result
        except Exception as e:
            error_msg = f"{e}\n{traceback.format_exc()}"
            if self.logger:
                self.logger.error(f"Task {task_id} failed: {error_msg}")
            return {
                TASK_RESULT_STATUS: TASK_FAILED,
                ERR_MSG: error_msg,
                TASK_RESULT_CONTENT: {},
            }

    def _sync_process_task(self, task_data: dict) -> dict:
        dataset_uuid = task_data.get(DATASET_UUID)
        qced_repo_gen_path = task_data.get(LEFORMAT_PATH)
        config_name = task_data.get(CONFIG_NAME)
        device_model_version = task_data.get(DEVICE_MODEL_VERSION)
        total_episodes = task_data.get(TOTAL_EPISODES, 0)
        
        self.logger.info(f"Processing dataset {dataset_uuid}")
        
        # Episode Selection Logic
        episode_idx = 0
        if not total_episodes or total_episodes <= 0:
            self.logger.info(f"Server provided invalid total_episodes, checking files in {qced_repo_gen_path}...")
            try:
                data_path = Path(qced_repo_gen_path) / "data"
                if data_path.exists():
                    count = 0
                    for chunk_dir in data_path.glob("chunk-*"):
                        if chunk_dir.is_dir():
                            count += sum(1 for _ in chunk_dir.glob("episode_*.parquet"))
                    if count > 0:
                        total_episodes = count
                        self.logger.info(f"Counted {total_episodes} episodes from filesystem.")
            except Exception as e:
                self.logger.error(f"Error counting episodes: {e}")

        if total_episodes and total_episodes > 0:
            episode_idx = random.randint(0, total_episodes - 1)
            self.logger.info(f"Selected random episode {episode_idx} from total {total_episodes}")
        else:
             self.logger.warning(f"Could not determine total episodes for {dataset_uuid}, defaulting to 0")

        # Close previous Rerun viewer
        try:
            subprocess.run(["pkill", "rerun"], check=False)
            time.sleep(0.5)
        except Exception:
            pass

        # Run replay
        print(f"\n--- Replaying Dataset {dataset_uuid} Episode {episode_idx} ---")
        print("Press Ctrl+C in the terminal to stop replay and provide feedback.")
        
        fail_reason = None
        status = TASK_SUCCESS # Tentative, waits for user confirmation
        
        try:
            run_replay(
                repo_path=qced_repo_gen_path,
                config_name=config_name,
                data_source="state_action_data", # Changed from sa_dpp
                data_type="all",
                episode_idx=episode_idx,
                auto_close=True,
                version=device_model_version or "default_version"
            )
        except Exception as e:
            self.logger.error(f"Error during replay: {e}\n{traceback.format_exc()}")
            # If replay fails, we can either set a fail status or continue to interactive check
            final_status = TASK_FAILED
            final_msg = f"Replay error: {e}\n{traceback.format_exc()}"
            # Don't return yet, let the user decide if it's a real fail or just a temporary issue
        
        # Interactive status check
        exit_after_update = False
        final_status = TASK_SUCCESS
        final_msg = ""
        
        while True:
            try:
                user_input = input(f"Dataset {dataset_uuid}: 通过 (p) / 失败 (f) / 通过并退出 (c)? [p]: ").strip().lower()
                if user_input in ["", "p", "pass"]:
                    final_status = TASK_SUCCESS
                    break
                elif user_input in ["c", "close", "exit"]:
                    final_status = TASK_SUCCESS
                    exit_after_update = True
                    break
                elif user_input in ["f", "fail"]:
                    final_status = TASK_FAILED
                    while True:
                        reason = input("输入错误原因: ").strip()
                        if reason:
                            final_msg = f"episode {episode_idx} error: {reason}"
                            break
                        print("原因不能为空")
                    break
                else:
                    print("无效输入. Please enter 'p', 'f', or 'c'.")
            except EOFError:
                # Handle case where input stream is closed
                self.logger.error("Input stream closed unexpectedly.")
                break

        if exit_after_update:
            self.logger.info("User requested exit after this task.")
            print("Please Ctrl+C to exit client.")
            
        return {
            TASK_RESULT_STATUS: final_status,
            ERR_MSG: final_msg,
            TASK_RESULT_CONTENT: {}
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sim Replay New Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--log_dir", type=str, default="./logs/sim_replay_new_client")
    
    args = parser.parse_args()
    
    logger = setup_logger("sim_replay_new_client", Path(args.log_dir))
    
    client = SimReplayNewClient(
        server_uri=f"ws://{args.host}:{args.port}",
        logger=logger
    )
    
    asyncio.run(client.run())
