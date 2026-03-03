"""
SimReplayNewServer is a task server that handles simulation replay tasks for the Robocoin dataset.

python scripts/sim_replay_new/sim_replay_server.py \
    --device_model Agilex_Cobot_Magic \
    --device_model_version default_version \
    --db_file_path db/postgresql_config.yaml \
    --sim_replay_config_path scripts/sim_replay_new/configs/sim_replay_map.yaml
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

# Add project root and src to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB, TaskStatus
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.distribution_computation.constant import (
    DATASET_UUID,
    DEVICE_MODEL,
    ERR_MSG,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
    TASK_FAILED,
)
from robocoin_dataset.sim_replay.sim_replay import _sync_sim_replay_tasks
from robocoin_dataset.utils.logger import setup_logger

# Define constants for task content
LEFORMAT_PATH = "leformat_path" # Keeping this key name for compatibility or consistency
DEVICE_MODEL_VERSION = "device_model_version"
CONFIG_NAME = "config_name"
TOTAL_EPISODES = "total_episodes"

class SimReplayNewServer(TaskServer):
    def __init__(
        self,
        db_file_path: str | Path,
        sim_replay_config_path: str | Path,
        host: str = "0.0.0.0",
        port: int = 8767,
        heartbeat_interval: float = 30.0,
        device_model: str = "",
        device_model_version: str = "",
        target_dataset_uuid: str | None = None,
        timeout: float = 15.0,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            logger=logger,
            host=host,
            port=port,
            heartbeat_interval=heartbeat_interval,
            timeout=timeout,
        )
        self.db_file_path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.logger = logger or logging.getLogger(__name__)
        
        self.device_model = device_model
        self.device_model_version = device_model_version
        self.target_dataset_uuid = target_dataset_uuid
        
        self.sim_replay_config_path = Path(sim_replay_config_path).expanduser().absolute()
        self.config_mapping = self._load_config_mapping()

        self.logger.info(
            f"SimReplayNew Server started, "
            f"device_model={self.device_model}, "
            f"device_model_version={self.device_model_version}, "
            f"target_dataset_uuid={self.target_dataset_uuid}"
        )
        
        # Initial sync
        with self.db.with_session() as session:
            _sync_sim_replay_tasks(
                session, 
                device_model=self.device_model if self.device_model else None,
                device_model_version=self.device_model_version if self.device_model_version else None,
                target_dataset_uuid=self.target_dataset_uuid
            )

    def _load_config_mapping(self) -> dict:
        if not self.sim_replay_config_path.exists():
            self.logger.warning(f"Config mapping file {self.sim_replay_config_path} not found.")
            return {}
        try:
            with open(self.sim_replay_config_path, "r") as f:
                data = yaml.safe_load(f)
                mapping = {}
                if data:
                    for k, v in data.items():
                        if isinstance(v, str):
                            mapping[k] = v
                return mapping
        except Exception as e:
            self.logger.error(f"Error loading config mapping: {e}")
            return {}

    def get_task_category(self) -> str:
        return "simulation_replay_new"

    def generate_task_content(self) -> dict | None:
        with self.db.with_session() as session:
            # Reusing the query logic from SimReplayServer but adapted
            query = session.query(DatasetDB).filter(
                and_(
                    DatasetDB.sa_dpp_status == TaskStatus.COMPLETED,
                    or_(
                        DatasetDB.sim_replay_status == TaskStatus.PENDING,
                        and_(
                            DatasetDB.sim_replay_status == TaskStatus.COMPLETED,
                            DatasetDB.sim_replay_version_ps < DatasetDB.sa_dpp_version,
                        ),
                    ),
                )
            )
            
            if self.target_dataset_uuid:
                query = query.filter(DatasetDB.dataset_uuid == self.target_dataset_uuid)
            elif self.device_model:
                query = query.filter(DatasetDB.device_model == self.device_model)
                
            if self.device_model_version and not self.target_dataset_uuid:
                query = query.filter(DatasetDB.device_model_version == self.device_model_version)

            item = query.first()

            if not item:
                return None

            # Mark as processing
            item.sim_replay_status = TaskStatus.PROCESSING
            item.sim_replay_version = item.sim_replay_version + 1
            item.sim_replay_version_ps = item.sa_dpp_version
            session.commit()
            
            # Resolve config name
            config_name = self.config_mapping.get(item.device_model)
            if not config_name:
                config_name = item.device_model.split("_")[0].lower()
                self.logger.warning(f"No config mapping found for {item.device_model}, trying {config_name}")

            return {
                DATASET_UUID: item.dataset_uuid,
                LEFORMAT_PATH: item.qced_repo_gen_path,
                DEVICE_MODEL: item.device_model,
                DEVICE_MODEL_VERSION: item.device_model_version,
                CONFIG_NAME: config_name,
                TOTAL_EPISODES: item.total_episodes or 0
            }

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        ds_uuid = task_content.get(DATASET_UUID)
        task_status = task_result_content.get(TASK_RESULT_STATUS)
        error_msg = task_result_content.get(ERR_MSG)

        status = TaskStatus.COMPLETED if task_status == TASK_SUCCESS else TaskStatus.FAILED

        with self.db.with_session() as session:
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
            if item is None:
                self.logger.error(f"Dataset {ds_uuid} not found in dataset DB.")
                return

            item.sim_replay_status = status
            if status == TaskStatus.FAILED and error_msg:
                item.sim_replay_error_msg = error_msg
            
            session.commit()
            self.logger.info(f"Updated {ds_uuid} sim replay status to {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sim Replay New Server")
    parser.add_argument("--db_file_path", type=str, required=True)
    parser.add_argument("--sim_replay_config_path", type=str, required=True) # yaml file for mapping
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--device_model", type=str, default="")
    parser.add_argument("--device_model_version", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="./logs/sim_replay_new_server")
    
    args = parser.parse_args()
    
    logger = setup_logger("sim_replay_new_server", Path(args.log_dir))
    
    server = SimReplayNewServer(
        db_file_path=args.db_file_path,
        sim_replay_config_path=args.sim_replay_config_path,
        host=args.host,
        port=args.port,
        device_model=args.device_model,
        device_model_version=args.device_model_version,
        logger=logger
    )
    
    import asyncio
    asyncio.run(server.start())
