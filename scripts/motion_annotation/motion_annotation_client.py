import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from robocoin_dataset.annotation.motion_annotation.motion_annotation_data_post_process import (
    MotionAnnotationDataPostProcessClient,
)
from robocoin_dataset.utils.logger import setup_logger


async def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="server host to connect to.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="server port to connect to.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="path to the log directory",
    )

    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=10.0,
        help="heartbeat interval for each client.",
    )

    args = parser.parse_args()
    logger = setup_logger(
        name="motion_annotation_client",
        log_dir=Path(args.log_dir),
        level=logging.INFO,
    )

    server_uri = f"ws://{args.host}:{args.port}"
    motion_annotation_client = MotionAnnotationDataPostProcessClient(
        server_uri=server_uri, logger=logger, heartbeat_interval=args.heartbeat_interval
    )
    await motion_annotation_client.run()


if __name__ == "__main__":
    asyncio.run(main())

"""usage:
# realman_rmc_aidal
python scripts/motion_annotation/motion_annotation_client.py \
    --host=127.0.0.1 \
    --port=8766 \
    --heartbeat-interval=10.0 \
    --log_dir ./logs/motion_annotation
"""
