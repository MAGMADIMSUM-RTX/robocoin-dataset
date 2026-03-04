import argparse
import asyncio
import logging
import multiprocessing as mp


import sys
from pathlib import Path
# Add project root and src to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


from robocoin_dataset.annotation.motion_annotation.motion_annotation_data_post_process import (
    MotionAnnotationDataPostProcessClient,
)
from robocoin_dataset.utils.logger import setup_logger


async def run_client_process(
    server_uri: str,
    heartbeat_interval: float,
    log_path: str,
    process_id: int,
) -> None:
    """
    每个进程运行的异步客户端逻辑。
    """
    # 为每个进程创建独立的日志文件或使用共享日志但区分进程
    logger = setup_logger(
        name=f"motion_annotation_client{process_id}",
        log_dir=log_path,
        level=logging.ERROR,
    )

    client = MotionAnnotationDataPostProcessClient(
        server_uri=server_uri,
        heartbeat_interval=heartbeat_interval,
        logger=logger,
    )
    await client.run()


def client_process_main(
    server_uri: str,
    heartbeat_interval: float,
    log_path: str,
    process_id: int,
) -> None:
    """
    多进程入口函数，每个进程启动自己的 asyncio 事件循环。
    """
    asyncio.run(
        run_client_process(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            log_path=log_path,
            process_id=process_id,
        )
    )


def main() -> None:
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
        default=8768,
        help="server port to connect to.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Path to the log directory",
    )

    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=10.0,
        help="Heartbeat interval for each client.",
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=4,
        help="Number of concurrent client processes to spawn.",
    )

    args = parser.parse_args()

    server_uri = f"ws://{args.host}:{args.port}"
    num_clients = max(1, min(args.num_clients, 8))

    server_uri = f"ws://{args.host}:{args.port}"

    # 使用 multiprocessing 启动多个客户端进程
    processes = []
    for i in range(num_clients):
        proc = mp.Process(
            target=client_process_main,
            kwargs=dict(
                server_uri=server_uri,
                heartbeat_interval=args.heartbeat_interval,
                log_path=args.log_dir,
                process_id=i,
            ),
        )
        proc.start()
        processes.append(proc)

    print(f"Started {num_clients} client processes. Waiting for them to finish...")

    try:
        for proc in processes:
            proc.join()  # 等待所有进程结束
    except KeyboardInterrupt:
        print("\nShutting down clients...")
        for proc in processes:
            proc.terminate()
            proc.join(timeout=2)


if __name__ == "__main__":
    # Windows 兼容性：避免多进程重复执行入口
    mp.set_start_method("spawn", force=True)
    main()


"""Usage:
python scripts/annotation/motion_annotation/motion_annotation_client.py \
    --host=172.16.13.140 \
    --port=2070 \
    --heartbeat-interval=10.0 \
    --log_dir=./logs \
    --num-clients=8
"""
