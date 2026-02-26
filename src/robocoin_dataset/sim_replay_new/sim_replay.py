import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import rerun as rr
import rerun.blueprint as rrb

# 将项目根目录加入路径，确保直接运行时导入正常
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # 从 src/robocoin_dataset/sim_replay_new 回到项目根目录
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.robocoin_dataset.sim_replay_new.SimReplayer import LerobotSimReplayer

def run_replay(repo_path, config_name, data_source="data", data_type="all", episode_idx=0):
    # 解析配置文件路径
    config_path = project_root / "configs" / "sim_replay" / f"{config_name}.yml"
    
    if not config_path.exists():
        # 兜底：用户可能传入了完整文件名
        if not config_name.endswith(".yml"):
            config_path_alt = project_root / "configs" / "sim_replay" / f"{config_name}"
            if config_path_alt.exists():
                config_path = config_path_alt
        
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Using config: {config_path}")
    
    # 使用 repo_path 的最后一段作为 Rerun 标题
    repo_name = Path(repo_path).name
    rr.init(repo_name, spawn=True)
    
    # 初始化 SimReplayer
    replayers = {}
    if data_type == "all":
        print("Initializing State Replayer...")
        replayers["state"] = LerobotSimReplayer(
            config_path=str(config_path),
            robot_version="default_version",
            repo_path=repo_path,
            episode_idx=episode_idx,
            data_source=data_source,
            data_type="state",
            headless=True,
            render_width=640,
            render_height=480
        )
        print("Initializing Action Replayer...")
        replayers["action"] = LerobotSimReplayer(
            config_path=str(config_path),
            robot_version="default_version",
            repo_path=repo_path,
            episode_idx=episode_idx,
            data_source=data_source,
            data_type="action",
            headless=True,
            render_width=640,
            render_height=480
        )
    else:
        print(f"Initializing {data_type} Replayer...")
        replayers[data_type] = LerobotSimReplayer(
            config_path=str(config_path),
            robot_version="default_version",
            repo_path=repo_path,
            episode_idx=episode_idx,
            data_source=data_source,
            data_type=data_type,
            headless=True,
            render_width=640,
            render_height=480
        )
    
    # 打开视频
    # 路径格式：videos/chunk-num/camera_name/episode_num.mp4
    chunk_idx = episode_idx // 1000
    chunk_dir_name = f"chunk-{chunk_idx:03d}"
    video_chunk_path = Path(repo_path) / "videos" / chunk_dir_name
    
    video_caps = {}
    if video_chunk_path.exists():
        print(f"Searching for videos in: {video_chunk_path}")
        for cam_dir in video_chunk_path.iterdir():
            if cam_dir.is_dir():
                # 检查视频文件
                video_path = cam_dir / f"episode_{episode_idx:06d}.mp4"
                if video_path.exists():
                    print(f"Found video: {video_path}")
                    cap = cv2.VideoCapture(str(video_path))
                    video_caps[cam_dir.name] = cap
                else:
                    print(f"Video not found in {cam_dir}: {video_path}")
    else:
        print(f"Warning: Video chunk directory not found at {video_chunk_path}")

    # Define layout keywords mapping
    def get_layout_pos(name):
        name_lower = name.lower()
        
        # Column: 0=Left, 1=Center, 2=Right
        col = 1
        if "left" in name_lower:
            col = 0
        elif "right" in name_lower:
            col = 2
            
        # Row: 0=Top, 1=Middle, 2=Bottom
        row = 1
        
        top_kws = ["head", "top", "upper", "global", "env"]
        bottom_kws = ["leg", "lower", "bottom", "wrist", "foot"]
        
        for kw in top_kws:
            if kw in name_lower:
                row = 0
                break
        
        if row == 1:
            for kw in bottom_kws:
                if kw in name_lower:
                    row = 2
                    break
        
        return col, row

    if not video_caps:
        video_container = rrb.Spatial2DView(origin="/01_videos", name="Videos")
    else:
        # Grid: [col][row] -> list of views
        grid = [[[] for _ in range(3)] for _ in range(3)]
        
        for cam_name in sorted(video_caps.keys()):
            col, row = get_layout_pos(cam_name)
            name=cam_name.split(".")[-1]
            view = rrb.Spatial2DView(origin=f"/01_videos/{cam_name}", name="前胸 "+name)
            grid[col][row].append(view)
            
        # Build columns
        cols = []
        # Explicitly iterate 0, 1, 2 to maintain Left-Center-Right order
        for c in range(3):
            col_views = []
            for r in range(3):
                col_views.extend(grid[c][r])
            
            if col_views:
                cols.append(rrb.Vertical(*col_views))
        
        if not cols:
            video_container = rrb.Spatial2DView(origin="/01_videos", name="Videos")
        elif len(cols) == 1:
            video_container = cols[0]
        else:
            video_container = rrb.Horizontal(*cols)

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                video_container,
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="/02_state", name="State"),
                    rrb.Spatial2DView(origin="/03_action", name="Action"),
                ),
            ),
            auto_layout=False,
        )
    )
    
    print(f"Starting replay for episode {episode_idx}...")
    
    frame_idx = 0
    dt = 0.01  # 假设 50Hz，可按需调整
    try:
        while True:
            # 推进所有 replayer
            steps_ok = True
            for name, replayer in replayers.items():
                if not replayer.step():
                    steps_ok = False
                    break
            
            if not steps_ok:
                break
            
            rr.set_time_sequence("frame_index", frame_idx)
            
            # 1. 记录视频（加前缀保证顺序）
            for cam_name, cap in video_caps.items():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 直接写入相机名（父目录名），使用前缀保证顺序
                    rr.log(f"01_videos/{cam_name}/image", rr.Image(frame_rgb))
            
            # 2. 记录 MuJoCo 图像（前缀区分 state/action）
            for name, replayer in replayers.items():
                mj_img = replayer.get_img()
                if mj_img is None:
                    continue
                
                prefix = ""
                if name == "state":
                    prefix = "02_state"
                elif name == "action":
                    prefix = "03_action"
                else:
                    prefix = f"04_{name}"
                    
                rr.log(f"{prefix}/image", rr.Image(mj_img))
                
            
            frame_idx += 1
            
            # 降速以便实时查看，避免瞬间播放完
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        for cap in video_caps.values():
            cap.release()
        for replayer in replayers.values():
            replayer.close_viewer()
        print("Replay finished.")

def main():
    parser = argparse.ArgumentParser(description="Replay Lerobot dataset with MuJoCo and Rerun")
    parser.add_argument("--repo_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--config_name", type=str, required=True, help="Name of the config file (e.g. agilex)")
    parser.add_argument("--data_source", type=str, default="data", choices=["data", "sa_dpp"], help="Data source folder")
    parser.add_argument("--data_type", type=str, default="all", help="Data columns to load (state/action/all)")
    parser.add_argument("--episode_idx", type=int, default=0, help="Episode index")
    
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
