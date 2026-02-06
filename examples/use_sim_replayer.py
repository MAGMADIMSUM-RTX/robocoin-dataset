"""
示例：如何将 LerobotSimReplayer 作为模块使用

这个文件展示了如何在其他 Python 脚本中导入和使用 LerobotSimReplayer 类。
"""

import os
from pathlib import Path

# 方式 1：在导入 sim_replay 模块之前设置环境变量（推荐）
# 这样可以确保 MuJoCo 使用正确的渲染后端
os.environ['MUJOCO_GL'] = 'egl'  # 无头模式使用 egl

# 现在可以安全地导入模块
from robocoin_pipeline.utils.sim_replay.sim_replay import LerobotSimReplayer


def main():
    # 配置参数
    config_path = "configs/task_params/sim_replay/default.yaml"
    robot_version = "default_version"
    repo_path = "/mnt/nas/synnas/docker2/robocoin-datasets/Split_aloha_basket_storage_banana"
    episode_idx = 0
    
    # 创建 replayer 实例
    replayer = LerobotSimReplayer(
        config_path=config_path,
        robot_version=robot_version,
        repo_path=repo_path,
        episode_idx=episode_idx,
        action=False,  # 使用 observation.state 数据
        headless=True,  # 无头模式
        render_width=640,
        render_height=480,
    )
    
    print(f"开始回放 episode {episode_idx}...")
    
    # 创建保存目录
    save_dir = Path("output_frames_from_module")
    save_dir.mkdir(exist_ok=True)
    
    frame_idx = 0
    try:
        while replayer.step():
            # 每10帧保存一张图片
            if frame_idx % 10 == 0:
                img_rgb = replayer.get_img()
                
                # 保存为 numpy 格式或转换为其他格式
                import cv2
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_dir / f"frame_{frame_idx:04d}.png"), img_bgr)
                print(f"Saved frame {frame_idx}")
            
            frame_idx += 1
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        replayer.close_viewer()
        print(f"完成！共处理 {frame_idx} 帧")
        print(f"图片保存在: {save_dir.absolute()}")


if __name__ == "__main__":
    main()
