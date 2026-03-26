import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ======================
# 单个数据集注入 EEF
# ======================
def inject_eef_to_dataset(repo_path):
    repo_path = Path(repo_path)
    data_dir = repo_path / "data"
    meta_dir = repo_path / "meta"
    meta_path = meta_dir / "info.json"
    stats_path = meta_dir / "episodes_stats.jsonl"

    # 自动备份
    backup_data = repo_path / "old_data"
    backup_meta = repo_path / "old_meta"
    if not backup_data.exists():
        shutil.copytree(data_dir, backup_data)
    if not backup_meta.exists():
        shutil.copytree(meta_dir, backup_meta)

    # EEF 字段名
    leader_cols = [
        "leader_left_arm_eef_x_m.pos",
        "leader_left_arm_eef_y_m.pos",
        "leader_left_arm_eef_z_m.pos",
        "leader_left_arm_rot_euler_x_rad.pos",
        "leader_left_arm_rot_euler_y_rad.pos",
        "leader_left_arm_rot_euler_z_rad.pos",
        "leader_right_arm_eef_x_m.pos",
        "leader_right_arm_eef_y_m.pos",
        "leader_right_arm_eef_z_m.pos",
        "leader_right_arm_rot_euler_x_rad.pos",
        "leader_right_arm_rot_euler_y_rad.pos",
        "leader_right_arm_rot_euler_z_rad.pos",
    ]

    follower_cols = [
        "follower_left_arm_eef_x_m.pos",
        "follower_left_arm_eef_y_m.pos",
        "follower_left_arm_eef_z_m.pos",
        "follower_left_arm_rot_euler_x_rad.pos",
        "follower_left_arm_rot_euler_y_rad.pos",
        "follower_left_arm_rot_euler_z_rad.pos",
        "follower_right_arm_eef_x_m.pos",
        "follower_right_arm_eef_y_m.pos",
        "follower_right_arm_eef_z_m.pos",
        "follower_right_arm_rot_euler_x_rad.pos",
        "follower_right_arm_rot_euler_y_rad.pos",
        "follower_right_arm_rot_euler_z_rad.pos",
    ]

    # 找到所有 EEF npz
    eef_files = sorted(list(repo_path.glob("eef_full_ep*.npz")))
    if len(eef_files) == 0:
        return

    print(f"[{repo_path.name}] 找到 {len(eef_files)} 个 EEF 文件")

    # 读取 stats
    all_stats = []
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            all_stats = [json.loads(line.strip()) for line in f if line.strip()]

    # 逐 EP 处理
    for eef_path in tqdm(eef_files, desc=f"[{repo_path.name}] 注入中"):
        ep_str = eef_path.stem.split("_")[-1].replace("ep", "")
        ep_idx = int(ep_str)
        pq_path = data_dir / f"chunk-000/episode_{ep_str}.parquet"
        if not pq_path.exists():
            continue

        # 加载 EEF
        data = np.load(eef_path, allow_pickle=True)
        state_list = data["state"]
        action_list = data["action"]

        # 转矩阵
        state_eef = np.array([np.concatenate([fr["left_eef"], fr["right_eef"]]) for fr in state_list], dtype=np.float32)
        action_eef = np.array([np.concatenate([fr["left_eef"], fr["right_eef"]]) for fr in action_list], dtype=np.float32)

        # 拼接回 parquet
        df = pd.read_parquet(pq_path)
        new_state = []
        new_action = []
        for i in range(len(df)):
            new_state.append(np.concatenate([df["observation.state"].iloc[i], state_eef[i]]).tolist())
            new_action.append(np.concatenate([df["action"].iloc[i], action_eef[i]]).tolist())

        df["observation.state"] = new_state
        df["action"] = new_action
        df.to_parquet(pq_path, index=False)

        # 更新 stats
        for s in all_stats:
            if s["episode_index"] == ep_idx:
                s["stats"]["observation.state"]["min"] += state_eef.min(axis=0).tolist()
                s["stats"]["observation.state"]["max"] += state_eef.max(axis=0).tolist()
                s["stats"]["observation.state"]["mean"] += state_eef.mean(axis=0).tolist()
                s["stats"]["observation.state"]["std"] += state_eef.std(axis=0).tolist()

                s["stats"]["action"]["min"] += action_eef.min(axis=0).tolist()
                s["stats"]["action"]["max"] += action_eef.max(axis=0).tolist()
                s["stats"]["action"]["mean"] += action_eef.mean(axis=0).tolist()
                s["stats"]["action"]["std"] += action_eef.std(axis=0).tolist()
                break

    # 保存 stats
    with open(stats_path, "w", encoding="utf-8") as f:
        for s in all_stats:
            json.dump(s, f, ensure_ascii=False)
            f.write("\n")

    # 更新 info.json
    with open(meta_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    info["features"]["observation.state"]["names"] += follower_cols
    info["features"]["observation.state"]["shape"][0] += 12
    info["features"]["action"]["names"] += leader_cols
    info["features"]["action"]["shape"][0] += 12

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    # ======================
    # ✅ 删除所有 npz
    # ======================
    for f in eef_files:
        f.unlink()

    print(f"[{repo_path.name}] ✅ 注入完成，已删除 .npz 文件")

# ======================
# 批量处理根目录
# ======================
def batch_inject_all(root_dir):
    root = Path(root_dir)
    dataset_dirs = [d for d in root.iterdir() if d.is_dir() and "Galaxea" in d.name]
    print(f"📌 找到 {len(dataset_dirs)} 个数据集")

    for d in dataset_dirs:
        try:
            inject_eef_to_dataset(d)
        except Exception as e:
            print(f"❌ {d.name} 失败: {e}")

    print("\n🎉🎉🎉 全部批量注入完成！已自动清理 .npz！")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="所有 Galaxea 根目录")
    args = parser.parse_args()
    batch_inject_all(args.root_dir)
