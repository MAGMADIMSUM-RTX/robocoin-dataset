import os
import sys
import json
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

import mujoco
from scipy.spatial.transform import Rotation as R
import pandas as pd
import yaml

# ======================
# 单个数据集导出函数
# ======================
def export_single_dataset(dataset_path, data_source="data", version="default_version"):
    try:
        repo_path = Path(dataset_path)
        print(f"\n==================================================")
        print(f"📂 正在处理数据集: {repo_path.name}")
        print(f"==================================================")

        info_path = repo_path / "meta" / "info.json"
        if not info_path.exists():
            print(f"❌ 跳过：无 meta/info.json")
            return

        with open(info_path, "r") as f:
            info = json.load(f)
        total = info.get("total_episodes", 0)
        if total == 0 and "splits" in info:
            total = sum(s.get("num_episodes", 0) for s in info["splits"].values())

        config_state = project_root / "configs/sim_replay/galaxea_state.yml"
        config_action = project_root / "configs/sim_replay/galaxea_action.yml"

        for ep in range(total):
            print(f"\n==> Episode {ep}/{total-1}")

            def compute_eef(data_type):
                try:
                    cfg_path = config_state if data_type == "state" else config_action
                    with open(cfg_path, 'r') as f:
                        full_cfg = yaml.safe_load(f)

                    if version in full_cfg and "sim" not in full_cfg:
                        sim_cfg = full_cfg[version]["sim"]
                    else:
                        sim_cfg = full_cfg.get("sim", {})

                    xml_path = sim_cfg["xml_path"]
                    if not os.path.isabs(xml_path):
                        xml_path = os.path.join(project_root, xml_path)
                    model = mujoco.MjModel.from_xml_path(xml_path)

                    mappings = []
                    for mj_name, data_key in sim_cfg["joints"].items():
                        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
                        if jid == -1: continue
                        qpos_addr = model.jnt_qposadr[jid]
                        mappings.append({"key": data_key, "qpos": qpos_addr})

                    eef_names = sim_cfg.get("eef_name", ["left_eef_site", "right_eef_site"])
                    eef_sites = {}
                    for name in eef_names:
                        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                        if sid != -1:
                            eef_sites[name] = sid

                    sub_dir = "data" if data_source == "data" else "state_action_data"
                    chunk = ep // 1000
                    parquet_path = repo_path / sub_dir / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"
                    df = pd.read_parquet(parquet_path)

                    info_p = repo_path / "meta" / ("info.json" if data_source == "data" else "state_action_info.json")
                    with open(info_p) as f:
                        inf = json.load(f)

                    key = "observation.state" if data_type == "state" else "action"
                    names = inf["features"][key]["names"]
                    arr = np.array(df[key].tolist())

                    eef_out = []
                    for i in range(len(arr)):
                        data = mujoco.MjData(model)
                        mujoco.mj_resetData(model, data)
                        frame = dict(zip(names, arr[i]))

                        for m in mappings:
                            k = m["key"]
                            if k in frame:
                                data.qpos[m["qpos"]] = frame[k]

                        mujoco.mj_forward(model, data)

                        out = {}
                        for name, sid in eef_sites.items():
                            bname = name.replace("_site", "")
                            pos = data.site_xpos[sid]
                            mat = data.site_xmat[sid].reshape(3,3)
                            ori = R.from_matrix(mat).as_euler("xyz")
                            out[bname] = [*pos, *ori]
                        eef_out.append(out)
                    return eef_out

                except Exception as e:
                    print(f"计算 {data_type} 失败: {e}")
                    return []

            eef_state = compute_eef("state")
            eef_action = compute_eef("action")

            save_path = repo_path / f"eef_full_ep{ep:06d}.npz"
            np.savez_compressed(save_path, state=eef_state, action=eef_action)
            print(f"✅ 保存: {save_path.name}")

    except Exception as e:
        print(f"💥 数据集处理失败: {e}")

# ======================
# 批量处理根目录
# ======================
def batch_export_all(root_dir):
    root = Path(root_dir)
    # 自动筛选所有 Galaxea 数据集文件夹
    dataset_dirs = [d for d in root.iterdir() if d.is_dir() and "Galaxea" in d.name]

    print(f"📌 找到 {len(dataset_dirs)} 个数据集")
    for d in dataset_dirs:
        export_single_dataset(d)

    print("\n🎉🎉🎉 全部数据集导出完成！")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="所有数据集的根目录")
    parser.add_argument("--data_source", default="data")
    parser.add_argument("--version", default="default_version")
    args = parser.parse_args()

    batch_export_all(
        root_dir=args.root_dir,
    )
