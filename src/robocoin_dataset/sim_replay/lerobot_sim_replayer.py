import json
import time
from collections.abc import Iterator
from pathlib import Path

# 设置matplotlib后端，优先TkAgg，失败则Agg
try:
    import matplotlib

    matplotlib.use("TkAgg")
except Exception:
    import matplotlib

    matplotlib.use("Agg")

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as R

from robocoin_dataset.sim_replay.configs.lerobot_sim_replay_config import (
    LerobotSimReplayConfig,
)
from robocoin_dataset.utils.le_path import get_parquet_files


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int) -> None:
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


class LerobotSimReplayer:
    def __init__(
        self,
        replay_config: LerobotSimReplayConfig,
        repo_path: str | Path,
        replay_source: str = "sa_dpp",
    ) -> None:
        self.mjcf_file_path = Path(replay_config.mjcf_path).expanduser().resolve()
        self.repo_path = Path(repo_path).expanduser().absolute()
        self.get_mjcf_gripper_joint_data = replay_config.get_mjcf_gripper_joint_data
        self.mjcf_site_names = replay_config.mjcf_site_names

        # 获取 gripper site 名称（如果配置中有定义）
        self.gripper_site_names = []
        if hasattr(replay_config, "left_gripper_mjcf_site_name") and hasattr(
            replay_config, "right_gripper_mjcf_site_name"
        ):
            self.gripper_site_names = [
                replay_config.left_gripper_mjcf_site_name,
                replay_config.right_gripper_mjcf_site_name,
            ]

        if not self.mjcf_file_path.exists():
            raise FileNotFoundError(f"MJCF file not found: {self.mjcf_file_path}")
        if not self.repo_path.exists():
            raise ValueError(f"Lerobot Repo Path does not exist: {self.repo_path}")

        try:
            self.mjcf_model = mujoco.MjModel.from_xml_path(str(self.mjcf_file_path))
            self.mjcf_data = mujoco.MjData(self.mjcf_model)
            self.mjcf_site_ids = [self.mjcf_model.site(name).id for name in self.mjcf_site_names]
            # 获取 gripper site IDs（如果有定义）
            self.gripper_site_ids = []
            if self.gripper_site_names:
                self.gripper_site_ids = [
                    self.mjcf_model.site(name).id for name in self.gripper_site_names
                ]
            self.mjcf_viewer = None
        except Exception as e:
            raise Exception(f"Error loading MJCF model: {e}")

        self.state_arm_joint_mjcf_names = replay_config.state_arm_joint_mjcf_names
        self.state_arm_joint_lerobot_names = replay_config.state_arm_joint_lerobot_names

        self.action_arm_joint_mjcf_names = replay_config.action_arm_joint_mjcf_names
        self.action_arm_joint_lerobot_names = replay_config.action_arm_joint_lerobot_names

        self.state_gripper_joint_mjcf_names = replay_config.state_gripper_joint_mjcf_names
        self.state_gripper_lerobot_names = replay_config.state_gripper_lerobot_names

        self.action_gripper_joint_mjcf_names = replay_config.action_gripper_joint_mjcf_names
        self.action_gripper_lerobot_names = replay_config.action_gripper_lerobot_names
        mjcf_joint_names = self._get_mjcf_joint_names()

        exist_flags = [
            mjcf_joint_name in mjcf_joint_names
            for mjcf_joint_name in self.state_arm_joint_mjcf_names
        ]
        non_exist_names = [
            self.state_arm_joint_mjcf_names[i] for i, flag in enumerate(exist_flags) if not flag
        ]
        if not all(exist_flags):
            raise Exception(
                f"Given State MJCF arm joint name: {non_exist_names} not found in MJCF file, MJCF joints: {mjcf_joint_names}"
            )

        exist_flags = [
            mjcf_joint_name in mjcf_joint_names
            for mjcf_joint_name in self.action_arm_joint_mjcf_names
        ]
        non_exist_names = [
            self.action_arm_joint_mjcf_names[i] for i, flag in enumerate(exist_flags) if not flag
        ]
        if not all(exist_flags):
            raise Exception(
                f"Given Action MJCF arm joint name: {non_exist_names} not found in MJCF file, MJCF joints: {mjcf_joint_names}"
            )

        exist_flags = [
            mjcf_joint_name in mjcf_joint_names
            for mjcf_joint_name in self.state_gripper_joint_mjcf_names
        ]
        non_exist_names = [
            self.state_gripper_joint_mjcf_names[i] for i, flag in enumerate(exist_flags) if not flag
        ]
        if not all(exist_flags):
            raise Exception(
                f"Given State MJCF gripper joint name: {non_exist_names} not found in MJCF file, MJCF joints: {mjcf_joint_names}"
            )

        exist_flags = [
            mjcf_joint_name in mjcf_joint_names
            for mjcf_joint_name in self.action_gripper_joint_mjcf_names
        ]
        non_exist_names = [
            self.action_gripper_joint_mjcf_names[i]
            for i, flag in enumerate(exist_flags)
            if not flag
        ]
        if not all(exist_flags):
            raise Exception(
                f"Given Action MJCF gripper joint name: {non_exist_names} not found in MJCF file, MJCF joints: {mjcf_joint_names}"
            )

        self.state_arm_joint_mjcf_addrs = [
            self._get_mjcf_joint_addr(mjcf_joint_name)
            for mjcf_joint_name in self.state_arm_joint_mjcf_names
        ]

        self.action_arm_joint_mjcf_addrs = [
            self._get_mjcf_joint_addr(mjcf_joint_name)
            for mjcf_joint_name in self.action_arm_joint_mjcf_names
        ]

        self.state_gripper_joint_mjcf_addrs = [
            self._get_mjcf_joint_addr(mjcf_joint_name)
            for mjcf_joint_name in self.state_gripper_joint_mjcf_names
        ]

        self.action_gripper_joint_mjcf_addrs = [
            self._get_mjcf_joint_addr(mjcf_joint_name)
            for mjcf_joint_name in self.action_gripper_joint_mjcf_names
        ]
        if replay_source == "sa_dpp":
            meta_file_path = self.repo_path / "meta/state_action_info.json"
        elif replay_source == "data":
            meta_file_path = self.repo_path / "meta/info.json"
        if not meta_file_path.exists():
            raise Exception(f"Meta file not found: {meta_file_path}")

        print(f"DEBUG: Reading meta file: {meta_file_path}")
        with open(meta_file_path) as f:
            json_data = json.load(f)
            features = json_data.get("features", None)
            if features is None:
                raise Exception("No features found in info.json")
            print(f"DEBUG: Features found: {list(features.keys())}")
            if "gripper_open_scale_state" in features:
                 print(f"DEBUG: gripper_open_scale_state: {features['gripper_open_scale_state']}")

            value = features.get("observation.state", None)
            if value is None:
                raise Exception("No states found in info.json")
            meta_state_names = value.get("names", None)
            if meta_state_names is None:
                raise Exception("No state names found in info.json")

            value = features.get("action", None)
            if value is None:
                raise Exception("No actions found in info.json")

            meta_action_names = value.get("names", None)
            if meta_action_names is None:
                raise Exception("No action names found in info.json")

            value = features.get("gripper_open_scale_state", None)
            if value:
                meta_gripper_state_names = value.get("names", [])
            else:
                meta_gripper_state_names = []

            value = features.get("gripper_open_scale_action", None)
            if value:
                meta_gripper_action_names = value.get("names", [])
            else:
                meta_gripper_action_names = []

        exist_flags = [
            state_name in meta_state_names for state_name in self.state_arm_joint_lerobot_names
        ]
        non_exist_names = [
            self.state_arm_joint_lerobot_names[i] for i, flag in enumerate(exist_flags) if not flag
        ]
        if not all(exist_flags):
            raise Exception(
                f"Given State Lerobot arm joint name: {non_exist_names} not found in info.json, State names: {meta_state_names}"
            )

        exist_flags = [
            action_name in meta_action_names for action_name in self.action_arm_joint_lerobot_names
        ]
        non_exist_names = [
            self.action_arm_joint_lerobot_names[i] for i, flag in enumerate(exist_flags) if not flag
        ]
        if not all(exist_flags):
            raise Exception(
                f"Given Action Lerobot arm joint name: {non_exist_names} not found in info.json, Action names: {meta_action_names}"
            )

        exist_flags = [
            state_name in meta_gripper_state_names for state_name in self.state_gripper_lerobot_names
        ]
        non_exist_names = [
            self.state_gripper_lerobot_names[i] for i, flag in enumerate(exist_flags) if not flag
        ]
        if not all(exist_flags):
            raise Exception(
                f"Given State Lerobot gripper joint name: {non_exist_names} not found in info.json, Gripper State names: {meta_gripper_state_names}"
            )

        exist_flags = [
            action_name in meta_gripper_action_names for action_name in self.action_gripper_lerobot_names
        ]
        non_exist_names = [
            self.action_gripper_lerobot_names[i] for i, flag in enumerate(exist_flags) if not flag
        ]
        if not all(exist_flags):
            raise Exception(
                f"Given Action Lerobot gripper joint name: {non_exist_names} not found in info.json, Gripper Action names: {meta_gripper_action_names}"
            )

        self.state_arm_joint_lerobot_ids = [
            meta_state_names.index(name) for name in self.state_arm_joint_lerobot_names
        ]

        self.action_arm_joint_lerobot_ids = [
            meta_action_names.index(name) for name in self.action_arm_joint_lerobot_names
        ]

        self.state_gripper_lerobot_ids = [
            meta_gripper_state_names.index(name) for name in self.state_gripper_lerobot_names
        ]

        self.action_gripper_lerobot_ids = [
            meta_gripper_action_names.index(name) for name in self.action_gripper_lerobot_names
        ]

        self.parquet_files = get_parquet_files(self.repo_path, "state_action")

    def _get_mjcf_joint_names(self) -> set[str]:
        return {self.mjcf_model.joint(i).name for i in range(self.mjcf_model.njnt)}

    def _get_mjcf_joint_addr(self, name: str) -> int:
        return self.mjcf_model.jnt_qposadr[self.mjcf_model.joint(name).id]

    def replay_episode_background(
        self,
        episode_index: int,
        is_state: bool = True,
        is_sa_dpp: bool = False,
        is_eef: bool = True,
    ) -> list[np.ndarray]:
        # 根据 is_sa_dpp 参数决定从哪个路径读取 parquet 文件
        if is_sa_dpp:
            # 从原始 data 目录读取
            parquet_file_path = (
                self.repo_path
                / "data"
                / f"chunk-{episode_index // 1000:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
        else:
            # 从 state_action_data 目录读取（默认）
            if episode_index >= len(self.parquet_files):
                raise ValueError(f"episode_index {episode_index} out of range")
            parquet_file_path = self.parquet_files[episode_index]

        # print(f"***[后台] 正在读取末端信息， episode 文件: {parquet_file_path}")
        if not parquet_file_path.exists():
            raise Exception(f"Parquet file not found: {parquet_file_path}")
        df = pd.read_parquet(str(parquet_file_path))
        results: list[np.ndarray] = []

        # 重置 MuJoCo 数据状态，避免前一个 episode 的状态影响当前计算
        mujoco.mj_resetData(self.mjcf_model, self.mjcf_data)

        if is_state:
            mjcf_arm_joint_addrs = self.state_arm_joint_mjcf_addrs
            mjcf_gripper_joint_addrs = self.state_gripper_joint_mjcf_addrs
            lerbot_arm_joint_ids = self.state_arm_joint_lerobot_ids
            leroot_gripper_ids = self.state_gripper_lerobot_ids
            data = df["observation.state"].to_list()
            if "gripper_open_scale_state" in df.columns:
                gripper_data = df["gripper_open_scale_state"].to_list()
            else:
                gripper_data = data
        else:
            mjcf_arm_joint_addrs = self.action_arm_joint_mjcf_addrs
            mjcf_gripper_joint_addrs = self.action_gripper_joint_mjcf_addrs
            lerbot_arm_joint_ids = self.action_arm_joint_lerobot_ids
            leroot_gripper_ids = self.action_gripper_lerobot_ids
            data = df["action"].to_list()
            if "gripper_open_scale_action" in df.columns:
                gripper_data = df["gripper_open_scale_action"].to_list()
            else:
                gripper_data = data

        # 根据 is_eef 参数决定使用哪个 site
        site_ids_to_use = (
            self.mjcf_site_ids
            if is_eef
            else (self.gripper_site_ids if self.gripper_site_ids else self.mjcf_site_ids)
        )

        try:
            for i in range(len(data)):
                lerobot_arm_joint_values = data[i][lerbot_arm_joint_ids]
                lerobot_gripper_data = gripper_data[i][leroot_gripper_ids]
                for mjcf_addr, lerobot_value in zip(mjcf_arm_joint_addrs, lerobot_arm_joint_values):
                    self.mjcf_data.qpos[mjcf_addr] = lerobot_value

                mjcf_gripper_joint_values = self.get_mjcf_gripper_joint_data(
                    lerobot_gripper_data=lerobot_gripper_data
                )
                for mjcf_addr, mjcf_data in zip(
                    mjcf_gripper_joint_addrs, mjcf_gripper_joint_values
                ):
                    self.mjcf_data.qpos[mjcf_addr] = mjcf_data

                mujoco.mj_forward(self.mjcf_model, self.mjcf_data)
                # 收集所有site的EEF数据
                frame_eef_results = []
                for site_id in site_ids_to_use:
                    site_pos = self.mjcf_data.site_xpos[site_id]
                    site_rot = self.mjcf_data.site_xmat[site_id]
                    try:
                        site_rot_euler = R.from_matrix(site_rot.reshape(3, 3)).as_euler(
                            "xyz", degrees=False
                        )
                    except:
                        print(
                            f"[ERROR] 获取 site_rot_euler 失败，site_rot: {site_rot}, episode_index: {episode_index}"
                        )
                        raise
                    # 拼接当前site的 EEF 位置和姿态
                    frame_eef_results.extend(site_pos)
                    frame_eef_results.extend(site_rot_euler)

                # 在所有site数据之后添加夹爪数据
                frame_eef_results.extend(lerobot_gripper_data)
                # 转换为numpy数组并添加到结果中
                results.append(np.array(frame_eef_results))
            return results

        finally:
            # 确保在任何情况下都能正确关闭界面
            pass

    def replay_episode(
        self,
        episode_index: int,
        is_state: bool = True,
        # sleep_time_ms: int = 0,
        replay_source: str = "sa_dpp",
        enable_gripper_plot: bool = False,
        gripper_plot_callback=None,
        target_fps: int = 30,
        is_eef: bool = True,
    ) -> None:
        if replay_source == "sa_dpp":
            parquet_file_path = (
                self.repo_path
                / "state_action_data"
                / f"chunk-{episode_index // 1000:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
        elif replay_source == "data":
            parquet_file_path = (
                self.repo_path
                / "data"
                / f"chunk-{episode_index // 1000:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
        print(f"***[界面] 正在播放 episode 文件: {parquet_file_path}")
        if not parquet_file_path.exists():
            raise Exception(f"Parquet file not found: {parquet_file_path}")
        df = pd.read_parquet(str(parquet_file_path))

        if is_state:
            mjcf_arm_joint_addrs = self.state_arm_joint_mjcf_addrs
            mjcf_gripper_joint_addrs = self.state_gripper_joint_mjcf_addrs
            lerbot_arm_joint_ids = self.state_arm_joint_lerobot_ids
            leroot_gripper_ids = self.state_gripper_lerobot_ids
            data = df["observation.state"].to_list()
            if "gripper_open_scale_state" in df.columns:
                gripper_data = df["gripper_open_scale_state"].to_list()
            else:
                gripper_data = data
        else:
            mjcf_arm_joint_addrs = self.action_arm_joint_mjcf_addrs
            mjcf_gripper_joint_addrs = self.action_gripper_joint_mjcf_addrs
            lerbot_arm_joint_ids = self.action_arm_joint_lerobot_ids
            leroot_gripper_ids = self.action_gripper_lerobot_ids
            data = df["action"].to_list()
            if "gripper_open_scale_action" in df.columns:
                gripper_data = df["gripper_open_scale_action"].to_list()
            else:
                gripper_data = data

        gripper_history = []
        fig, ax, lines = None, None, []

        # target frame duration (seconds) - enforce a strict playback rate
        if target_fps <= 0:
            target_fps = 30
        frame_duration = 1.0 / float(target_fps)

        # 启动 MuJoCo 界面
        self.start_viewer()

        # 用于记录所有帧的EEF y轴坐标
        left_y_history = []
        right_y_history = []

        # 根据 is_eef 参数决定使用哪个 site
        site_ids_to_use = (
            self.mjcf_site_ids
            if is_eef
            else (self.gripper_site_ids if self.gripper_site_ids else self.mjcf_site_ids)
        )

        try:
            for i in range(len(data)):
                frame_start = time.perf_counter()
                lerobot_arm_joint_values = data[i][lerbot_arm_joint_ids]
                lerobot_gripper_data = gripper_data[i][leroot_gripper_ids]
                for mjcf_addr, lerobot_value in zip(mjcf_arm_joint_addrs, lerobot_arm_joint_values):
                    self.mjcf_data.qpos[mjcf_addr] = lerobot_value

                mjcf_gripper_joint_values = self.get_mjcf_gripper_joint_data(
                    lerobot_gripper_data=lerobot_gripper_data
                )
                for mjcf_addr, mjcf_data in zip(
                    mjcf_gripper_joint_addrs, mjcf_gripper_joint_values
                ):
                    self.mjcf_data.qpos[mjcf_addr] = mjcf_data

                mujoco.mj_forward(self.mjcf_model, self.mjcf_data)

                # 收集所有EEF位置用于计算距离
                eef_positions = []
                for site_id in site_ids_to_use:
                    eef_results = []
                    site_pos = self.mjcf_data.site_xpos[site_id]
                    eef_positions.append(site_pos.copy())
                    site_rot = self.mjcf_data.site_xmat[site_id]
                    site_rot_euler = R.from_matrix(site_rot.reshape(3, 3)).as_euler(
                        "xyz", degrees=False
                    )
                    eef_results = np.concatenate([eef_results, site_pos, site_rot_euler], axis=0)

                # 如果是双臂机器人（有2个EEF），记录y轴坐标并打印当前帧的y轴距离
                # if len(eef_positions) == 2:
                #     left_eef_pos = eef_positions[0]
                #     right_eef_pos = eef_positions[1]
                #     left_y_history.append(left_eef_pos[1])
                #     right_y_history.append(right_eef_pos[1])
                #     eef_distance = np.abs(left_eef_pos[1] - right_eef_pos[1])
                #     print(f"[Frame {i:04d}] EEF Y-axis Distance: {eef_distance:.4f}m | Left: [{left_eef_pos[0]:.3f}, {left_eef_pos[1]:.3f}, {left_eef_pos[2]:.3f}] | Right: [{right_eef_pos[0]:.3f}, {right_eef_pos[1]:.3f}, {right_eef_pos[2]:.3f}]")

                # 实时gripper曲线刷新
                # Only plot gripper data when:
                # - plotting enabled by caller,
                # - a callback is provided,
                # - the dataset actually contains gripper fields (leroot_gripper_ids),
                # - AND the replay config declares gripper lerobot names (not necessarily MJCF names,
                #   as some configs like yinhe have gripper data but don't map to MJCF joints)
                if (
                    enable_gripper_plot
                    and gripper_plot_callback is not None
                    and len(leroot_gripper_ids) > 0
                    and len(self.state_gripper_lerobot_names) > 0
                ):
                    gripper_history.append(list(lerobot_gripper_data))
                    gripper_plot_callback(gripper_history, ax, lines)
                    # 使用matplotlib进行短暂暂停以更新图表
                    import matplotlib.pyplot as plt

                    plt.pause(0.001)

                self._sync_viewer()
                elapsed = time.perf_counter() - frame_start
                remaining = frame_duration - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            # 播放完成后，打印EEF y轴距离统计信息（计算平均y坐标的差）
            if len(left_y_history) > 0 and len(right_y_history) > 0:
                # 计算每帧左右臂的y轴距离（用于min、max、std统计）
                frame_distances = np.abs(np.array(left_y_history) - np.array(right_y_history))

                # 计算平均y坐标的差（这是主要指标）
                mean_left_y = np.mean(left_y_history)
                mean_right_y = np.mean(right_y_history)
                mean_y_distance = np.abs(mean_left_y - mean_right_y)

                # 其他统计指标
                min_distance = np.min(frame_distances)
                max_distance = np.max(frame_distances)
                std_distance = np.std(frame_distances)

                print(f"\n{'=' * 80}")
                print("[EEF Y-axis Distance Statistics]")
                print(f"  Total Frames: {len(left_y_history)}")
                print(f"  Mean Y-axis Distance (avg of means): {mean_y_distance:.4f}m")
                print(f"  Min Y-axis Distance:  {min_distance:.4f}m")
                print(f"  Max Y-axis Distance:  {max_distance:.4f}m")
                print(f"  Std Y-axis Distance:  {std_distance:.4f}m")
                print(f"{'=' * 80}\n")

        finally:
            # 确保在任何情况下都能正确关闭界面
            pass

    def close_all_interfaces(self):
        """同时关闭 MuJoCo 界面和可视化图表"""
        print("[界面] 正在关闭 MuJoCo 界面和可视化图表...")

        # 关闭 matplotlib 图表
        try:
            import matplotlib.pyplot as plt

            plt.ioff()
            plt.close("all")
        except Exception as e:
            print(f"[界面] 关闭可视化图表时出现错误: {e}")

        # 关闭 MuJoCo 界面
        self.close_viewer()

        print("[界面] 所有界面已关闭")

    def start_viewer(self) -> None:
        if self.mjcf_viewer is None:
            self.mjcf_viewer = mujoco.viewer.launch_passive(self.mjcf_model, self.mjcf_data)
            self.mjcf_viewer.sync()

    def _sync_viewer(self) -> None:
        if self.mjcf_viewer is not None:
            self.mjcf_viewer.sync()

    def close_viewer(self) -> None:
        if self.mjcf_viewer is not None:
            # print("[界面] 正在关闭 MuJoCo 界面...")
            self.mjcf_viewer.close()
            # print("[界面] MuJoCo 界面已关闭")
        self.mjcf_viewer = None
