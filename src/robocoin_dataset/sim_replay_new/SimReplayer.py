import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import yaml

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class LerobotSimReplayer:
    def __init__(
        self,
        config_path: str,
        version: str,
        repo_path: str,
        episode_idx: int = 0,
        data_source: str = "data",
        data_type: str = "all",
        headless: bool = False,
        render_width: int = 640,
        render_height: int = 480,
        show_plt: bool = True,
    ):
        # 1. 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.full_config = yaml.safe_load(f)

        # Handle version fallback
        if not version or version not in self.full_config:
            default_version = self.full_config.get("default_version")
            if default_version:
                 print(f"Warning: Version '{version}' not found or empty. Using default_version: '{default_version}'")
                 version = default_version
            else:
                 if version:
                     print(f"Warning: Version '{version}' not found and no default_version provided.")
                 else:
                     print("Warning: No version provided and default_version is empty.")

        # Handle new nested config structure (top-level version key)
        if version and version in self.full_config and "sim" not in self.full_config:
            self.version_config = self.full_config[version]
            self.sim_cfg = self.version_config["sim"]
            self.robot_cfg = self.version_config[version]
        else:
            # Old structure or direct access
            self.sim_cfg = self.full_config.get("sim", {})
            if version and version in self.full_config:
                self.robot_cfg = self.full_config[version]
            else:
                self.robot_cfg = self.full_config
                print("Warning: Using full config as robot_cfg since version is invalid.")


        self.xml_path = self.sim_cfg["xml_path"]
        self.gripper_config = self.sim_cfg["gripper_config"]
        # Normalize camera config: allow either a dict or a list of single-key dicts (YAML style)
        raw_camera = self.sim_cfg.get("camera", {})
        if isinstance(raw_camera, list):
            merged = {}
            for entry in raw_camera:
                if isinstance(entry, dict):
                    merged.update(entry)
            self.camera_config = merged
        elif isinstance(raw_camera, dict):
            self.camera_config = raw_camera
        else:
            self.camera_config = {}

        self.gripper_names = self.gripper_config["gripper_name"]

        # 2. 初始化 MuJoCo
        if not os.path.exists(self.xml_path):
            # Try relative to project root if not absolute
            if not os.path.isabs(self.xml_path):
                 self.xml_path = os.path.join(project_root, self.xml_path)
            if not os.path.exists(self.xml_path):
                raise RuntimeError(f"Module xml file {self.xml_path} not found.")
        self.mjcf_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mjcf_data = mujoco.MjData(self.mjcf_model)

        # 3. 初始化状态
        self.data_source = data_source
        self.data_type = data_type
        self.repo_path = Path(repo_path)
        self.episode_idx = episode_idx
        self.headless = headless
        self.mjcf_viewer = None  # 延迟初始化
        self.mappings = []
        self._build_mappings()

        # 解析 EEF Sites
        self.eef_site_ids = {}
        # 优先从配置读取，如果没有则默认尝试 left_eef_site 和 right_eef_site
        target_eefs = self.sim_cfg.get("eef_name")
        if not target_eefs:
            target_eefs = ["left_eef_site", "right_eef_site"]
            
        for name in target_eefs:
            site_id = mujoco.mj_name2id(self.mjcf_model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id != -1:
                self.eef_site_ids[name] = site_id

        # 4. 【优化】预加载 Fix 函数，避免在 step 中重复 import
        self._init_fix_function()

        # 5. 初始化生成器
        self.frame_generator = self.get_frame_data()
        self.data_frame = {} # Current frame data

        self.render_width = render_width
        self.render_height = render_height

        # 1. 初始化 Renderer
        self.renderer = mujoco.Renderer(
            self.mjcf_model, height=render_height, width=render_width
        )

        # 2. 初始化 Camera 并设置参数
        self.camera = mujoco.MjvCamera()

        # Use .get with defaults and coerce to float to avoid KeyError/TypeError
        lookat_x = float(self.camera_config.get("lookat_x", 0.0))
        lookat_y = float(self.camera_config.get("lookat_y", 0.0))
        lookat_z = float(self.camera_config.get("lookat_z", 0.0))
        self.camera.lookat = np.array([lookat_x, lookat_y, lookat_z])

        self.camera.distance = float(self.camera_config.get("distance", 1.0))
        self.camera.azimuth = float(self.camera_config.get("azimuth", 0.0))
        self.camera.elevation = float(self.camera_config.get("elevation", 0.0))
        # UI Viewer 状态
        self.mjcf_viewer = None

        # 6. Initialize Plotting
        self.show_plt = show_plt
        self.plt_keys = self.full_config.get("show_plt", [])
        if self.plt_keys is None:
            self.plt_keys = []
        elif isinstance(self.plt_keys, str):
            self.plt_keys = [self.plt_keys]
            
        self.plt_initialized = False
        
        # Flatten plt_keys for initial data storage if needed
        # Actual structure is handled in _init_plot
        flat_keys = []
        if self.plt_keys:
            if isinstance(self.plt_keys[0], list):
                for group in self.plt_keys:
                    flat_keys.extend(group)
            elif isinstance(self.plt_keys[0], str):
                flat_keys = self.plt_keys
        
        self.plt_data = {k: [] for k in flat_keys}
        self.plt_frames = []
        self.fig = None
        self.ax = None
        self.axs = None # Initialize axs
        self.lines = {}

    def _init_fix_function(self):
        """预先加载并实例化修正函数，提高运行效率"""
        try:
            func_name = self.sim_cfg.get("function")
            module_path = self.sim_cfg.get("module_file")

            if func_name and module_path:
                # 假设 module_file 格式为 "config.xxx"
                module_name = module_path.replace("config.", "configs.", 1)
                print(f"Loading fix function from: {module_name}.CustomFix.{func_name}")

                module = importlib.import_module(module_name)
                cls = getattr(module, "CustomFix")
                self.fix_instance = cls()  # 实例化
                self.fix_method = getattr(self.fix_instance, func_name)  # 获取方法引用
            else:
                self.fix_method = lambda x: x  # 空函数
        except Exception as e:
            raise RuntimeError(f"Failed to load fix function: {e}")
            
            # print(f"[Warning] Failed to load fix function: {e}")
            self.fix_method = lambda x: x

    def _build_mappings(self):
        """解析 YAML，建立 DataField -> QposIndex 的映射表"""
        print("-" * 30)
        print("正在构建数据映射...")
        for mj_name, data_field in self.sim_cfg["joints"].items():
            joint_id = mujoco.mj_name2id(
                self.mjcf_model, mujoco.mjtObj.mjOBJ_JOINT, mj_name
            )
            if joint_id == -1:
                print(f"[警告] XML中找不到关节: {mj_name}")
                continue
            qpos_adr = self.mjcf_model.jnt_qposadr[joint_id]
            self.mappings.append(
                {"data_key": data_field, "qpos_adr": qpos_adr, "name": mj_name}
            )
        # print("数据映射构建完成:")
        # for item in self.mappings:
        #     print(
        #         f"  数据 '{item['data_key']}' -> 关节 '{item['name']}' (qpos地址: {item['qpos_adr']})"
        #     )
        # print("-" * 30)

    def map_gripper_val(self):
        """映射到夹爪"""
        self.gripper_value_open = self.robot_cfg["gripper_config"]["gripper_value_open"]
        self.gripper_value_close = self.robot_cfg["gripper_config"][
            "gripper_value_close"
        ]
        self.gripper_joint_max = self.gripper_config["gripper_joint_max"]
        self.gripper_joint_min = self.gripper_config["gripper_joint_min"]

        for name in self.gripper_names:
            if name in self.data_frame:
                val = self.data_frame[name]
                if abs(self.gripper_value_open - self.gripper_value_close) < 1e-6:
                    self.data_frame[name] = self.gripper_joint_min
                    continue

                ratio = (val - self.gripper_value_close) / (
                    self.gripper_value_open - self.gripper_value_close
                )
                # 检查范围，超出则报错
                if ratio < 0.0 or ratio > 1.0:
                    # Soft warning instead of hard error to allow partial replays or slightly noisy data
                    # print(f"Warning: 夹爪值 {name}={val} 超出范围 [{self.gripper_value_close}, {self.gripper_value_open}]，归一化比例={ratio}")
                    pass

                sim_val = self.gripper_joint_min + ratio * (
                    self.gripper_joint_max - self.gripper_joint_min
                )
                self.data_frame[name] = sim_val

    def get_frame_data(self):
        """读取文件并建立生成器"""
        sub_dir = "data" if self.data_source == "data" else "state_action_data"
        
        parquet_file_path = (
            self.repo_path
            / sub_dir
            / f"chunk-{self.episode_idx // 1000:03d}"
            / f"episode_{self.episode_idx:06d}.parquet"
        )

        if not parquet_file_path.exists():
            # Fallback for backward compatibility or different structure?
            pass
            
        if not parquet_file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")

        print(f"正在加载数据: {parquet_file_path}")
        df = pd.read_parquet(str(parquet_file_path))

        info_file_path = self.repo_path / "meta" / "info.json" if self.data_source == "data" else self.repo_path / "meta" / "state_action_info.json"
        # info_file_path = self.repo_path / "meta" / "info.json"
        with open(info_file_path, "r") as f:
            info = json.load(f)

        keys_to_load = []
        if self.data_type == "all":
            # Load observation.state and action if they exist in features
            if "observation.state" in info["features"]:
                keys_to_load.append("observation.state")
            if "action" in info["features"]:
                keys_to_load.append("action")
        elif self.data_type == "state":
            keys_to_load.append("observation.state")
        elif self.data_type == "action":
            keys_to_load.append("action")
        
        # Prepare data streams
        data_streams = {}
        length = None
        
        for key in keys_to_load:
            if key in df.columns:
                names = info["features"][key]["names"]
                data_list = df[key].to_list()
                if length is None:
                    length = len(data_list)
                
                data_streams[key] = {"names": names, "data": data_list}
            else:
                print(f"Warning: Key {key} not found in dataframe columns")

        if length is None:
             print("Warning: No data loaded.")
             return

        for i in range(length):
            frame_dict = {}
            for key, stream in data_streams.items():
                vals = stream["data"][i]
                # vals should be a list or array
                frame_dict.update(dict(zip(stream["names"], vals)))
            yield frame_dict

    def get_img(self):
        # 1. 将当前的物理状态(data) 更新到 渲染器场景中
        self.renderer.update_scene(self.mjcf_data, camera=self.camera)

        # 2. 渲染像素
        pixels = self.renderer.render()

        return pixels  # 返回 numpy array (H, W, 3) RGB格式

    def _init_plot(self, key_groups):
        """Initialize matplotlib plot with multiple subplots based on key groups"""
        try:
            import matplotlib.pyplot as plt
            
            # Normalize key_groups to be a list of lists
            # If it's a flat list, wrap it in a single list
            # If it's a string, wrap it in a list of list
            if not key_groups:
                 return

            if isinstance(key_groups, str):
                self.plot_groups = [[key_groups]]
            elif isinstance(key_groups, list):
                if not key_groups:
                    return
                # Check if it is a list of strings (flat) or list of lists
                if isinstance(key_groups[0], str):
                    self.plot_groups = [key_groups]
                else:
                    self.plot_groups = key_groups
            else:
                 print(f"Invalid plot config: {key_groups}")
                 return

            # Flatten all keys for data storage
            self.all_plot_keys = []
            for group in self.plot_groups:
                self.all_plot_keys.extend(group)

            self.plt_data = {k: [] for k in self.all_plot_keys}
            
            plt.ion()
            num_plots = len(self.plot_groups)
            self.fig, self.axs = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
            
            # Ensure axs is iterable even if there is only one plot
            if num_plots == 1:
                self.axs = [self.axs]
            
            self.lines = {}
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
            
            for i, group in enumerate(self.plot_groups):
                ax = self.axs[i]
                for j, k in enumerate(group):
                    color = colors[j % len(colors)]
                    line, = ax.plot([], [], label=k, color=color, linewidth=1.5)
                    self.lines[k] = line
                
                # Add reference lines from config (only to the first plot or if specifically mapped? 
                # For now, let's add to all plots if they seem to be gripper related or just keep simple)
                # Let's only add gripper reference lines if "gripper" is in any of the keys in this group
                if any("gripper" in k for k in group):
                     if "gripper_config" in self.robot_cfg:
                        gc = self.robot_cfg["gripper_config"]
                        open_val = gc.get("gripper_value_open")
                        close_val = gc.get("gripper_value_close")
                        
                        if isinstance(open_val, (int, float)):
                            ax.axhline(y=open_val, color='green', linestyle='--', label=f'Open ({open_val})', alpha=0.6)
                        if isinstance(close_val, (int, float)):
                            ax.axhline(y=close_val, color='red', linestyle='--', label=f'Close ({close_val})', alpha=0.6)

                ax.set_ylabel("Value")
                ax.legend(loc='upper right', fontsize='small')
                ax.grid(True, linestyle=':', alpha=0.6)
            
            self.axs[-1].set_xlabel("Frame")
            plt.tight_layout()
            
            self.plt_initialized = True
        except Exception as e:
            print(f"Failed to initialize plot: {e}\n{traceback.format_exc()}")
            self.show_plt = False

    def _update_plot(self):
        try:
            frame_idx = len(self.plt_frames)
            self.plt_frames.append(frame_idx)
            
            # Update data for all keys
            for k in self.all_plot_keys:
                val = self.data_frame.get(k, 0)
                self.plt_data[k].append(val)
                self.lines[k].set_data(self.plt_frames, self.plt_data[k])
            
            # Rescale all axes
            for ax in self.axs:
                ax.relim()
                ax.autoscale_view()
            
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Plot update error: {e}\n{traceback.format_exc()}")

    def step(self) -> bool:
        try:
            self.data_frame = next(self.frame_generator)
        except StopIteration:
            print(">>> 数据回放结束")
            return False

        # 1. 归一化 gripper 值
        if self.gripper_config.get("gripper", False):  # 安全获取 boolean
            self.map_gripper_val()

        # 2. 执行修正函数 (直接调用预加载的方法)
        self.data_frame = self.fix_method(self.data_frame)

        # 3. 第一次运行时检查字段
        if not hasattr(self, "checked") or not self.checked:
            required_keys = set(item["data_key"] for item in self.mappings)
            if self.gripper_names:
                required_keys.update(self.gripper_names)
            # Only check for keys that are actually expected from the data source
            # If we are only replaying 'state', we shouldn't fail if 'action' keys are missing, 
            # unless those keys are mapped to joints. 
            # But the mappings are usually state->joint. 
            missing = required_keys - set(self.data_frame.keys())
            if missing:
                # print(f"Warning: Missing required data fields for simulation: {missing}. Simulation might look weird.")
                pass
            self.checked = True

        # 4. 赋值给 MuJoCo
        for item in self.mappings:
            key = item["data_key"]
            if key in self.data_frame:
                self.mjcf_data.qpos[item["qpos_adr"]] = self.data_frame[key]

        # 5. 更新物理状态
        mujoco.mj_forward(self.mjcf_model, self.mjcf_data)

        # 6. 获取 EEF 位姿并添加到 data_frame
        if self.eef_site_ids:
             for name, site_id in self.eef_site_ids.items():
                 # 获取位置
                 pos = self.mjcf_data.site_xpos[site_id]
                 # 获取旋转矩阵并转为欧拉角
                 mat = self.mjcf_data.site_xmat[site_id].reshape(3, 3)
                 euler = R.from_matrix(mat).as_euler("xyz", degrees=False)
                 
                 # Debug info
                 # if np.allclose(pos, 0):
                #  print(f"[DEBUG] Frame {self.episode_idx} | {name}: pos={pos}, euler={euler}")

                 base_name = name.replace("_site", "")
                 self.data_frame[f"{base_name}_pos_x_sim"] = pos[0]
                 self.data_frame[f"{base_name}_pos_y_sim"] = pos[1]
                 self.data_frame[f"{base_name}_pos_z_sim"] = pos[2]
                 self.data_frame[f"{base_name}_ori_x_sim"] = euler[0]
                 self.data_frame[f"{base_name}_ori_y_sim"] = euler[1]
                 self.data_frame[f"{base_name}_ori_z_sim"] = euler[2]

        # Update Plot (Moved to end of step to include simulated values)
        if self.show_plt and self.plt_keys:
            if not self.plt_initialized:
                # Trust user config: initialize all keys. Missing keys will be plotted as 0.
                if isinstance(self.plt_keys[0], str):
                    missing = [k for k in self.plt_keys if k not in self.data_frame]
                else:
                    # Flatten keys for check
                    all_keys = []
                    for group in self.plt_keys:
                        all_keys.extend(group)
                    missing = [k for k in all_keys if k not in self.data_frame]

                if missing:
                    print(f"Warning: Plot keys not found in data frame: {missing}. Will plot as 0.")
                self._init_plot(self.plt_keys)
            
            if self.plt_initialized:
                self._update_plot()

        # 如果开启了界面，同步界面
        if self.mjcf_viewer is not None:
            self.mjcf_viewer.sync()

        return True

    def start_viewer(self) -> None:
        """只在非 headless 模式下启动 viewer"""
        if not self.headless and self.mjcf_viewer is None:
            print("启动 MuJoCo Viewer...")
            self.mjcf_viewer = mujoco.viewer.launch_passive(
                self.mjcf_model, self.mjcf_data
            )
            self.mjcf_viewer.sync()

    def sync_viewer(self) -> None:
        """每一帧都需要调用"""
        if self.mjcf_viewer is not None:
            self.mjcf_viewer.sync()

    def close_viewer(self) -> None:
        if self.mjcf_viewer is not None:
            self.mjcf_viewer.close()
            self.mjcf_viewer = None
