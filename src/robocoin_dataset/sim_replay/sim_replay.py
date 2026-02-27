import importlib
import json
import logging
from pathlib import Path

# For gripper value visualization
import asyncio
import matplotlib.pyplot as plt
import yaml
from sqlalchemy.orm import Session
from sqlalchemy.sql import and_, or_

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    DatasetDB,
    TaskStatus,
)
from robocoin_dataset.distribution_computation.constant import (
    DATASET_UUID,
    DEVICE_MODEL,
    ERR_MSG,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
    TASK_FAILED,
    TASK_RESULT_CONTENT,
    TASK_ID,
)
from robocoin_dataset.distribution_computation.task_client import TaskClient
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.format_converter.tolerobot.constant import (
    LEFORMAT_PATH,
)
from robocoin_dataset.sim_replay.configs.lerobot_sim_replay_config import LerobotSimReplayConfig
from robocoin_dataset.sim_replay.lerobot_sim_replayer import LerobotSimReplayer

SIM_REPLAY_CONFIG_MODULE_PATH = "sim_replay_config_module_path"
SIM_REPLAY_CONFIG_CLASS_NAME = "sim_replay_config_class"
SIM_REPLAY_LOG_DIR = "sim_replay_log_dir"
SIM_REPLAY_LOG_NAME = "sim_replay_log_name"
DEVICE_MODEL_VERSION = "device_model_version"


def _get_config(class_module: str, class_name: str) -> type[LerobotSimReplayConfig]:
    config_class = importlib.import_module(class_module).__getattribute__(class_name)
    return config_class()


def _get_config_class_module(
    replay_config_classes: dict[tuple[str, str], type[LerobotSimReplayConfig]],
    device_model: str,
    device_model_version: str,
) -> tuple[str | None, str | None]:
    class_module, class_name = replay_config_classes.get(
        (device_model, device_model_version), (None, None)
    )
    return class_module, class_name


def _get_config_classes_path_dict(
    file_path: str | Path,
) -> dict[tuple[str, str], tuple[str, str]]:
    if not file_path.exists():
        raise FileNotFoundError(f"processor_class_config_path {file_path} not exists")
    with open(file_path) as f:
        yaml_dict = yaml.safe_load(f)

    replay_config_classes_dict = {}
    for device_model_name, configs in yaml_dict.items():
        for config in configs:
            device_version = config["version"]
            class_module_path = config.get("mujoco_sim_replay_config_module", None)
            if class_module_path is None:
                continue
            class_name = config.get("mujoco_sim_replay_config_class", None)
            if class_name is None:
                continue
            replay_config_classes_dict[(device_model_name, device_version)] = (
                class_module_path,
                class_name,
            )

    return replay_config_classes_dict


def _sync_sim_replay_tasks(
    session: Session, 
    device_model: str | None = None, 
    device_model_version: str | None = None,
    target_dataset_uuid: str | None = None  # 新增
) -> None:
    # 基础查询条件
    query = session.query(DatasetDB).filter(
        and_(
            # 必要前提：convert必须成功
            DatasetDB.sa_dpp_status == TaskStatus.COMPLETED,
            # 两个触发分支
            or_(
                # 分支1: 正在排队
                DatasetDB.sim_replay_status == TaskStatus.PENDING,
                # 分支2: 已完成但版本过期
                and_(
                    DatasetDB.sim_replay_status == TaskStatus.COMPLETED,
                    DatasetDB.sim_replay_version_ps < DatasetDB.sa_dpp_version,
                ),
            ),
        )
    )
    
    # 新增：指定UUID时只处理该数据集
    if target_dataset_uuid:
        query = query.filter(DatasetDB.dataset_uuid == target_dataset_uuid)
    elif device_model:
        query = query.filter(DatasetDB.device_model == device_model)
        if device_model_version:
            query = query.filter(DatasetDB.device_model_version == device_model_version)

    items = query.all()

    if not items:
        return
    for item in items:
        item.sim_replay_status = TaskStatus.PENDING
        item.sim_replay_version = 0 if item.sim_replay_version is None else item.sim_replay_version + 1
        item.sim_replay_version_ps = item.sa_dpp_version

    session.commit()


def _gen_one_sim_replay_task(
    session: Session, 
    device_model: str, 
    device_model_version: str | None = None,
    target_dataset_uuid: str | None = None  # 新增
) -> tuple[str | None, str | None, str | None, str | None]:
    # 基础查询条件
    query = session.query(DatasetDB).filter(
        DatasetDB.sim_replay_status == TaskStatus.PENDING,
    )
    
    # 新增：指定UUID时只查询该数据集
    if target_dataset_uuid:
        query = query.filter(DatasetDB.dataset_uuid == target_dataset_uuid)
    elif device_model:
        query = query.filter(DatasetDB.device_model == device_model)
        if device_model_version:
            query = query.filter(DatasetDB.device_model_version == device_model_version)
    
    item = query.first()
    if not item:
        return None, None, None, None
    
    # 检查 qced_repo_gen_path 是否存在
    if not item.qced_repo_gen_path or not Path(item.qced_repo_gen_path).exists():
        raise FileNotFoundError(f"qced_repo_gen_path not found for dataset {item.dataset_uuid}: {item.qced_repo_gen_path}")
    
    item.sim_replay_status = TaskStatus.PROCESSING
    session.commit()
    return (
        item.dataset_uuid,
        item.qced_repo_gen_path,  # 修改：替换为 qced_repo_gen_path
        item.device_model,
        item.device_model_version,
    )


def plot_gripper_values(gripper_values, title="Gripper Values"):
    """
    静态方法：绘制gripper值的图表，支持gripper_left和gripper_right各自包含1-5个子数据

    Args:
        gripper_values: gripper值的数组或列表
        title: 图表标题
    """
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.array(gripper_values)
    if arr.ndim == 1:
        arr = arr[:, None]

    # 假设有两个gripper：gripper_left 和 gripper_right
    gripper_names = ["gripper_left", "gripper_right"]
    gripper_colors = ["green", "red"]  # gripper_left用绿色，gripper_right用红色

    # 检测每个gripper的子数据数量
    total_grippers = arr.shape[1]
    grippers_per_side = (
        total_grippers // 2 if total_grippers % 2 == 0 else (total_grippers + 1) // 2
    )

    # print(f"[静态Gripper可视化] 检测到总共 {total_grippers} 个gripper数据")
    # print(f"[静态Gripper可视化] 每个gripper预计包含 {grippers_per_side} 个子数据")

    # 为每个gripper类型创建一个窗口
    for gripper_idx, gripper_name in enumerate(gripper_names):
        if gripper_idx * grippers_per_side >= total_grippers:
            break

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"{gripper_name} Values")

        # 在这个窗口中绘制该gripper的所有子数据，使用相同颜色
        start_idx = gripper_idx * grippers_per_side
        end_idx = min(start_idx + grippers_per_side, total_grippers)

        base_color = gripper_colors[gripper_idx]  # 使用gripper对应的基础颜色

        for sub_idx in range(start_idx, end_idx):
            local_sub_idx = sub_idx - start_idx
            label = f"{gripper_name}_{local_sub_idx}"
            ax.plot(arr[:, sub_idx], color=base_color, label=label)

        ax.set_xlabel("Step")
        ax.set_ylabel("Gripper Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()

        # print(f"[静态Gripper可视化] 已创建 {gripper_name} 窗口，包含 {end_idx - start_idx} 条曲线")


def _sim_replay_dataset(
    repo_path: str | Path,
    sim_replay_config: LerobotSimReplayConfig,
    replay_source: str = "sa_dpp",
    episode_idx: int = 0,
    is_eef: bool = False
) -> None:
    if sim_replay_config is None:
        raise ValueError("sim_replay_config_class is None")
    

    simulator = LerobotSimReplayer(sim_replay_config, repo_path, replay_source=replay_source)

    def gripper_plot_callback(gripper_history, ax, lines) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        arr = np.array(gripper_history)
        if arr.ndim == 1:
            arr = arr[:, None]

        # 检查是否已经创建了窗口（使用全局状态避免重复创建）
        if not hasattr(gripper_plot_callback, "figs"):
            gripper_plot_callback.figs = []
            gripper_plot_callback.axes = []
            gripper_plot_callback.lines = []

            # 假设有两个gripper：gripper_left 和 gripper_right
            gripper_names = ["gripper_left", "gripper_right"]
            # 使用6种分明的颜色，适用于所有维度
            distinct_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

            # 检测每个gripper的子数据数量
            total_grippers = arr.shape[1]
            grippers_per_side = (
                total_grippers // 2 if total_grippers % 2 == 0 else (total_grippers + 1) // 2
            )

            print(f"[Gripper可视化] 每个gripper预计包含 {grippers_per_side} 个子数据")

            # 为每个gripper类型创建一个窗口
            for gripper_idx, gripper_name in enumerate(gripper_names):
                if gripper_idx * grippers_per_side >= total_grippers:
                    break

                fig, ax_sub = plt.subplots(figsize=(10, 6))
                fig.suptitle(f"{gripper_name} Values")

                # 在这个窗口中绘制该gripper的所有子数据，使用不同颜色
                gripper_lines = []
                start_idx = gripper_idx * grippers_per_side
                end_idx = min(start_idx + grippers_per_side, total_grippers)

                for sub_idx in range(start_idx, end_idx):
                    local_sub_idx = sub_idx - start_idx
                    label = f"{gripper_name}_{local_sub_idx}"
                    # 使用6种分明的颜色，循环使用
                    color = distinct_colors[local_sub_idx % len(distinct_colors)]
                    (line,) = ax_sub.plot(arr[:, sub_idx], color=color, label=label)
                    gripper_lines.append(line)

                ax_sub.set_xlabel("Step")
                ax_sub.set_ylabel("Gripper Value")
                # 确定该窗口的 y 轴上下界：
                # 1. 如果 has_gripper=True 且配置了固定范围，使用固定范围（y轴不动态变化）
                # 2. 如果 has_gripper=False 或未配置固定范围，使用动态范围（y轴动态变化）
                use_dynamic_ylim = True  # 默认使用动态y轴
                try:
                    # 检查 has_gripper 属性
                    has_gripper_attr = getattr(sim_replay_config, "has_gripper", None)
                    
                    # 如果 has_gripper 明确为 True 且配置了固定范围，则使用固定范围
                    if (
                        has_gripper_attr is True
                        and hasattr(sim_replay_config, "gripper_value_close")
                        and hasattr(sim_replay_config, "gripper_value_open")
                    ):
                        ymin = float(getattr(sim_replay_config, "gripper_value_close"))
                        ymax = float(getattr(sim_replay_config, "gripper_value_open"))
                        # 保证 ymin <= ymax
                        if ymin > ymax:
                            ymin, ymax = ymax, ymin
                        use_dynamic_ylim = False  # 使用固定y轴
                    else:
                        # 动态范围：根据当前数据计算 min/max 并添加边距（增加到20%）
                        col = arr[:, start_idx:end_idx]
                        vmin = float(col.min())
                        vmax = float(col.max())
                        if vmin == vmax:
                            # 单一值时给一个小范围
                            margin = max(abs(vmin) * 0.2, 0.02)
                        else:
                            margin = (vmax - vmin) * 0.2
                        ymin = vmin - margin
                        ymax = vmax + margin
                        use_dynamic_ylim = True  # 使用动态y轴
                except Exception:
                    ymin, ymax = -0.1, 1.1
                    use_dynamic_ylim = False
                    
                ax_sub.set_ylim(ymin, ymax)
                # 记录初始 ylim 和是否使用动态y轴
                if not hasattr(gripper_plot_callback, "ylims"):
                    gripper_plot_callback.ylims = []
                    gripper_plot_callback.use_dynamic_ylims = []
                gripper_plot_callback.ylims.append((ymin, ymax))
                gripper_plot_callback.use_dynamic_ylims.append(use_dynamic_ylim)
                ax_sub.legend()
                ax_sub.grid(True, alpha=0.3)

                gripper_plot_callback.figs.append(fig)
                gripper_plot_callback.axes.append(ax_sub)
                gripper_plot_callback.lines.extend(gripper_lines)

                plt.show(block=False)

                print(
                    f"[Gripper可视化] 已创建 {gripper_name} 窗口，包含 {len(gripper_lines)} 条曲线"
                )

            # 将创建的lines返回给调用者
            lines.extend(gripper_plot_callback.lines)
        else:
            # 更新现有的线条数据
            for i, line in enumerate(gripper_plot_callback.lines):
                if i < arr.shape[1]:
                    line.set_ydata(arr[:, i])
                    line.set_xdata(np.arange(arr.shape[0]))

            # 对每个 axes 重新计算数据范围
            for idx, ax_sub in enumerate(gripper_plot_callback.axes):
                # 检查是否使用动态y轴
                use_dynamic = False
                if hasattr(gripper_plot_callback, "use_dynamic_ylims") and idx < len(
                    gripper_plot_callback.use_dynamic_ylims
                ):
                    use_dynamic = gripper_plot_callback.use_dynamic_ylims[idx]
                
                # 先重新计算数据边界
                try:
                    ax_sub.relim()
                    
                    if use_dynamic:
                        # 动态y轴：手动计算当前数据的y轴范围
                        # 获取该窗口对应的数据列范围
                        gripper_names = ["gripper_left", "gripper_right"]
                        total_grippers = arr.shape[1]
                        grippers_per_side = (
                            total_grippers // 2 if total_grippers % 2 == 0 else (total_grippers + 1) // 2
                        )
                        
                        start_idx = idx * grippers_per_side
                        end_idx = min(start_idx + grippers_per_side, total_grippers)
                        
                        # 计算这些列的数据范围
                        col = arr[:, start_idx:end_idx]
                        vmin = float(col.min())
                        vmax = float(col.max())
                        
                        # 添加边距（增加到20%以提供更多视觉冗余）
                        if vmin == vmax:
                            margin = max(abs(vmin) * 0.2, 0.02)
                        else:
                            margin = (vmax - vmin) * 0.2
                        
                        ymin = vmin - margin
                        ymax = vmax + margin
                        
                        # 设置新的y轴范围
                        ax_sub.set_ylim(ymin, ymax)
                        # 只自动缩放x轴
                        ax_sub.autoscale_view(scalex=True, scaley=False)
                    else:
                        # 固定y轴：仅对 x 轴自动缩放，保持 y 轴不变
                        ax_sub.autoscale_view(scalex=True, scaley=False)
                        # 恢复初始 ylim 以保证固定上下界
                        if hasattr(gripper_plot_callback, "ylims") and idx < len(
                            gripper_plot_callback.ylims
                        ):
                            ymin, ymax = gripper_plot_callback.ylims[idx]
                            ax_sub.set_ylim(ymin, ymax)
                except Exception as e:
                    # 如果计算失败，忽略并继续（保持之前的显示）
                    import traceback
                    print(f"[Gripper可视化] 更新axes {idx} 时出错: {e}")
                    traceback.print_exc()
                    pass

            # 请求界面重绘
            for fig in gripper_plot_callback.figs:
                fig.canvas.draw_idle()

    # 读取 meta/info.json 中的 fps（优先），若不存在则回退到 30
    fps_from_meta = None
    try:
        meta_info_file = Path(repo_path) / "meta" / "info.json"
        if meta_info_file.exists():
            with open(meta_info_file) as f:
                info = json.load(f)
                fps_from_meta = info.get("fps", None)
                print(f"[数据集回放] 从 meta/info.json 中读取到 fps: {fps_from_meta}")
    except Exception:
        fps_from_meta = None

    try:
        # 启动界面（MuJoCo窗口）
        simulator.start_viewer()
        print(f"[数据集回放] 开始回放数据集数据，数据集地址为: {repo_path}")
        print(f"[数据集回放] MuJoCo 窗口已启动")
        
        # 预创建图表窗口（读取第一帧数据来初始化窗口）
        print("[数据集回放] 正在初始化可视化图表窗口...")
        
        # 检查是否有gripper字段：只要 state_gripper_lerobot_names 有字段就启用绘图
        mjcf_gripper_names = getattr(sim_replay_config, "state_gripper_lerobot_names", [])
        cfg_has_gripper = len(mjcf_gripper_names) > 0
        
        if cfg_has_gripper:
            has_gripper_attr = getattr(sim_replay_config, "has_gripper", None)
            print(f"[数据集回放] 检测到gripper字段，数量: {len(mjcf_gripper_names)}, has_gripper={has_gripper_attr}")
        else:
            print("[数据集回放] 未检测到gripper字段，跳过图表窗口创建")
        
        try:
            # 只有当配置启用了gripper时才尝试创建图表
            if cfg_has_gripper:
                # 读取第一个 parquet 文件来获取 gripper 数据的维度
                import pandas as pd
                if replay_source == "sa_dpp":
                    first_parquet = Path(repo_path) / "state_action_data" / "chunk-000" / "episode_000000.parquet"
                elif replay_source == "data":
                    first_parquet = Path(repo_path) / "data" / "chunk-000" / "episode_000000.parquet"
                else:
                    raise ValueError(f"未知的 replay_source: {replay_source}")
                print(f"[数据集回放] 读取数据文件: {first_parquet}")
                if first_parquet.exists():
                    df = pd.read_parquet(str(first_parquet))
                    print(f"[数据集回放] 数据文件包含 {len(df)} 帧")
                    print(f"[数据集回放] Gripper ID 数量: {len(simulator.state_gripper_lerobot_ids)}")
                    if len(df) > 0 and len(simulator.state_gripper_lerobot_ids) > 0:
                        # 使用第一帧数据初始化图表窗口
                        first_state = df["observation.state"].iloc[0]
                        first_gripper_data = first_state[simulator.state_gripper_lerobot_ids]
                        print(f"[数据集回放] Gripper 数据维度: {len(first_gripper_data)}")
                        # 调用callback创建窗口（传入单帧数据）
                        ax_dummy = None
                        lines_dummy = []
                        gripper_plot_callback([list(first_gripper_data)], ax_dummy, lines_dummy)
                        print("[数据集回放] 图表窗口已创建")
                        # 启用交互模式，让图表窗口保持响应
                        plt.ion()
                        # 强制刷新显示
                        plt.pause(0.1)
                    else:
                        print("[数据集回放] 数据文件为空或无gripper数据")
                else:
                    print(f"[数据集回放] 数据文件不存在: {first_parquet}")
        except Exception as e:
            print(f"[数据集回放] 初始化图表窗口时出现警告: {e}")
            import traceback
            traceback.print_exc()
        
        print("[数据集回放] 正在准备 replay...，请在mujoco中调整好观察视角")
        input("[数据集回放] 请按回车键开始 state replay...")

        # State replay 循环
        while True:
            try:
                print("[State Replay] 开始播放状态数据...")
                simulator.replay_episode(
                    episode_idx,
                    is_state=True,
                    replay_source=replay_source,
                    enable_gripper_plot=cfg_has_gripper,
                    gripper_plot_callback=gripper_plot_callback if cfg_has_gripper else None,
                    target_fps=int(fps_from_meta) if fps_from_meta else 30,
                    is_eef=is_eef,
                )
                print("[State Replay] 状态数据播放完成")
            except Exception as e:
                print(f"[State Replay] 播放过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
                raise

            choice = (
                input(
                    "[State Replay] 按c键确认结果正确，按r键重新播放，按e键输入错误信息 (r/e/c): "
                )
                .strip()
                .lower()
            )

            if choice == "c":
                print("[State Replay] 用户确认状态结果正确，继续action replay")
                break
            if choice == "r":
                print("[State Replay] 正在准备重新播放状态数据...")
                continue
            if choice == "e":
                error_message = input("[State Replay] 请输入state replay错误信息: ").strip()
                print(f"[State Replay] 收到错误信息: {error_message}")
                confirm = input("[State Replay] 确认错误请按回车键，修改错误信息请按其他键后回车: ")
                if confirm == "":
                    raise RuntimeError(f"state replay发生错误: {error_message}")
            else:
                print("[State Replay] 无效输入，请输入 c、r 或 e")

        input("[Action Replay] 请按回车键开始 action replay...")

        # Action replay 循环
        while True:
            try:
                print("[Action Replay] 开始播放动作数据...")
                simulator.replay_episode(
                    episode_idx,
                    is_state=False,
                    replay_source=replay_source,
                    enable_gripper_plot=cfg_has_gripper,
                    gripper_plot_callback=gripper_plot_callback if cfg_has_gripper else None,
                    target_fps=int(fps_from_meta) if fps_from_meta else 30,
                    is_eef=is_eef,
                )
                print("[Action Replay] 动作数据播放完成")
            except Exception as e:
                print(f"[Action Replay] 播放过程中发生错误: {e}")
                import traceback
                traceback.print_exc()
                raise

            choice = (
                input(
                    "[Action Replay] 按c键确认结果正确，按r键重新播放，按e键输入错误信息 (r/e/c): "
                )
                .strip()
                .lower()
            )

            if choice == "c":
                print("[Action Replay] 用户确认动作结果正确，回放完成")
                break
            if choice == "r":
                print("[Action Replay] 正在准备重新播放动作数据...")
                continue
            if choice == "e":
                error_message = input("[Action Replay] 请输入action replay错误信息: ").strip()
                print(f"[Action Replay] 收到错误信息: {error_message}")
                confirm = input(
                    "[Action Replay] 确认错误请按回车键，修改错误信息请按其他键后回车: "
                )
                if confirm == "":
                    raise RuntimeError(f"action replay发生错误: {error_message}")
            else:
                print("[Action Replay] 无效输入，请输入 c、r 或 e")

        print("[数据集回放] 所有回放完成")

    finally:
        # 确保界面被关闭（同时关闭MuJoCo窗口和图表窗口）
        print("[数据集回放] 正在关闭所有界面...")
        
        # 关闭matplotlib图表窗口
        try:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.close("all")
            # 清理gripper_plot_callback中的状态
            if hasattr(gripper_plot_callback, "figs"):
                delattr(gripper_plot_callback, "figs")
            if hasattr(gripper_plot_callback, "axes"):
                delattr(gripper_plot_callback, "axes")
            if hasattr(gripper_plot_callback, "lines"):
                delattr(gripper_plot_callback, "lines")
            if hasattr(gripper_plot_callback, "ylims"):
                delattr(gripper_plot_callback, "ylims")
            if hasattr(gripper_plot_callback, "use_dynamic_ylims"):
                delattr(gripper_plot_callback, "use_dynamic_ylims")
            print("[数据集回放] 图表窗口已关闭")
        except Exception as e:
            print(f"[数据集回放] 关闭图表窗口时出现错误: {e}")
        
        # 关闭MuJoCo窗口
        simulator.close_viewer()
        print("[数据集回放] MuJoCo界面已关闭")
        print("[数据集回放] 所有界面已关闭")


class SimReplay:
    def __init__(
        self,
        db_file_path: str | Path,
        sim_replay_config_path: str | Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.logger = logger or logging.getLogger(__name__)
        self.sim_replay_config_classes_path_dict = _get_config_classes_path_dict(
            file_path=sim_replay_config_path
        )

    def sim_replay_datasets(self, device_model: str, device_model_version: str = "", target_dataset_uuid: str | None = None) -> None:  # 新增参数
        with self.db.with_session() as session:
            _sync_sim_replay_tasks(session, device_model, device_model_version=device_model_version, target_dataset_uuid=target_dataset_uuid)  # 传参
            dataset_uuid, convert_path, device_model, device_model_version = (
                _gen_one_sim_replay_task(
                    session=session,
                    device_model=device_model,
                    device_model_version=device_model_version,
                    target_dataset_uuid=target_dataset_uuid,  # 传参
                )
            )

        if dataset_uuid is None:
            return

        try:
            class_module, class_name = _get_config_class_module(
                self.sim_replay_config_classes_path_dict,
                device_model=device_model,
                device_model_version=device_model_version,
            )
            replay_config = _get_config(class_module=class_module, class_name=class_name)
            if not isinstance(replay_config, LerobotSimReplayConfig):
                raise RuntimeError(
                    f"数据集回放配置类 {replay_config} 不是 LerobotSimReplayConfig 子类"
                )
            _sim_replay_dataset(repo_path=convert_path, sim_replay_config=replay_config)
            with self.db.with_session() as session:
                item = (
                    session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                )
                if not item:
                    raise RuntimeError(f"数据集 {dataset_uuid} 不存在")

                item.sim_replay_status = TaskStatus.COMPLETED
                session.commit()

        except Exception as e:
            self.logger.error(e)
            with self.db.with_session() as session:
                item = (
                    session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                )
                if not item:
                    raise RuntimeError(f"数据集 {dataset_uuid} 不存在")
                item.sim_replay_status = TaskStatus.FAILED
                item.sim_replay_error_msg = str(e)
                session.commit()


class SimReplayServer(TaskServer):
    def __init__(
        self,
        db_file_path: str | Path,
        sim_replay_config_path: str | Path,
        host: str = "0.0.0.0",
        port: int = 8767,
        heartbeat_interval: float = 30.0,  # 服务端每30秒发一次 ping
        device_model: str = "",
        device_model_version: str = "",
        target_dataset_uuid: str | None = None,  # 新增
        timeout: float = 15.0,  # 等待 pong 超过15秒则断开
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            logger=logger,
            host=host,
            port=port,
            heartbeat_interval=heartbeat_interval,
            timeout=timeout,
        )
        db_file_path = Path(db_file_path).expanduser().absolute()
        self.device_model = device_model
        self.device_model_version = device_model_version
        self.target_dataset_uuid = target_dataset_uuid  # 新增

        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.logger = logger or logging.getLogger(__name__)
        sim_replay_config_path = Path(sim_replay_config_path).expanduser().absolute()
        if not sim_replay_config_path.exists():
            raise FileNotFoundError(
                f"sim_replay_config_classes_file_path {sim_replay_config_path} not exists"
            )

        self.sim_replay_classes_config_dict = _get_config_classes_path_dict(sim_replay_config_path)

        self.logger.info(
            f"Simulation Replay Server started, "
            f"device_model={self.device_model}, "
            f"device_model_version={self.device_model_version}, "
            f"target_dataset_uuid={self.target_dataset_uuid}"  # 新增日志
        )

    def get_task_category(self) -> str:
        return "simulation_replay"

    def generate_task_content(self) -> dict | None:
        with self.db.with_session() as session:
            query = session.query(DatasetDB).filter(
                and_(
                    # 必要前提：convert必须成功
                    DatasetDB.sa_dpp_status == TaskStatus.COMPLETED,
                    # 两个触发分支
                    or_(
                        # 分支1: 正在排队
                        DatasetDB.sim_replay_status == TaskStatus.PENDING,
                        # 分支2: 已完成但版本过期
                        and_(
                            DatasetDB.sim_replay_status == TaskStatus.COMPLETED,
                            DatasetDB.sim_replay_version_ps < DatasetDB.sa_dpp_version,
                        ),
                    ),
                )
            )
            # 新增：指定UUID时只处理该数据集
            if self.target_dataset_uuid:
                query = query.filter(DatasetDB.dataset_uuid == self.target_dataset_uuid)
            elif self.device_model is not None:
                query = query.filter(
                    DatasetDB.device_model == self.device_model,
                )

            if self.device_model_version is not None and not self.target_dataset_uuid:  # 指定UUID时忽略版本过滤
                query = query.filter(DatasetDB.device_model_version == self.device_model_version)

            item = query.first()

            if not item:
                return None

            item.sim_replay_status = TaskStatus.PROCESSING
            item.sim_replay_version = item.sim_replay_version + 1
            item.sim_replay_version_ps = item.sa_dpp_version

            session.commit()
            sim_replay_config_module_path, sim_replay_config_class_name = (
                self.sim_replay_classes_config_dict.get(
                    (item.device_model, item.device_model_version),
                    (None, None),
                )
            )
            if sim_replay_config_module_path is None or sim_replay_config_class_name is None:
                raise ValueError(
                    f"No processor config found for device model {item.device_model} and version {item.device_model_version}"
                )

            return {
                DATASET_UUID: item.dataset_uuid,
                LEFORMAT_PATH: item.qced_repo_gen_path,  # 修改：替换为 qced_repo_gen_path
                DEVICE_MODEL: item.device_model,
                DEVICE_MODEL_VERSION: item.device_model_version,
                SIM_REPLAY_CONFIG_MODULE_PATH: sim_replay_config_module_path,
                SIM_REPLAY_CONFIG_CLASS_NAME: sim_replay_config_class_name,
            }

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        ds_uuid = task_content.get(DATASET_UUID)

        task_status = task_result_content.get(TASK_RESULT_STATUS)
        task_status_msg = task_result_content.get(ERR_MSG)

        convert_status = TaskStatus.COMPLETED if task_status == TASK_SUCCESS else TaskStatus.FAILED

        # 🆕 合并为单个session，保证原子性
        with self.db.with_session() as session:
            # 查询 device_model_version
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
            if item is None:
                self.logger.error(f"Dataset {ds_uuid} not found in dataset DB.")

            # 在同一个session中更新转换状态
            item.sim_replay_status = convert_status
            item.sim_replay_error_msg = task_status_msg
            session.commit()
            self.logger.info(
                f"Upsert {item.qced_repo_gen_path} sim replay status to {convert_status}, "  # 修改：替换为 qced_repo_gen_path
                f"update_message: {task_status_msg}"
            )


class SimReplayClient(TaskClient):
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
        return "simulation_replay"

    def generate_task_request_desc(self) -> dict:
        """客户端可自定义任务请求参数"""
        return {}

    def _sync_process_task(self, task_content: dict) -> dict:
        repo_path = task_content.get(LEFORMAT_PATH)  # 现在是 qced_repo_gen_path
        sim_replay_config_module_path = task_content.get(SIM_REPLAY_CONFIG_MODULE_PATH)
        sim_replay_config_class_name = task_content.get(SIM_REPLAY_CONFIG_CLASS_NAME)

        sim_replay_config_class = importlib.import_module(
            sim_replay_config_module_path
        ).__getattribute__(sim_replay_config_class_name)

        _sim_replay_dataset(
            repo_path=repo_path,
            sim_replay_config=sim_replay_config_class(),
        )
        print(f"sim replay dataset {repo_path} succeeded")

        return {}

    async def process_task(self, task_data: dict) -> dict:
        """覆盖父类的 process_task 方法，只保存错误消息，不包含 traceback"""
        loop = asyncio.get_event_loop()
        task_id = task_data.get(TASK_ID)
        try:
            task_result_content = await loop.run_in_executor(
                None, self._sync_process_task, task_data
            )
            return {TASK_RESULT_STATUS: TASK_SUCCESS, TASK_RESULT_CONTENT: task_result_content}
        except Exception as e:
            # 只保存错误消息本身，不包含 traceback，与 SimReplay 的行为一致
            error_msg = str(e)
            if self.logger:
                self.logger.error(f"Task {task_id} failed: {error_msg}")
            return {
                TASK_RESULT_STATUS: TASK_FAILED,
                ERR_MSG: error_msg,
                TASK_RESULT_CONTENT: {},
            }
