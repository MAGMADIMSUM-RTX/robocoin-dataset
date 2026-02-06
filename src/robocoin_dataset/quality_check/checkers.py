from pathlib import Path

import av
import imagehash
import numpy as np
import cv2

from robocoin_dataset.quality_check.checker_registry import (
    data_video_consistency_checker_registry,
    dataset_data_checker_registry,
    episode_data_checker_registry,
    episode_video_checker_registry,
)


@dataset_data_checker_registry("few_episode_frames")
def detect_short_episodes(
    episode_frame_nums: dict[int, int], ignored_indices: set[int], threshold: int = 150
) -> set[int]:
    """
    检查视频数量是否过小。

    参数:
        video_paths (dict): 键为索引，值为视频路径列表
        ignored_indices (list): 忽略的索引列表

    返回:
        list: 索引列表，表示视频数量小于10的索引
    """
    bad_episodes = set()
    for ep_idx, frame_num in episode_frame_nums.items():
        if ep_idx in ignored_indices:
            continue
        if frame_num < threshold:
            bad_episodes.add(ep_idx)

    return bad_episodes


@dataset_data_checker_registry("too_few_episodes")
def detect_few_episodes(
    episode_frame_nums: dict[int, int], ignored_indices: set[int], threshold: int = 30
) -> set[int]:
    """
    检查视频数量是否过小。

    参数:
        video_paths (dict): 键为索引，值为视频路径列表
        ignored_indices (list): 忽略的索引列表

    返回:
        list: 索引列表，表示视频数量小于10的索引
    """
    bad_episodes = set()
    if len(set(episode_frame_nums.keys()) - (ignored_indices)) < threshold:
        return set(episode_frame_nums.keys())
    return bad_episodes


@dataset_data_checker_registry("abnormal_episode_length")
def detect_frame_num_outliers_mad_idx(
    episode_frame_nums: dict[int, int], ignored_indices: set[int], threshold: float = 2.0
) -> set[int]:
    episode_frame_nums.values()
    bad_episodes = set()
    episode_frame_nums = {
        idx: num for idx, num in episode_frame_nums.items() if idx not in ignored_indices
    }
    data = np.asarray(list(episode_frame_nums.values()))
    if data.size == 0:
        return bad_episodes

    median = np.median(data)
    data = median / data
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    # 避免除零：如果所有值相同，mad=0，则无离群值
    if mad == 0:
        return bad_episodes

    # 计算修正的 Z-score（基于 MAD）
    modified_z_scores = 0.6745 * (data - median) / mad

    # 找出绝对值超过阈值的索引
    outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0]

    bad_episodes_frame_nums = {}

    for idx in outlier_indices:
        bad_episodes_frame_nums[list(episode_frame_nums.keys())[idx]] = list(
            episode_frame_nums.values()
        )[idx]
        bad_episodes.add(list(episode_frame_nums.keys())[idx])
    return bad_episodes


def check_approximately_static_by_std_rms_per_dim(
    data: np.ndarray, threshold: float = 0.01
) -> bool:
    """要求每个维度都近似静止"""
    data = np.asarray(data)
    if data.size == 0 or data.shape[0] <= 1:
        return True

    stds = np.std(data, axis=0)  # shape: (D,)
    rms_vals = np.sqrt(np.mean(data**2, axis=0))  # shape: (D,)

    # 处理某维度全为零的情况
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_stds = np.divide(stds, rms_vals)
        relative_stds[rms_vals == 0] = 0.0  # 全零维度视为静止

    return not np.all(relative_stds < threshold)


def normalize_per_dimension(data: np.ndarray) -> np.ndarray:
    """
    对每个维度独立进行 Z-score 归一化。
    输入: (T, D)
    输出: (T, D)，每列均值≈0，标准差≈1（若原标准差>0）
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # 避免除零：标准差为0的维度保持原值（或设为0）
    normalized = np.zeros_like(data)
    nonzero_std = std > 1e-12
    normalized[:, nonzero_std] = (data[:, nonzero_std] - mean[nonzero_std]) / std[nonzero_std]
    # std=0 的维度已经是常数，归一化后可视为0（不影响静止判断）
    return normalized


def is_window_static(window_data: np.ndarray, threshold: float = 0.01) -> bool:
    if window_data.shape[0] <= 1:
        return True
    stds = np.std(window_data, axis=0)
    rms_vals = np.sqrt(np.mean(window_data**2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_stds = np.divide(stds, rms_vals)
        rel_stds[rms_vals == 0] = 0.0
    count = np.sum(rel_stds > threshold)
    if count >= 1:
        return False
    return True


@episode_data_checker_registry("static_frame_rate")
def count_total_static_frames_rate(
    data: np.ndarray, window_size: int = 5, threshold: float = 0.01
) -> float:
    """
    统计所有被判定为静止的帧的总数量（去重，每帧只算一次）。

    参数:
        data: np.ndarray, shape (T, D)
        window_size: int, 判断静止的窗口大小
        step: int, 滑动步长
        threshold: float, 归一化后的静止阈值

    返回:
        int: 所有静止帧的总数占比
    """
    if data.size == 0:
        return 0
    frame_count = data.shape[0]
    if frame_count == 1:
        return 1  # 单帧视为静止

    # Step 1: 归一化
    data_norm = normalize_per_dimension(data)

    # Step 2: 标记哪些帧属于至少一个静止窗口
    frame_static = np.zeros(frame_count, dtype=bool)

    for i in range(0, frame_count - window_size + 1, 1):
        if is_window_static(data_norm[i : i + window_size], threshold):
            frame_static[i : i + window_size] = True

    # 可选：处理末尾不足 window_size 的帧（保守起见，可跳过）
    # 如果希望更敏感，也可对末尾单独判断（但窗口太小可能不准）

    # Step 3: 统计总静止帧数占比
    total_static = np.sum(frame_static)
    return float(total_static / frame_count)


@episode_data_checker_registry("static_joint")
def detect_static_joint(
    data: np.ndarray, static_joints_percent: float = 0.4, epsilon: float = 1e-3
) -> float:
    """
    检测机械臂数据是否处于“静态”状态（全局判断，非滑动窗口）。

    判断逻辑：
        - 对每个维度（关节），计算其在整个时间序列上的标准差；
        - 如果标准差 < epsilon，则认为该维度“几乎不变”；
        - 若“几乎不变”的维度数 ≥ static_joints，则返回 1.0（静态），否则 0.0。

    参数:
        data: shape (T, D) 的数组，T 为时间步，D 为关节数/维度数
        static_joints: 判定为静态所需的“几乎不变”维度数量阈值
        epsilon: 判定“几乎不变”的标准差容差（默认 1e-2）

    返回:
        float: 1.0 表示静态（有足够多关节未动），0.0 表示非静态
    """
    if data.ndim != 2:
        raise ValueError("Input data must be 2D array of shape (time_steps, dimensions).")

    frame_num, dim = data.shape
    static_joint_threshold = int(dim * static_joints_percent)
    if dim == 0:
        return 1.0  # 无维度，默认静态

    if frame_num <= 1:
        # 只有一帧或空数据：所有维度视为不变
        static_dim_count = dim
    else:
        # 计算每个维度的全局标准差
        stds = np.std(data, axis=0)  # shape (D,)
        is_static_dim = stds < epsilon
        static_dim_count = np.sum(is_static_dim)

    return 1.0 if static_dim_count >= static_joint_threshold else 0.0


def detect_stable_then_jump_frames(
    video_path: str,
    hash_size: int = 16,
    stable_distance_threshold: int = 1,  # 静止期：phash 距离 <= 1
    min_stable_frames: int = 2,  # 静止期最少连续帧数（关键帧对）
) -> tuple[int, int]:
    """
    检测在连续静止关键帧之后，图像跳变的最大 phash 距离。

    算法逻辑：
    1. 提取所有关键帧（I-frame）
    2. 计算相邻关键帧的 phash 汉明距离
    3. 找到所有“连续多个距离 <= 阈值”的静止段
    4. 对每个静止段，查看其**后一个距离**（即跳变）
    5. 返回这些跳变距离中的最大值

    Args:
        video_path: 视频路径
        hash_size: phash 哈希尺寸（推荐 16）
        stable_distance_threshold: 判断为“静止”的最大 phash 距离
        min_stable_frames: 静止段最少包含多少个“小距离”（即连续静止帧对数）

    Returns:
        int: 所有“静止后跳变”中，跳变距离的最大值。如果没有符合条件的跳变，返回 0。
    """
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_count = 1
    except Exception as e:
        raise RuntimeError(f"无法打开视频: {e}")

    keyframe_hashes = []
    total_frame_count = 0
    keyframe_count = 0

    # 提取所有关键帧的 phash
    for packet in container.demux(video=0):
        for frame in packet.decode():
            total_frame_count += 1

            if not (frame.key_frame or frame.pict_type == "I"):
                continue

            try:
                img = frame.to_image()
                h = imagehash.phash(img, hash_size=hash_size)
                keyframe_hashes.append(h)
                keyframe_count += 1
            except Exception as e:
                print(f"处理第 {total_frame_count - 1} 帧时出错: {e}")
                continue

    container.close()

    if len(keyframe_hashes) < 2:
        return 0  # 帧数不足

    # 计算相邻关键帧的 phash 距离
    distances = [
        keyframe_hashes[i] - keyframe_hashes[i - 1] for i in range(1, len(keyframe_hashes))
    ]

    max_jump = 0  # 存储最大跳变距离
    frame_idx = 0

    # 遍历所有可能的跳变点（从第 min_stable_frames 个距离开始）
    for i in range(min_stable_frames, len(distances)):
        # 检查前 min_stable_frames 个距离是否都 <= 阈值（静止段）
        is_stable = all(
            d <= stable_distance_threshold for d in distances[i - min_stable_frames : i]
        )

        if is_stable:
            # 当前距离就是“跳变”
            jump_distance = distances[i]
            if jump_distance > max_jump:
                max_jump = jump_distance
                frame_idx = i

    return max_jump, frame_idx


@episode_video_checker_registry("max_frame_stable_then_jump_rate")
def dectect_max_frame_stable_then_jump(video_paths: list[str | Path]) -> float:
    max_jump = 0
    for video_path in video_paths:
        if not Path(video_path).exists() or not Path(video_path).is_file():
            return 1
        current_jump, _ = detect_stable_then_jump_frames(str(video_path))
        max_jump = max(max_jump, current_jump)
    return max_jump / 100


@episode_video_checker_registry("max_frame_jump_dist")
def detect_max_frame_jump_dist(
    video_paths: list[str | Path], max_dist_threshold: int = 50
) -> float:
    max_jump = 0
    for video_path in video_paths:
        if not Path(video_path).exists() or not Path(video_path).is_file():
            return 1
        current_jump, _ = detect_stable_then_jump_frames(
            str(video_path), stable_distance_threshold=1, min_stable_frames=0
        )
        max_jump = max(max_jump, current_jump)

    return 1 if max_jump > max_dist_threshold else 0


# @episode_video_checker_registry("frame_jump")
# def detect_max_frame_jump_and_static(
#     video_paths: list[str | Path],
#     hash_size: int = 32,
#     static_threshold: int = 1,
#     max_phash_distance_threshold: int = 200,
#     max_static_frames_count_threshold: int = 5,
#     max_frames: int = None,  # 可选：限制处理帧数（调试用）
# ) -> float:
#     """
#     使用 PyAV 和 pHash 分析视频帧变化。

#     参数:
#         video_path (str): 视频文件路径
#         hash_size (int): pHash 尺寸（默认 8 → 64位哈希）
#         static_threshold (int): 判定静止的最大汉明距离（默认 2）
#         max_frames (int or None): 最大处理帧数（None 表示全部）

#     返回:
#         dict: {
#             'max_hamming_distance': int,
#             'max_static_frames': int,
#             'total_frames': int
#         }
#     """
#     for video_path in video_paths:
#         if not Path(video_path).exists() or not Path(video_path).is_file():
#             return 1
#         container = av.open(video_path)
#         stream = container.streams.video[0]

#         prev_hash = None
#         max_hamming = 0
#         current_static_run = 0
#         max_static_run = 0
#         frame_count = 0
#         max_static_frame_idx = 0

#         try:
#             for frame in container.decode(stream):
#                 if max_frames and frame_count >= max_frames:
#                     break

#                 # 转为 RGB numpy 数组
#                 img_array = frame.to_rgb().to_ndarray()
#                 pil_img = Image.fromarray(img_array)

#                 # 计算 pHash
#                 curr_hash = imagehash.phash(pil_img, hash_size=hash_size)

#                 if prev_hash is not None:
#                     hamming_dist = curr_hash - prev_hash  # imagehash 重载了减号为汉明距离
#                     max_hamming = max(max_hamming, hamming_dist)

#                     if hamming_dist <= static_threshold:
#                         current_static_run += 1
#                         if current_static_run > max_static_run:
#                             max_static_run = current_static_run
#                             max_static_frame_idx = frame_count

#                         max_static_run = max(max_static_run, current_static_run)
#                     else:
#                         current_static_run = 0  # 重置静止计数

#                 prev_hash = curr_hash
#                 frame_count += 1

#         finally:
#             container.close()

#         # 注意：连续静止帧数 = 连续“间隔”数 + 1
#         # 例如：3 帧完全相同 → 有 2 个“静止间隔”，但实际静止帧数为 3
#         # 我们这里统计的是“连续静止的帧总数”，所以需要 +1
#         max_static_frames = (
#             max_static_run + 1 if max_static_run > 0 else 1 if frame_count > 0 else 0
#         )

#         if (
#             max_hamming > max_phash_distance_threshold
#             or max_static_frames > max_static_frames_count_threshold
#         ):
#             return 1

#     return 0


# @episode_video_checker_registry("frame_jump")
# def detect_max_frame_jump_and_static(
#     video_paths: list[str | Path],
#     hash_size: int = 16,
#     jump_threshold: int = 30,  # 新增：判定跳帧所需的最小汉明距离
#     max_frames: int = None,
# ) -> float:
#     """
#     改进版：使用三帧窗口检测“跳帧”（孤立大幅变化帧）+ 静止帧检测。

#     跳帧判定条件：
#         当前帧与前一帧、后一帧的 pHash 汉明距离均 > jump_threshold，
#         且前一帧与后一帧之间的距离 <= jump_threshold（可选增强条件）。

#     返回:
#         1 表示检测到异常（跳帧或静止帧超标），0 表示正常。
#     """
#     for video_path in video_paths:
#         if not Path(video_path).exists() or not Path(video_path).is_file():
#             return 1

#         container = av.open(video_path)
#         stream = container.streams.video[0]

#         frame_hashes = []  # 存储所有帧的 pHash（或最多 max_frames 帧）
#         frame_count = 0

#         try:
#             for frame in container.decode(stream):
#                 if max_frames and frame_count >= max_frames:
#                     break
#                 img_array = frame.to_rgb().to_ndarray()
#                 pil_img = Image.fromarray(img_array)
#                 phash = imagehash.phash(pil_img, hash_size=hash_size)
#                 frame_hashes.append(phash)
#                 frame_count += 1
#         finally:
#             container.close()

#         if frame_count == 0:
#             return 1  # 无帧，视为异常

#         # === 跳帧检测：三帧窗口 ===
#         jump_detected = False
#         n = len(frame_hashes)
#         max_jump_dist = 0
#         for i in range(2, n - 1):  # 跳过首尾帧
#             prev_hash_2 = frame_hashes[i - 2]
#             prev_hash = frame_hashes[i - 1]
#             curr_hash = frame_hashes[i]
#             next_hash = frame_hashes[i + 1]

#             diff_value_pre = prev_hash_2 - prev_hash
#             diff_value_curr = prev_hash - curr_hash
#             diff_value_next = curr_hash - next_hash

#             jump_value_pre = abs(diff_value_pre - diff_value_curr)
#             jump_value_next = abs(diff_value_curr - diff_value_next)

#             jump_dist = abs(jump_value_pre - jump_value_next)
#             if jump_dist > max_jump_dist:
#                 max_jump_dist = jump_dist


#         if max_jump_dist > jump_threshold:
#             return 1

#         if jump_detected:
#             return 1

#     return 0


@data_video_consistency_checker_registry("LengthConsistencyChecker")
def decect_inconsistent_length(video_paths: list[str | Path], frame_num: int) -> bool:
    for video_path in video_paths:
        if not Path(video_path).exists() or not Path(video_path).is_file():
            return True
        container = av.open(video_path)
        if container.streams.video[0].frames != frame_num:
            return True

    return False

# 保留原有的辅助函数 + 适配算子格式
def compute_mean_colors(image: np.ndarray) -> dict:
    """计算图像各通道的平均颜色（BGR）"""
    return {
        "B": np.mean(image[:, :, 0]),
        "G": np.mean(image[:, :, 1]),
        "R": np.mean(image[:, :, 2]),
    }

@episode_video_checker_registry("video_color_shift_detection")  # 新算子注册名
def detect_video_color_shift(
    video_paths: list[str | Path], 
    color_diff_threshold: float = 30.0
) -> float:
    """
    检测视频帧间的色差问题（替换原有模糊检测算子）
    参数:
        video_paths: 视频路径列表
        color_diff_threshold: 颜色差异阈值（0-255，默认30）
    返回:
        float: 色差帧占比（1=全色差，0=无色差）→ 用于后续得分计算
    """
    total_color_shift_frames = 0
    total_valid_frames = 0

    for video_path in video_paths:
        video_path = Path(video_path)
        if not video_path.exists():
            return 1.0  # 路径不存在视为全色差
        
        try:
            # 强制软件解码（解决AV1解码问题）
            container = av.open(str(video_path))
            video_stream = next(s for s in container.streams.video)
            video_stream.codec_context.options = {"hwaccel": "none"}
        except Exception as e:
            print(f"打开视频{video_path}失败: {e}")
            return 1.0  # 解码失败视为全色差

        prev_mean_colors = None
        frame_idx = 0

        try:
            for frame in container.decode(video_stream):
                frame_idx += 1
                # 转换为BGR格式（适配OpenCV）
                image = frame.to_ndarray(format="bgr24")
                current_mean_colors = compute_mean_colors(image)

                if prev_mean_colors is None:
                    prev_mean_colors = current_mean_colors
                    continue

                # 计算帧间最大颜色差异
                color_diff = max(
                    abs(current_mean_colors["B"] - prev_mean_colors["B"]),
                    abs(current_mean_colors["G"] - prev_mean_colors["G"]),
                    abs(current_mean_colors["R"] - prev_mean_colors["R"]),
                )

                # 超过阈值视为色差帧
                if color_diff > color_diff_threshold:
                    total_color_shift_frames += 1
                total_valid_frames += 1
                prev_mean_colors = current_mean_colors

        except Exception as e:
            print(f"处理视频{video_path}帧失败: {e}")
            container.close()
            return 1.0  # 处理失败视为全色差

        container.close()

    # 无有效帧视为全色差
    if total_valid_frames == 0:
        return 1.0
    
    # 返回色差帧占比（1=全色差，0=无色差）
    color_shift_ratio = total_color_shift_frames / total_valid_frames
    return color_shift_ratio
