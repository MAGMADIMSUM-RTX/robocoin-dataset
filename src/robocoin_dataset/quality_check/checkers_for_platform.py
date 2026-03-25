from pathlib import Path
import av
import imagehash
import math
import numpy as np
import cv2
import platform
import json
from robocoin_dataset.quality_check.checker_registry import (
    data_video_consistency_checker_registry,
    dataset_data_checker_registry,
    episode_data_checker_registry,
    episode_video_checker_registry,
)

import time
from functools import wraps

# 全局缓存：key=(episode_idx, video_path), value=解码结果
VIDEO_DECODE_CACHE = {}
# 记录当前活跃的Episode，避免跨Episode缓存污染
ACTIVE_EPISODE_IDX = None

# 全局标记：是否有可用的GPU（只检测一次，避免重复检测）
HAS_CUDA_GPU = None

def timer(label=""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            res = func(*args, **kwargs)
            cost = time.time() - t0
            print(f"[TIMER] {label} {func.__name__} | cost: {cost:.3f}s")
            return res
        return wrapper
    return decorator

def check_cuda_availability() -> bool:
    """
    检测系统是否有可用的NVIDIA GPU + CUDA环境
    返回: True=有可用GPU，False=无GPU/环境不可用
    """
    global HAS_CUDA_GPU
    if HAS_CUDA_GPU is not None:
        return HAS_CUDA_GPU
    
    # 快速检测（避免耗时）
    try:
        # 1. 检测系统是否为Linux（Windows需调整）
        if platform.system() != "Linux":
            HAS_CUDA_GPU = False
            return False
        
        # 2. 检测nvidia-smi是否可用
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0 or not result.stdout.strip():
            HAS_CUDA_GPU = False
            return False
        
        # 3. 检测PyAV是否支持CUDA解码
        test_video_path = Path(__file__).parent / "test.mp4"
        container = av.open(str(test_video_path)) if test_video_path.exists() else None
        if container:
            video_stream = container.streams.video[0]
            video_stream.codec_context.options = {"hwaccel": "cuda"}
            container.close()
        
        HAS_CUDA_GPU = True
        print("[GPU检测] 检测到可用的NVIDIA GPU，将启用CUDA加速解码")
    except Exception as e:
        HAS_CUDA_GPU = False
        print(f"[GPU检测] 无可用GPU/CUDA环境，将使用CPU解码 | 原因：{str(e)}")
    
    return HAS_CUDA_GPU

# @timer("DECODE | 自动适配")
def decode_video_once(video_path: str | Path, episode_idx: int = None):
    """
    自动适配GPU/CPU的视频解码函数：
    - 有GPU：启用CUDA加速
    - 无GPU：自动降级为CPU解码
    - GPU解码失败：自动重试CPU解码
    """
    global VIDEO_DECODE_CACHE, ACTIVE_EPISODE_IDX
    
    # 标准化路径
    video_path = Path(video_path).resolve()
    cache_key = (episode_idx, str(video_path))
    
    # 1. 切换Episode时清空旧缓存
    if episode_idx != ACTIVE_EPISODE_IDX:
        keys_to_delete = [k for k in VIDEO_DECODE_CACHE if k[0] != episode_idx]
        # 输出被清理的缓存路径（可选）
        if keys_to_delete:
            deleted_paths = [k[1] for k in keys_to_delete]
            # print(f"[Cache] 切换到Episode {episode_idx}，清理旧缓存路径：{deleted_paths}")
        # else:
        #     print(f"[Cache] 切换到Episode {episode_idx}，无旧缓存需要清理")
        
        for k in keys_to_delete:
            del VIDEO_DECODE_CACHE[k]
        ACTIVE_EPISODE_IDX = episode_idx
        # 输出剩余缓存的路径（可选）
        remaining_paths = [k[1] for k in VIDEO_DECODE_CACHE.keys()]
        # print(f"[Cache] 切换后剩余缓存项：{len(VIDEO_DECODE_CACHE)}个 | 路径列表：{remaining_paths}")
    
    # 2. 命中缓存直接返回（核心修改：输出完整路径）
    if cache_key in VIDEO_DECODE_CACHE:
        # 输出缓存命中的完整路径 + 文件名
        # print(f"[Cache] 命中缓存：Episode {episode_idx} | 文件名：{video_path.name} | 完整路径：{str(video_path)}")
        return VIDEO_DECODE_CACHE[cache_key]
    
    # 3. 核心解码逻辑（自动适配GPU/CPU）
    keyframe_hashes = []
    all_frames_bgr = []
    valid = True
    
    def _decode_with_config(use_gpu: bool):
        """内部解码函数，支持不同配置"""
        nonlocal keyframe_hashes, all_frames_bgr, valid
        try:
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]
            
            # 设置解码参数（根据是否使用GPU）
            if use_gpu:
                video_stream.codec_context.options = {
                    "hwaccel": "cuda",
                    "hwaccel_device": "0",
                    "thread_count": "8",
                    "preset": "fast"
                }
                decode_label = "GPU"
            else:
                video_stream.codec_context.options = {
                    "thread_count": "8",  # CPU多线程
                    "preset": "fast"
                }
                decode_label = "CPU"
            
            # 批量解码所有帧
            frames = list(container.decode(video_stream))
            
            # 转换帧数据
            for frame in frames:
                # GPU帧转换（如果启用GPU）
                if use_gpu and hasattr(frame, 'hw_device') and frame.hw_device is not None:
                    frame = frame.reformat(format="bgr24", hwaccel="cuda")
                bgr = frame.to_ndarray(format="bgr24")
                all_frames_bgr.append(bgr)
                
                # 关键帧哈希计算
                if frame.key_frame or frame.pict_type == "I":
                    img = frame.to_image()
                    h = imagehash.phash(img, hash_size=16)
                    keyframe_hashes.append(h)
            
            container.close()
            # print(f"[{decode_label}解码] 完成：{video_path.name} | 总帧数：{len(all_frames_bgr)} | 关键帧数：{len(keyframe_hashes)} | 完整路径：{str(video_path)}")
            return True
        except Exception as e:
            # print(f"[{decode_label}解码失败] Episode {episode_idx} | {video_path.name} | 完整路径：{str(video_path)} | 错误：{str(e)}")
            return False
    
    # 4. 先尝试GPU解码（如果有GPU），失败则重试CPU
    use_gpu = check_cuda_availability()
    if use_gpu:
        gpu_decode_success = _decode_with_config(use_gpu=True)
        if not gpu_decode_success:
            # GPU解码失败，重试CPU
            keyframe_hashes = []
            all_frames_bgr = []
            _decode_with_config(use_gpu=False)
    else:
        # 无GPU，直接CPU解码
        _decode_with_config(use_gpu=False)
    
    # 5. 存入缓存
    result = {
        "keyframe_hashes": keyframe_hashes,
        "all_frames_bgr": all_frames_bgr,
        "valid": valid and len(all_frames_bgr) > 0
    }
    VIDEO_DECODE_CACHE[cache_key] = result
    # 输出缓存新增的路径（可选）
    # print(f"[Cache] 新增缓存项：Episode {episode_idx} | 路径：{str(video_path)}")
    
    return result

def clear_video_decode_cache():
    global VIDEO_DECODE_CACHE, ACTIVE_EPISODE_IDX
    VIDEO_DECODE_CACHE.clear()
    ACTIVE_EPISODE_IDX = None
    print("[Cache] 全局视频解码缓存已清空")

@dataset_data_checker_registry("few_episode_frames")
def detect_short_episodes(
    episode_frame_nums: dict[int, int], ignored_indices: set[int], threshold: int = 150
) -> set[int]:
    """
    【数据集级算子】检测单Episode帧数过少的异常
    功能：筛选出帧数小于阈值的Episode，判定为异常Episode
    入参说明：
            - episode_frame_nums：字典，key=Episode索引，value=该Episode的帧数
            - ignored_indices：已被标记为异常的Episode索引集合，检测时会跳过这些Episode
            - threshold：帧数阈值，默认150帧，低于该值的Episode会被判定为异常
    得分影响因素：
        1. threshold（阈值）：默认150帧，阈值越小越容易判定为正常，越大越严格
        2. episode_frame_nums：各Episode的实际帧数
        3. ignored_indices：已被标记为异常的Episode索引（会跳过检测）
    分数/返回值：
        - 返回值：异常Episode的索引集合（set[int]）
        - 无直接分数，被判定为异常的Episode会被加入bad_episodes，最终标记为is_bad=1
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
    episode_frame_nums: dict[int, int], ignored_indices: set[int], threshold: int = 30,
) -> set[int]:
    """
    【数据集级算子】检测数据集总有效Episode数量过少的异常
    功能：若数据集内有效Episode数量（排除ignored_indices）小于阈值，判定整个数据集所有Episode为异常
    入参说明：
        - episode_frame_nums：字典，key=Episode索引，value=该Episode的帧数
        - ignored_indices：已被标记为异常的Episode索引集合，最终会从总数中扣除这些Episode
        - threshold：Episode数量阈值，默认30个Episode，低于该值的数据集会被判定为异常
    得分影响因素：
        1. threshold（阈值）：默认30个Episode，阈值越小越容易判定为正常，越大越严格
        2. episode_frame_nums：数据集内所有Episode的索引集合
        3. ignored_indices：已被标记为异常的Episode索引（会从总数中扣除）
    分数/返回值：
        - 返回值：若有效数<阈值，返回所有Episode索引集合；否则返回空集合
        - 无直接分数，被判定为异常的Episode会被加入bad_episodes，最终标记为is_bad=1
    """
    bad_episodes = set()
    if len(set(episode_frame_nums.keys()) - (ignored_indices)) < threshold:
        return set(episode_frame_nums.keys())
    return bad_episodes


@dataset_data_checker_registry("abnormal_episode_length")
def detect_frame_num_outliers_mad_idx(
    episode_frame_nums: dict[int, int], ignored_indices: set[int], threshold: float = 5.0
) -> set[int]:
    """
    【数据集级算子】基于MAD（中位数绝对偏差）检测Episode帧数离群值
    功能：找出帧数显著偏离数据集整体分布的Episode（过短/过长），判定为异常
    入参说明：
        - episode_frame_nums：字典，key=Episode索引，value=该Episode的帧数
        - ignored_indices：已被标记为异常的Episode索引集合，检测时会跳过这些Episode
        - threshold：MAD阈值，默认5.0，值越大越宽松（更少Episode被判定为离群），值越小越严格
    得分影响因素：
        1. threshold（MAD阈值）：默认5.0，值越大越宽松（更少Episode被判定为离群），值越小越严格
        2. episode_frame_nums：各Episode的实际帧数（仅检测未被忽略的Episode）
        3. 数据集整体帧数分布：中位数越集中，越容易检测出离群值
    分数/返回值：
        - 返回值：帧数离群的Episode索引集合（set[int]）
        - 无直接分数，被判定为异常的Episode会被加入bad_episodes
    判定逻辑：
        1. 计算所有有效Episode帧数的中位数
        2. 计算每个Episode帧数相对中位数的比值
        3. 基于MAD计算修正Z-score，绝对值>threshold则判定为离群值
    """
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
    """
    辅助函数：按维度检测数据是否近似静止（每个维度都需满足静止条件）
    返回：True=静止，False=非静止
    """
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
    辅助函数：对每个维度独立进行 Z-score 归一化
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
    """
    辅助函数：检测单个滑动窗口内的数据是否静止
    返回：True=静止，False=非静止
    """
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
    data: np.ndarray, window_size: int = 5, threshold: float = 0.01,  skipsize: float = 0.05
) -> float:
    """
    【Episode数据算子】统计数据中静止帧的占比
    功能：通过滑动窗口检测每个窗口是否静止，统计所有被标记为静止的帧的总占比
    入参说明：
        - data：形状为 (T, D) 的数值型数组，T为帧数，D为维度数
        - window_size（窗口大小）：默认5帧，表示滑动窗口的长度，越大越容易检测到静止（单帧抖动不影响），越小越敏感
        - threshold（静止阈值）：默认0.01，表示相对标准差的阈值，值越小越严格（更少帧被判定为静止），值越大越宽松
        - skipsize（跳过比例）：默认0.05，表示在统计静止占比时跳过前后一定比例的帧（避免开头结尾的过渡帧），值越大跳过越多，有效帧范围越小  
    得分影响因素：
        1. window_size（窗口大小）：默认5帧，窗口越大越容易检测到静止（单帧抖动不影响），越小越敏感
        2. threshold（静止阈值）：默认0.01，值越小越严格（更少帧被判定为静止），值越大越宽松
        3. skipsize（跳过比例）：默认0.05，值越大跳过越多，有效帧范围越小
        4. 数据波动程度：数据越平稳，静止帧占比越高；波动越大，占比越低
        5. 数据归一化：每个维度独立归一化，消除量纲影响
    分数/返回值：
        - 返回值：0.0 ~ 1.0 的浮点数，表示「中间 (1-2*skipsize) 有效帧」中静止帧的占比
        - 分数越高：静止帧越多，数据越“不动”，质检得分越低（最终会用1 - 该值计算得分）
        - 特殊值：
          - 0.0：无静止帧（数据全程波动）
          - 1.0：全帧静止（数据无变化）
          - 单帧数据直接返回1.0（视为静止）
    """
    skipsize = max(0.0, min(0.1, skipsize))
    if data.size == 0:
        return 0.0  # 统一返回浮点数，保持类型一致
    frame_count = data.shape[0]
    if frame_count == 1:
        return 1.0  # 单帧视为静止

    # 总帧数过少（<20帧）时，不跳过帧（避免无有效帧）
    if frame_count < 20:
        start_idx = 0
        end_idx = frame_count
    else:
        # 向上取整，确保跳过帧数为整数
        skip_frames = int(np.ceil(frame_count * skipsize))
        start_idx = skip_frames
        end_idx = frame_count - skip_frames
        # 极端情况：跳过过多导致无有效帧（比如skipsize=0.6，总帧数10→跳过6帧，中间剩-2帧）
        if start_idx >= end_idx:
            return 0.0

    # 切片获取有效帧
    data_valid = data[start_idx:end_idx]
    valid_frame_count = data_valid.shape[0]
    # 有效帧仅1帧时，视为静止
    if valid_frame_count == 1:
        return 1.0

    # Step 1: 对有效帧做归一化
    data_norm = normalize_per_dimension(data_valid)

    # Step 2: 标记有效帧中哪些属于静止窗口（数组长度改为有效帧数量）
    frame_static = np.zeros(valid_frame_count, dtype=bool)

    # 滑动窗口仅遍历有效帧（范围改为有效帧的长度）
    for i in range(0, valid_frame_count - window_size + 1, 1):
        if is_window_static(data_norm[i : i + window_size], threshold):
            frame_static[i : i + window_size] = True

    # Step 3: 统计有效帧中的静止占比（分母改为有效帧总数）
    total_static = np.sum(frame_static)
    return float(total_static / valid_frame_count)



@episode_data_checker_registry("static_joint")
def detect_static_joint(
    data: np.ndarray, epsilon: float = 0.001
) -> float:
    """
    【Episode数据算子】计算机械臂关节数据的**静态关节占比**
    功能：返回 【标准差 < epsilon 的关节数量 / 总关节数】
    入参说明：
    - data：形状为 (T, D) 的数值型数组，T为帧数，D为关节维度数
    - epsilon：标准差阈值，用于判断关节数是否静止，默认0.001，值越小越严格（更少关节被判定为静止），值越大越宽松
    
    返回值：0.0 ~ 1.0 之间的浮点数
        - 0.0：所有关节都在运动
        - 0.5：一半关节静止
        - 1.0：所有关节全程不动
    """
    if data.ndim != 2:
        raise ValueError("Input data must be 2D array of shape (time_steps, dimensions).")

    frame_num, dim = data.shape
    if dim == 0:
        return 1.0  # 无维度，默认静态

    if frame_num <= 1:
        # 只有一帧：所有关节都视为静止
        return 1.0

    # 计算每个关节的全局标准差
    stds = np.std(data, axis=0)
    is_static_dim = stds < epsilon
    static_dim_count = np.sum(is_static_dim)

    # 直接返回【静态关节比例】
    static_ratio = static_dim_count / dim
    return static_ratio

def detect_stable_then_jump_frames(
    video_path: str,
    hash_size: int = 16,
    stable_distance_threshold: int = 1,
    min_stable_frames: int = 2,
    episode_idx: int = None
) -> tuple[int, int]:
    """
    辅助函数：检测视频关键帧中“稳定后突然跳变”的最大汉明距离
    功能：先检测连续稳定的关键帧序列，再找序列后第一个大幅跳变的汉明距离
    得分影响因素：
        1. hash_size（哈希尺寸）：默认16，尺寸越大哈希越精细，汉明距离值越大
        2. stable_distance_threshold（稳定阈值）：默认1，值越小稳定判定越严格
        3. min_stable_frames（最小稳定帧数）：默认2，值越大越难检测到稳定序列
        4. 关键帧数量：关键帧越少，越难检测到跳变
    返回值：(max_jump, frame_idx)
        - max_jump：稳定后跳变的最大汉明距离（≥0）
        - frame_idx：该跳变对应的关键帧索引
    """
    if episode_idx is None:
        raise ValueError("detect_stable_then_jump_frames必须传递episode_idx参数")
    # 从缓存取（只解码一次）
    cache = decode_video_once(video_path,episode_idx)
    if not cache["valid"]:
        return 0, 0

    keyframe_hashes = cache["keyframe_hashes"]

    if len(keyframe_hashes) < 2:
        return 0, 0

    distances = [
        keyframe_hashes[i] - keyframe_hashes[i-1]
        for i in range(1, len(keyframe_hashes))
    ]

    max_jump = 0
    frame_idx = 0

    for i in range(min_stable_frames, len(distances)):
        is_stable = all(
            d <= stable_distance_threshold
            for d in distances[i - min_stable_frames:i]
        )
        if is_stable:
            jump_distance = distances[i]
            if jump_distance > max_jump:
                max_jump = jump_distance
                frame_idx = i

    return max_jump, frame_idx


@episode_video_checker_registry("max_frame_stable_then_jump_rate")
def dectect_max_frame_stable_then_jump(video_paths: list[str | Path], episode_idx: int = None) -> float:
    """
    【Episode视频算子】检测视频稳定后跳变的最大汉明距离
    功能：计算所有视频中“稳定后跳变”的最大汉明距离，并归一化到0~1范围
    得分影响因素：
        1. 视频关键帧的跳变程度：跳变越大，max_jump值越大，返回分数越高
        2. 视频文件有效性：视频不存在/无法解码，直接返回1.0（最高异常分）
        3. 关键帧数量：关键帧越少，越难检测到跳变，分数越低
    分数/返回值：
        - 返回值：0.0 ~ 1.0 的浮点数（max_jump / 125）
        - 分数越高：跳变越严重，视频越异常，质检得分越低（最终会用1 - 该值计算得分）
        - 特殊值：
          - 0.0：无跳变（所有关键帧稳定）
          - 1.0：跳变汉明距离≥125 或 视频无效
          - 中间值：跳变汉明距离在0~125之间，分数线性映射（跳变越大分数越高）

    """
    max_jump = 0
    for video_path in video_paths:
        if not Path(video_path).exists() or not Path(video_path).is_file():
            return 1
        current_jump, _ = detect_stable_then_jump_frames(str(video_path), episode_idx=episode_idx)
        max_jump = max(max_jump, current_jump)
    if max_jump > 125:
        max_jump = 125  # 限制最大值，避免极端跳变导致分数过高
    return max_jump / 125  # 归一化，假设100以上视为严重异常，125是为了让分数在0~1范围内更平滑

运行

@episode_video_checker_registry("max_frame_jump_dist")
def detect_max_frame_jump_dist(
    video_paths: list[str | Path], max_dist_threshold: int = 250, episode_idx: int = None
) -> float:
    """
    【Episode视频算子】检测视频帧的最大跳变距离
    功能：计算视频中最大的画面跳变值，并归一化为 0~1 的连续分数
    入参说明：
        - video_paths：视频文件路径列表
        - max_dist_threshold：跳变阈值，默认250，越小越严格
        - episode_idx：Episode索引，用于解码缓存
    返回值（0 ~ 1）：
        - 0.0：无任何跳变
        - 0~1：跳变距离线性映射（距离越大分数越高）
        - 1.0：跳变超过阈值 或 视频文件无效
    """
    max_jump = 0
    
    # 遍历所有视频
    for video_path in video_paths:
        # 文件无效 → 直接返回最高异常分
        if not Path(video_path).exists() or not Path(video_path).is_file():
            return 1.0
        
        # 计算当前视频最大跳变
        current_jump, _ = detect_stable_then_jump_frames(
            str(video_path),
            stable_distance_threshold=1,
            min_stable_frames=0,
            episode_idx=episode_idx
        )
        max_jump = max(max_jump, current_jump)

    # 封顶，防止分数超过 1
    max_jump = min(max_jump, max_dist_threshold)

    # 归一化到 0~1，防止除零
    if max_dist_threshold <= 0:
        return 1.0
    return max_jump / max_dist_threshold

@data_video_consistency_checker_registry("LengthConsistencyChecker")
def decect_inconsistent_length(video_paths: list[str | Path], frame_num: int) -> bool:
    """
    【数据-视频一致性算子】检测视频帧数与数据帧数是否一致
    功能：验证每个视频的帧数是否等于给定的frame_num，不一致则判定为异常
    得分影响因素：
        1. frame_num：数据的目标帧数（基准值）
        2. 视频实际帧数：视频流的frames属性值（部分视频可能返回0/None，会判定为不一致）
        3. 视频文件有效性：视频不存在/无法打开，直接判定为不一致
    分数/返回值：
        - 返回值：bool值（True=不一致，False=一致）
        - True：帧数不一致 → 标记为bad_episode，最终is_bad=1
        - False：帧数一致 → 正常，is_bad=0
    """
    for video_path in video_paths:
        if not Path(video_path).exists() or not Path(video_path).is_file():
            return True
        container = av.open(video_path)
        if container.streams.video[0].frames != frame_num:
            return True

    return False


def compute_mean_colors(image: np.ndarray) -> dict:
    """辅助函数：计算图像各通道的平均颜色（BGR）"""
    return {
        "B": np.mean(image[:, :, 0]),
        "G": np.mean(image[:, :, 1]),
        "R": np.mean(image[:, :, 2]),
    }


@episode_video_checker_registry("video_color_shift_detection")
def detect_video_color_shift(
    video_paths: list[str | Path],
    color_diff_threshold: float = 30.0,
    episode_idx: int = None
) -> float:
    """
    【Episode视频算子】检测视频帧间颜色突变的占比
    功能：统计视频中相邻帧颜色差值超过阈值的帧数占比
    得分影响因素：
        1. color_diff_threshold（颜色差阈值）：默认30.0，值越小越容易检测到突变，越大越宽松
        2. 帧间颜色变化：B/G/R通道中最大差值超过阈值则判定为突变
        3. 视频解码有效性：无法解码/无帧，直接返回1.0（最高异常分）
        4. 视频帧数：帧数越多，统计越准确
    分数/返回值：
        - 返回值：0.0 ~ 1.0 的浮点数，表示颜色突变帧占总有效帧的比例
        - 分数越高：颜色突变越多，视频越异常，质检得分越低（最终会用1 - 该值计算得分）
        - 特殊值：
          - 0.0：无颜色突变（所有帧颜色稳定）
          - 1.0：全帧突变 或 视频无效
          - 无有效帧（total_valid_frames=0）：返回1.0
    """
    total_color_shift_frames = 0
    total_valid_frames = 0

    for video_path in video_paths:
        cache = decode_video_once(video_path, episode_idx=episode_idx)
        if not cache["valid"]:
            return 1.0

        all_frames_bgr = cache["all_frames_bgr"]
        if not all_frames_bgr:
            return 1.0

        prev_mean = None
        for img in all_frames_bgr:
            B = np.mean(img[:, :, 0])
            G = np.mean(img[:, :, 1])
            R = np.mean(img[:, :, 2])
            curr_mean = (B, G, R)

            if prev_mean is not None:
                diff = max(abs(curr_mean[0]-prev_mean[0]),
                           abs(curr_mean[1]-prev_mean[1]),
                           abs(curr_mean[2]-prev_mean[2]))
                if diff > color_diff_threshold:
                    total_color_shift_frames += 1
                total_valid_frames += 1

            prev_mean = curr_mean

    if total_valid_frames == 0:
        return 1.0

    return total_color_shift_frames / total_valid_frames


def compute_phash(image: np.ndarray, hash_size: int = 16) -> str:
    """计算图像的感知哈希（复用原有解码的BGR帧，无需重复解码）"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 调整大小
    resized = cv2.resize(gray, (hash_size, hash_size))
    # 计算DCT
    dct = cv2.dct(np.float32(resized))
    # 取左上角8x8区域的平均值
    dct_left = dct[:8, :8]
    avg = np.mean(dct_left)
    # 生成二进制哈希
    phash = "".join(["1" if x > avg else "0" for x in dct_left.flatten()])
    return phash

def hamming_distance(hash1: str, hash2: str) -> int:
    """计算两个哈希的汉明距离"""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

@episode_video_checker_registry("consecutive_static_frames")
def detect_consecutive_static_frames(
    video_paths: list[str | Path],
    phash_dist_threshold: int = 5,
    sample_count: int = 8,  # 均匀采样的关键帧数量
    skipsize: float = 0.05,  # 统计静止占比时跳过首尾帧的比例
    episode_idx: int = None
) -> float:
    """
    【Episode视频算子】混合检测视频卡死+连续静止帧占比（适配机器人左右手交替场景）
    核心优化：
        1. 卡死判定：先统计关键帧总数，再均匀采样8个关键帧比对，避免局部连续重复误判；
        2. 静止占比：基于全部视频帧（剔除首尾5%）计算，保证统计全面性；
        3. 日志输出关键帧数量，方便调试。
    返回值规则：
        0.0 = 无连续静止帧（正常）
        1.0 = 采样关键帧哈希一致（相机卡死）/视频无效
        0~1 之间 = 中间90%全帧内连续静止占比（越高越异常）
    """
    max_abnormal_score = 1.0

    for video_path in video_paths:
        video_path = Path(video_path)
        if not video_path.exists() or not video_path.is_file():
            return 1.0
        
        # 1. 从缓存获取解码数据
        cache = decode_video_once(video_path, episode_idx=episode_idx)
        if not cache["valid"]:
            return 1.0
        
        # 关键帧相关（核心：先统计数量，再均匀采样）
        keyframe_hashes = cache["keyframe_hashes"]
        total_keyframes = len(keyframe_hashes)
        
        # 2. 卡死判定：均匀采样关键帧比对（替代原相邻比对）
        camera_freeze_detected = False
        # 边界：关键帧数量≥采样数，才进行采样比对
        if total_keyframes >= sample_count:
            # 计算采样步长（均匀分布）
            sample_step = math.ceil(total_keyframes / sample_count)
            # 选取采样的关键帧索引
            sample_indices = [i * sample_step for i in range(sample_count)]
            # 确保最后一个采样索引不越界
            sample_indices[-1] = min(sample_indices[-1], total_keyframes - 1)
            
            # 获取采样关键帧的哈希
            sample_hashes = [keyframe_hashes[idx] for idx in sample_indices]
            # 检查所有采样哈希是否完全一致
            first_hash = sample_hashes[0]
            all_sample_same = True
            for hash_val in sample_hashes[1:]:
                # 兼容字符串/数值型哈希
                if isinstance(first_hash, str) and isinstance(hash_val, str):
                    dist = hamming_distance(first_hash, hash_val)
                else:
                    dist = first_hash - hash_val
                
                if dist != 0:
                    all_sample_same = False
                    break
            
            # 所有采样帧哈希一致 → 判定相机卡死
            if all_sample_same:
                camera_freeze_detected = True
                print(f"⚠️ [严重异常] Episode {episode_idx} 视频 {video_path} 均匀采样{sample_count}个关键帧（索引：{sample_indices}）哈希完全一致，判定相机卡死")
        
        # 检测到卡死，直接返回最高异常分
        if camera_freeze_detected:
            return 1.0
        
        # 3. 静止占比统计：基于全部视频帧（剔除首尾5%）
        all_frames_bgr = cache.get("all_frames_bgr", [])
        total_frames = len(all_frames_bgr)
        all_frame_hashes = []
        if total_frames > 0:
            for frame_bgr in all_frames_bgr:
                all_frame_hashes.append(compute_phash(frame_bgr))
        
        abnormal_score = 0.0
        max_static_frames = 0
        if total_frames >= 2 and len(all_frame_hashes) == total_frames:
            # 剔除首尾5%的帧
            trim_count = int(total_frames * skipsize)
            start_frame_idx = trim_count
            end_frame_idx = total_frames - trim_count
            middle_frame_hashes = all_frame_hashes[start_frame_idx:end_frame_idx]
            middle_frames_total = len(middle_frame_hashes)
            
            if middle_frames_total >= 2:
                static_count = 1
                prev_hash = middle_frame_hashes[0]
                for frame_idx in range(1, middle_frames_total):
                    current_hash = middle_frame_hashes[frame_idx]
                    distance = hamming_distance(prev_hash, current_hash)
                    
                    if distance <= phash_dist_threshold:
                        static_count += 1
                        max_static_frames = max(max_static_frames, static_count)
                    else:
                        static_count = 1
                    
                    prev_hash = current_hash
                abnormal_score = max_static_frames / middle_frames_total if middle_frames_total > 0 else 1.0
            else:
                abnormal_score = 0.0
        else:
            abnormal_score = 0.0
        
        # 更新最小异常分
        if abnormal_score < max_abnormal_score:
            max_abnormal_score = abnormal_score

    return max_abnormal_score


@episode_video_checker_registry("camera_resolution_consistency")
def detect_camera_resolution_consistency(
    video_paths: list[str | Path],
    episode_idx: int = None
) -> float:
    """
    【适配真实目录结构】自动从视频路径推导info.json路径，检测分辨率一致性
    目录结构（关键）：
        根目录：~/Documents/after_convert/Agilex_Cobot_Magic_Agilex_Cobot_Magic_pour_drink_bottle_cup/
        ├── meta/
        │   └── info.json （目标文件）
        └── videos/
            └── chunk-000/
                └── observation.images.cam_high_rgb/
                    └── episode_000000.mp4 （视频文件）
    返回值规则：
        0.0 = 所有相机分辨率一致（正常）
        1.0 = 任意异常（分辨率不匹配/文件损坏/路径推导失败）
    """
    # 0. 校验视频路径是否有效
    if not video_paths or not Path(video_paths[0]).exists():
        return 1.0

    # 1. 自动推导info.json路径（核心适配你的目录结构）
    first_video_path = Path(video_paths[0]).absolute()
    info_json_path = None
    
    # 向上回溯目录，找到包含 "meta" 文件夹的根目录
    current_dir = first_video_path.parent
    while current_dir != current_dir.parent:  # 直到根目录
        meta_dir = current_dir / "meta"
        if meta_dir.exists() and meta_dir.is_dir():
            info_json_path = meta_dir / "info.json"
            break
        current_dir = current_dir.parent
    
    # 校验：未找到info.json，直接标记异常
    if not info_json_path or not info_json_path.exists():
        print(f"未找到info.json，路径推导失败：{info_json_path}")
        return 1.0

    # 2. 读取info.json，提取所有相机的标准分辨率（修复核心：正确解析层级）
    try:
        with open(info_json_path, "r", encoding="utf-8") as f:
            info_data = json.load(f)
        
        # 构建相机分辨率映射表：{相机完整名: (标准宽, 标准高)}
        camera_resolution_map = {}
        features = info_data.get("features", {})
        
        # 筛选规则：key以 "observation.images." 开头，且是视频类型
        for cam_full_key, cam_info in features.items():
            if not cam_full_key.startswith("observation.images."):
                continue  # 跳过非相机的key（如state/action等）
            
            cam_detail_info = cam_info.get("info", {})
            standard_width = cam_detail_info.get("video.width", 0)
            standard_height = cam_detail_info.get("video.height", 0)
            
            if standard_width > 0 and standard_height > 0:
                camera_resolution_map[cam_full_key] = (standard_width, standard_height)
        
        # 校验：无有效相机分辨率配置
        if not camera_resolution_map:
            print(f"info.json中未找到有效相机分辨率配置，camera_resolution_map={camera_resolution_map}")
            return 1.0
        
            
    except Exception as e:
        print(f"解析info.json失败：{str(e)}")
        return 1.0  # info.json解析失败

    # 3. 遍历所有视频，检查分辨率
    for video_path in video_paths:
        video_path = Path(video_path).absolute()
        
        # 跳过不存在的视频
        if not video_path.exists() or not video_path.is_file():
            print(f"视频文件不存在：{video_path}")
            return 1.0
        
        # 提取相机完整名（视频父目录名，如 observation.images.cam_high_rgb）
        cam_full_name = video_path.parent.name
        
        # 校验：该相机是否在info.json中有配置
        if cam_full_name not in camera_resolution_map:
            print(f"相机{cam_full_name}不在info.json的配置列表中，配置列表：{camera_resolution_map.keys()}")
            return 1.0
        
        # 获取标准分辨率
        standard_width, standard_height = camera_resolution_map[cam_full_name]
        
        # 4. 读取视频实际分辨率（不解码帧，仅读流信息）
        try:
            container = av.open(str(video_path))
            video_stream = None
            for stream in container.streams:
                if stream.type == "video":
                    video_stream = stream
                    break
            if video_stream is None:
                container.close()
                print(f"视频{video_path}无视频流")
                return 1.0
            
            actual_width = video_stream.width
            actual_height = video_stream.height
            container.close()
            
            # 分辨率不一致，直接返回异常
            if actual_width != standard_width or actual_height != standard_height:
                print(f"分辨率不匹配：视频{video_path}实际({actual_width},{actual_height})，标准({standard_width},{standard_height})")
                return 1.0
                
        except Exception as e:
            print(f"读取视频{video_path}分辨率失败：{str(e)}")
            return 1.0  # 视频打开/读取失败

    # 所有视频分辨率都匹配，返回正常
    return 0.0

@episode_data_checker_registry("motion_data_valid_frame_range")
def get_valid_motion_frame_range_from_data(
    data: np.ndarray,
    window_size: int = 5,
    threshold: float = 0.01
) -> tuple[int, int]:
    """
    【Episode数据算子】基于运动数据获取有效帧区间（剔除首尾静止帧，留1帧余量）
    返回：(start_index, end_index) 保证 0 ≤ start ≤ end < 总帧数
    """
    if data is None or data.size == 0:
        return (-1, -1)
    frame_count = data.shape[0]
    if frame_count <= 1:
        return (0, frame_count - 1)
    
    data_norm = normalize_per_dimension(data)
    
    # 首部第一个非静止帧
    first_motion_frame = 0
    for i in range(0, frame_count - window_size + 1, 1):
        if not is_window_static(data_norm[i:i+window_size], threshold):
            first_motion_frame = i
            break
    
    # 尾部最后一个非静止帧
    last_motion_frame = frame_count - 1
    for i in range(frame_count - window_size, -1, -1):
        if not is_window_static(data_norm[i:i+window_size], threshold):
            last_motion_frame = i + window_size - 1
            break
    
    # 留1帧余量
    start_index = max(0, first_motion_frame - 1)  # 首部往前多留1帧
    end_index = min(frame_count - 1, last_motion_frame + 1)  # 尾部往后多留1帧
    # 最终兜底：确保start ≤ end
    start_index = min(start_index, end_index)
    return (start_index, end_index)


@episode_video_checker_registry("video_valid_frame_range")
def get_valid_motion_frame_range_from_video(
    video_paths: list[str | Path] | str | Path,
    episode_idx: int,
    phash_dist_threshold: int = 5  # 仅保留单帧阈值，移除连续帧参数
) -> tuple[int, int]:
    """
    【核心功能】
    1. 对单个视频：仅通过单帧哈希距离>阈值，找到首尾第一个非静止帧，剔除首尾静止帧
    2. 对多视频：取所有视频中「最小起始帧 + 最大结束帧」（最宽容区间）
    返回：(start_index, end_index) 保证 0 ≤ start ≤ end < 总帧数
    """
    # 1. 统一路径格式：单路径转列表
    if not isinstance(video_paths, list):
        video_paths = [video_paths]
    
    all_valid_ranges = []  # 存储每个视频的有效区间
    total_frames_list = []  # 存储每个视频的总帧数

    # 2. 遍历每个视频，计算其有效区间（仅单帧判定剔除首尾静止帧）
    for path in video_paths:
        single_path = Path(path)
        if not single_path.exists() or not single_path.is_file():
            continue
        
        # 解码视频（复用你的缓存逻辑）
        cache = decode_video_once(single_path, episode_idx=episode_idx)
        if not cache["valid"] or len(cache["all_frames_bgr"]) <= 1:
            # 无效视频/帧数过少，直接返回全区间
            total_frames = len(cache["all_frames_bgr"]) if cache["valid"] else 0
            all_valid_ranges.append((0, max(0, total_frames - 1)))
            total_frames_list.append(total_frames)
            continue
        
        all_frames_bgr = cache["all_frames_bgr"]
        total_frames = len(all_frames_bgr)
        frame_hashes = [compute_phash(img) for img in all_frames_bgr]

        # ===================== 核心：找首部第一个非静止帧 =====================
        start_idx = 0
        for i in range(1, total_frames):
            dist = hamming_distance(frame_hashes[i-1], frame_hashes[i])
            if dist > phash_dist_threshold:
                start_idx = i  # 找到第一个非静止帧
                break
        start_idx = max(0, start_idx - 1)  # 留1帧余量

        # ===================== 核心：找尾部最后一个非静止帧 =====================
        end_idx = total_frames - 1
        for i in range(total_frames - 1, 0, -1):
            dist = hamming_distance(frame_hashes[i-1], frame_hashes[i])
            if dist > phash_dist_threshold:
                end_idx = i  # 找到最后一个非静止帧
                break
        end_idx = min(total_frames - 1, end_idx + 1)  # 留1帧余量

        # 兜底：确保start ≤ end
        start_idx = min(start_idx, end_idx)
        all_valid_ranges.append((start_idx, end_idx))
        total_frames_list.append(total_frames)
        # 可选调试日志：查看单个视频的剔除结果
        # print(f"[调试] {single_path.name} | 总帧：{total_frames} | 有效区间：({start_idx}, {end_idx})")

    # 3. 处理无有效视频的情况
    if not all_valid_ranges:
        return (-1, -1)
    
    # 4. 多视角最宽容区间：最小start + 最大end
    min_start = min([r[0] for r in all_valid_ranges])  # 保留最早的起始帧
    max_end = max([r[1] for r in all_valid_ranges])    # 保留最晚的结束帧

    # 5. 最终边界校验（避免越界）
    if total_frames_list:
        max_total = max(total_frames_list)
        max_end = min(max_end, max_total - 1)
    min_start = max(0, min_start)
    min_start = min(min_start, max_end)

    return (min_start, max_end)
