import numpy as np
from scipy.spatial.transform import Rotation as r  # noqa: N813


def cm2m(input: np.ndarray) -> np.ndarray:
    return input / 100


def mm2m(input: np.ndarray) -> np.ndarray:
    return input / 1000


def quat_xyzw_2_euler_xyz(input: np.ndarray) -> np.ndarray:
    return r.from_quat(input).as_euler("xyz")


def quat_xyzw_to_wxyz(input: np.ndarray) -> np.ndarray:
    """
    将 ROS 四元数 (x, y, z, w) 转换为 (w, x, y, z) 顺序
    """
    if input.ndim == 1:
        return np.concatenate([input[-1:], input[:-1]])
    # 批量输入
    w = input[..., -1:]
    xyz = input[..., :-1]
    return np.concatenate([w, xyz], axis=-1)


def quat_wxyz_2_euler_xyz(input: np.ndarray) -> np.ndarray:
    quat_xyzw = np.roll(input, -1)
    return r.from_quat(quat_xyzw).as_euler("xyz")


def rot6d_to_matrix(vec6: np.ndarray) -> np.ndarray:
    """
    将 6D rotation 转为 3x3 旋转矩阵
    vec6: (6,)
    返回: (3, 3)
    """
    # 单个样本
    v1 = vec6[:3]  # x-axis
    v2 = vec6[3:]  # y-axis

    # 归一化 x-axis
    x_axis = v1 / np.linalg.norm(v1)

    # 投影并归一化 y-axis
    v2_proj = v2 - np.dot(v2, x_axis) * x_axis
    y_axis = v2_proj / np.linalg.norm(v2_proj)

    # z-axis = x × y
    z_axis = np.cross(x_axis, y_axis)

    return np.column_stack([x_axis, y_axis, z_axis])


def rot6d_to_euler_xyz(vec6: np.ndarray) -> np.ndarray:
    """
    将 6D rotation 转为欧拉角
    vec6: (6,) 或 (N, 6)
    """
    return r.from_matrix(rot6d_to_matrix(vec6)).as_euler("xyz")


def degree2rad(input: np.ndarray) -> np.ndarray:
    return np.deg2rad(input)


def multiply_by_10(input: np.ndarray) -> np.ndarray:
    """Multiply input by 10 - used to correct Magic gripper data that is 10x smaller than actual"""
    return input * 10.0

def multiply_by_100(input: np.ndarray) -> np.ndarray:
    """Multiply input by 10 - used to correct Magic gripper data that is 10x smaller than actual"""
    return input * 100.0


def to_float32(input: np.ndarray) -> np.ndarray:
    """
    Convert input to float32 dtype.
    This is a passthrough function used when data needs dtype conversion without any numerical transformation.
    Useful for H5 files that store data in float64 but LeRobot requires float32.
    """
    return input


spatial_covertor_funcs = {
    "mm2m": mm2m,
    "cm2m": cm2m,
    "quat_xyzw_to_wxyz": quat_xyzw_to_wxyz,
    "quat_wxyz_2_euler_xyz": quat_wxyz_2_euler_xyz,
    "quat_xyzw_2_euler_xyz": quat_xyzw_2_euler_xyz,
    "rot6d_to_euler_xyz": rot6d_to_euler_xyz,
    "degree2rad": degree2rad,
    "multiply_by_10": multiply_by_10,
    "to_float32": to_float32,
}
