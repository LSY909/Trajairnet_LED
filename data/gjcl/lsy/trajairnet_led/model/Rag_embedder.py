"""Time Series Embedding Module.

This module provides functionality for converting time series data into fixed-length
embeddings using a combination of resampling and statistical features. The embeddings
can be used for similarity search and retrieval tasks.

Example:
    >>> embedder = TimeSeriesEmbedder(target_length=256)
    >>> time_series = [1.0, 2.0, 3.0, 2.0, 1.0]
    >>> embedding = embedder.embed(time_series)
    >>> print(embedding.shape)
    (1, 260)  # 256 resampled points + 4 statistical features
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
from typing import Union, List, Tuple

class TimeSeriesEmbedder:
    """A class for converting time series data into fixed-length embeddings.

    This class provides functionality to transform variable-length time series into
    fixed-length embeddings by combining resampled values with statistical features.
    The embeddings can be used for similarity search and other downstream tasks.

    Attributes:
        target_length (int): The desired length of the resampled time series.
            Default is 256 points.
        scaler (StandardScaler): A scikit-learn StandardScaler instance for
            normalizing the time series data.

    Example:
        >>> embedder = TimeSeriesEmbedder(target_length=128)
        >>> time_series = np.sin(np.linspace(0, 10, 1000))
        >>> embedding = embedder.embed(time_series)
        >>> print(f"Embedding shape: {embedding.shape}")
    """
    """
        针对飞机轨迹优化的 Embedder。
        核心改进：在 Embedding 之前，先将轨迹进行【平移归零】和【旋转对齐】，
        提取的是轨迹的“形状特征”而非“位置特征”。
        """

    def __init__(self, target_length: int = 256):

        self.target_length = target_length
        # self.scaler = StandardScaler()

    def _align_trajectory(self, traj: np.ndarray) -> np.ndarray:
        """
        核心辅助函数：将轨迹标准化。
        1. 平移：将起点移动到 (0, 0)
        2. 旋转：将初始航向旋转到 X 轴正方向 (0度)
        """
        # 1. 确保是 (T, D) 格式
        if len(traj.shape) == 1:
            traj = traj.reshape(-1, 2)  # 默认补齐为 2 维

        # 2. 平移 (Translation): 减去起点，消除地理位置差异
        start_point = traj[0, :].copy()
        centered_traj = traj - start_point

        # 如果轨迹只有一个点或所有点重合，直接返回
        if np.allclose(centered_traj, 0):
            return centered_traj

        # 3. 旋转 (Rotation): 消除航向差异
        # 为了防止起步时的抖动/噪点，取前几个点的平均方向作为初始航向
        lookahead = min(5, len(centered_traj) - 1)
        # 计算初始向量 (dx, dy)
        delta = centered_traj[lookahead] - centered_traj[0]
        angle = np.arctan2(delta[1], delta[0])
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])
        # 执行旋转
        # traj[:, :2] @ R.T 实现坐标变换
        aligned_traj = centered_traj.copy()
        aligned_traj[:, :2] = centered_traj[:, :2] @ R.T

        return aligned_traj

    def embed(self, time_series: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        修正后的单条轨迹 Embedding 函数
        """
        # 1. 格式转换
        if isinstance(time_series, list):
            time_series = np.array(time_series)

        # 确保是 2D 数组 (Time, D)
        if len(time_series.shape) == 1:
            time_series = time_series.reshape(-1, 2)

        if time_series.size == 0:
            raise ValueError("Input time series is empty")

        # 2. 【关键修正】计算对齐后的轨迹
        aligned_series = self._align_trajectory(time_series)

        # 3. 【关键修正】使用 aligned_series 进行重采样
        # 错误写法: resampled = resample(time_series, self.target_length)
        resampled = resample(aligned_series, self.target_length)  # <--- 必须用 aligned_series

        # 4. 【关键修正】使用 aligned_series 提取特征
        # 错误写法: mean = np.mean(time_series, axis=0) ...
        mean = np.mean(aligned_series, axis=0)  # <--- 必须用 aligned_series
        std = np.std(aligned_series, axis=0)  # <--- 必须用 aligned_series
        max_val = np.max(aligned_series, axis=0)  # <--- 必须用 aligned_series
        min_val = np.min(aligned_series, axis=0)  # <--- 必须用 aligned_series

        # 5. 拼接特征
        features = np.concatenate([
            resampled.flatten(),
            mean,
            std,
            max_val,
            min_val
        ])
        return features.reshape(1, -1)

    def embed_batch(self, time_series: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        批量处理轨迹 Embedding。
        逻辑：遍历 -> 对齐(Align) -> 重采样(Resample) -> 统计特征 -> 堆叠
        """
        embeddings = []

        # 1. 确定遍历对象
        # 如果是 3D 数组 (Batch, Time, Feat)，直接按第一维遍历
        # 如果是 List of Arrays，直接遍历 List
        num_samples = len(time_series)

        for i in range(num_samples):
            # 获取单条轨迹
            traj = time_series[i]

            # --- 关键修改 1: 几何对齐 ---
            # 这一步至关重要！它把“绝对坐标”变成了“标准形状”。
            # 必须调用类中定义的 _align_trajectory 方法
            aligned_traj = self._align_trajectory(traj)

            # --- 关键修改 2: 基于对齐后的数据重采样 ---
            resampled = resample(aligned_traj, self.target_length)

            # --- 关键修改 3: 提取形状统计特征 ---
            # 此时的 mean/max 是相对于起点的相对值，代表了轨迹的形态
            mean = np.mean(aligned_traj, axis=0)
            std = np.std(aligned_traj, axis=0)
            max_val = np.max(aligned_traj, axis=0)
            min_val = np.min(aligned_traj, axis=0)

            # --- 拼接 ---
            features = np.concatenate([
                resampled.flatten(),
                mean,
                std,
                max_val,
                min_val
            ])

            # 修复原代码 bug: features.reshape(1, -1) 不会改变原变量
            # 这里不需要 reshape，直接 append 1D 向量即可，最后统一 vstack
            embeddings.append(features)

        # 将 list 转为 numpy 矩阵: (Batch_Size, Feature_Dim)
        return np.vstack(embeddings)