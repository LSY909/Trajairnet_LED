import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from model.utils import TrajectoryDataset, TrajectoryDataset_RAG, seq_collate_with_padding
from model.Rag_embedder import TimeSeriesEmbedder

# === 配置 ===
K_RETRIEVE = 50  # 检索数量越多，方差(宽度)计算越准
N_CLUSTERS = 3
SAVE_DIR = './traj_vis_corridors'
DATASET_NAME = '7days1'  # 请根据实际情况修改
RAG_DIR = 'rag_file_7days2'


def get_corridors(obs, rag_system, embedder):
    """
    核心功能：输入观测，输出带宽度的航道
    """
    # 1. 检索
    query_emb = embedder.embed_batch(obs[None, :, :]).astype(np.float32)
    res = rag_system.search_batch(query_emb, k=K_RETRIEVE)[0]

    # 提取原始轨迹
    raw = np.array([
        np.array(r['pred_data']).T if np.array(r['pred_data']).shape[0] == 3 else np.array(r['pred_data'])
        for r in res
    ])

    # 2. 归一化
    start_points = raw[:, 0:1, :]
    relative_raw = raw - start_points
    flat_relative_raw = relative_raw.reshape(K_RETRIEVE, -1)

    # 3. GMM 聚类 & 宽度计算
    # covariance_type='diag' -> 计算每个维度的独立方差
    gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag', random_state=42).fit(flat_relative_raw)

    # A. 中心线 (Means) [3, 12, 3]
    relative_routes = gmm.means_.reshape(N_CLUSTERS, 12, 3)

    # B. 宽度 (Std) [3, 12, 1]
    # gmm.covariances_ 是方差，开根号得标准差
    variances = gmm.covariances_.reshape(N_CLUSTERS, 12, 3)
    stds = np.sqrt(variances)
    # 取空间的平均宽度 (或者取 max)
    widths = np.mean(stds, axis=2, keepdims=True)

    return relative_routes, widths, gmm.weights_, relative_raw


def main():
    root = os.getcwd()
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    print("1. Loading Data...")
    data_path = os.path.join(root, 'dataset', DATASET_NAME, 'processed_data', 'train')
    loader = DataLoader(TrajectoryDataset(data_path, 11, 120, 10), batch_size=1,
                        collate_fn=seq_collate_with_padding, shuffle=True)

    rag_path = os.path.join(root, 'dataset', RAG_DIR)
    rag = TrajectoryDataset_RAG(rag_path, 11, 120, 10).rag_system
    embedder = TimeSeriesEmbedder()

    # 2. 获取样本
    batch = next(iter(loader))
    obs = batch[0][0, 0].detach().cpu().numpy().transpose(1, 0)  # (11, 3)

    # 3. 计算航道
    print("2. Calculating Corridors...")
    rel_routes, widths, weights, raw_bg = get_corridors(obs, rag, embedder)

    # 还原绝对位置
    current_end_pos = obs[-1]
    abs_routes = rel_routes + current_end_pos

    # 4. 绘图
    plt.figure(figsize=(10, 10), dpi=120)
    plt.grid(True, linestyle=':', alpha=0.6)

    # 画历史
    plt.plot(obs[:, 0], obs[:, 1], 'g-o', lw=3, zorder=20, label='Observed')
    plt.scatter(obs[-1, 0], obs[-1, 1], c='k', s=120, zorder=21)

    # 画背景
    for r in raw_bg:
        line = np.concatenate([obs[-1:], r + current_end_pos], axis=0)
        plt.plot(line[:, 0], line[:, 1], 'gray', alpha=0.05, zorder=1)

    # 画航道
    colors = ['r', 'b', 'orange']
    for i in range(N_CLUSTERS):
        path_x = np.concatenate(([obs[-1, 0]], abs_routes[i, :, 0]))
        path_y = np.concatenate(([obs[-1, 1]], abs_routes[i, :, 1]))

        # 宽度 (起点设为0)
        w = np.concatenate(([0], widths[i, :, 0]))
        bound = 1.5

        color = colors[i % len(colors)]
        plt.plot(path_x, path_y, c=color, lw=2, ls='--', zorder=10, label=f'Route {i + 1}')
        plt.fill_between(path_x, path_y - w * bound, path_y + w * bound, color=color, alpha=0.2, zorder=5)

    plt.title(f"Flight Corridors Visualization\nDataset: {DATASET_NAME}")
    plt.legend()
    plt.savefig(f'{SAVE_DIR}/corridor_vis.png')
    print(f"✅ Done! Saved to {SAVE_DIR}/corridor_vis.png")


if __name__ == '__main__':
    main()