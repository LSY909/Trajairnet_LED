import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from model.utils import TrajectoryDataset, TrajectoryDataset_RAG, seq_collate_with_padding
from model.Rag_embedder import TimeSeriesEmbedder

# === 配置 ===
K_RETRIEVE = 30
N_CLUSTERS = 3
SAVE_DIR = './traj_vis_corridors'

# 1. 加载
root = os.getcwd()
print(f"Loading data (K={K_RETRIEVE}, N={N_CLUSTERS})...")

loader = DataLoader(TrajectoryDataset(f'{root}/dataset/7days1/processed_data/train', 11, 120, 10), batch_size=1,
                    collate_fn=seq_collate_with_padding, shuffle=True)
rag = TrajectoryDataset_RAG(f'{root}/dataset/rag_file_7days2', 11, 120, 10).rag_system

# 2. 数据 & 检索
batch = next(iter(loader))
obs = batch[0][0, 0].detach().cpu().numpy().transpose(1, 0)  # (11, 3)

# Embedding & Search
res = rag.search_batch(TimeSeriesEmbedder().embed_batch(obs[None, :, :]).astype(np.float32), k=K_RETRIEVE)[0]
raw = np.array(
    [np.array(r['pred_data']).T if np.array(r['pred_data']).shape[0] == 3 else np.array(r['pred_data']) for r in res])

# --- 坐标归一化 ---
start_points = raw[:, 0:1, :]
relative_raw = raw - start_points
flat_relative_raw = relative_raw.reshape(K_RETRIEVE, -1)

# 3. GMM 聚类 & 获取宽度信息
print("Clustering...")
gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag', random_state=42).fit(flat_relative_raw)

# A. 获取中心线 (Means)
relative_routes = gmm.means_.reshape(N_CLUSTERS, 12, 3)

# B. 获取宽度 (Standard Deviations)
relative_stds = np.sqrt(gmm.covariances_).reshape(N_CLUSTERS, 12, 3)

# 4. 还原到当前观测位置
current_end_pos = obs[-1]
absolute_routes = relative_routes + current_end_pos

# ------------------------------------------
# 5. 绘图
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
plt.figure(figsize=(10, 10), dpi=120)
plt.grid(True, linestyle=':', alpha=0.6)

# A. 画历史 (绿色)
plt.plot(obs[:, 0], obs[:, 1], 'g-o', lw=3, zorder=20, label='Observed History')
plt.scatter(obs[-1, 0], obs[-1, 1], c='k', s=120, zorder=21)

# B. 画原始检索背景 (灰色)
for r in relative_raw:
    line = np.concatenate([obs[-1:], r + current_end_pos], axis=0)
    plt.plot(line[:, 0], line[:, 1], 'gray', alpha=0.05, zorder=1)

# C. 画拓宽后的航道 (Corridors)
colors = ['r', 'b', 'orange', 'purple', 'cyan']  # 增加颜色以支持更多簇

for i in range(N_CLUSTERS):
    mean_line = absolute_routes[i]
    sigma = relative_stds[i]
    color = colors[i % len(colors)]

    path_x = np.concatenate(([obs[-1, 0]], mean_line[:, 0]))
    path_y = np.concatenate(([obs[-1, 1]], mean_line[:, 1]))

    sigma_x = np.concatenate(([0], sigma[:, 0]))
    sigma_y = np.concatenate(([0], sigma[:, 1]))

    bound_factor = 1.5
    upper_x = path_x + sigma_x * bound_factor
    lower_x = path_x - sigma_x * bound_factor
    upper_y = path_y + sigma_y * bound_factor
    lower_y = path_y - sigma_y * bound_factor

    # 绘制中心虚线
    plt.plot(path_x, path_y, c=color, lw=2, ls='--', zorder=10,
             label=f'Route {i + 1} (w={gmm.weights_[i]:.2f})')

    # 绘制航道填充
    vertices_x = np.concatenate([upper_x, lower_x[::-1]])
    vertices_y = np.concatenate([upper_y, lower_y[::-1]])
    plt.fill(vertices_x, vertices_y, c=color, alpha=0.15, zorder=5)

    # 绘制边界线
    plt.plot(upper_x, upper_y, c=color, lw=1, ls=':', alpha=0.6, zorder=6)
    plt.plot(lower_x, lower_y, c=color, lw=1, ls=':', alpha=0.6, zorder=6)

plt.title(f"Generated Flight Corridors (Mean ± 1.5$\sigma$)\nK={K_RETRIEVE}, N={N_CLUSTERS}")
plt.legend(loc='best')
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

save_filename = f'gmm_corridors_k{K_RETRIEVE}_n{N_CLUSTERS}.png'
save_path = os.path.join(SAVE_DIR, save_filename)

plt.savefig(save_path)
print(f"✅ 成功！图片已保存至: {save_path}")