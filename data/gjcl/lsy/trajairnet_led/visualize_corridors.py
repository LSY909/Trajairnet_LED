import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from model.utils import TrajectoryDataset, TrajectoryDataset_RAG, seq_collate_with_padding
from model.Rag_embedder import TimeSeriesEmbedder

# === 配置 ===
K_RETRIEVE = 50  # 检索数量稍微大一点，计算出的方差更稳定
N_CLUSTERS = 3
SAVE_DIR = './traj_vis_corridors'

# 1. 加载 (保持不变)
root = os.getcwd()
# 注意：这里假设使用 train 数据演示
loader = DataLoader(TrajectoryDataset(f'{root}/dataset/7days1/processed_data/train', 11, 120, 10), batch_size=1,
                    collate_fn=seq_collate_with_padding, shuffle=True)
rag = TrajectoryDataset_RAG(f'{root}/dataset/rag_file_7days2', 11, 120, 10).rag_system

# 2. 数据 & 检索 (保持不变)
print("正在检索...")
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
print("正在聚类并计算航道宽度...")
# covariance_type='diag' 意味着我们得到的是每个维度独立的方差
gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag', random_state=42).fit(flat_relative_raw)

# A. 获取中心线 (Means)
relative_routes = gmm.means_.reshape(N_CLUSTERS, 12, 3)

# B. 获取宽度 (Standard Deviations)
# gmm.covariances_ 形状: (n_components, n_features) -> (3, 36)
# 我们开根号得到标准差 (Sigma)
relative_stds = np.sqrt(gmm.covariances_).reshape(N_CLUSTERS, 12, 3)

# 4. 还原到当前观测位置
current_end_pos = obs[-1]
absolute_routes = relative_routes + current_end_pos
# 标准差是相对量，不需要加位置偏移，直接用即可

# ------------------------------------------
# 5. 绘图
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
plt.figure(figsize=(10, 10), dpi=120)
plt.grid(True, linestyle=':', alpha=0.6)

# A. 画历史 (绿色)
plt.plot(obs[:, 0], obs[:, 1], 'g-o', lw=3, zorder=20, label='Observed History')
plt.scatter(obs[-1, 0], obs[-1, 1], c='k', s=120, zorder=21)  # 起点

# B. 画原始检索背景 (灰色，极淡)
# 这样可以对比看看生成的“航道”是否覆盖了这些原始线
for r in relative_raw:
    line = np.concatenate([obs[-1:], r + current_end_pos], axis=0)
    plt.plot(line[:, 0], line[:, 1], 'gray', alpha=0.05, zorder=1)

# C. 画拓宽后的航道 (Corridors)
colors = ['r', 'b', 'orange']
fill_colors = ['#ffcccc', '#ccccff', '#ffeeb0']  # 对应的浅色填充

for i in range(N_CLUSTERS):
    # 提取中心线和宽度
    mean_line = absolute_routes[i]  # (12, 3)
    sigma = relative_stds[i]  # (12, 3)

    # 为了画图连贯，把起点加进去
    # 起点的 Sigma 设为 0 (因为当前位置是确定的)
    path_x = np.concatenate(([obs[-1, 0]], mean_line[:, 0]))
    path_y = np.concatenate(([obs[-1, 1]], mean_line[:, 1]))

    sigma_x = np.concatenate(([0], sigma[:, 0]))
    sigma_y = np.concatenate(([0], sigma[:, 1]))

    # 计算边界 (1 Sigma 范围，覆盖约 68% 的概率区间)
    # 你也可以乘 2 来看 2 Sigma (95%) 范围
    bound_factor = 1.5

    upper_x = path_x + sigma_x * bound_factor
    lower_x = path_x - sigma_x * bound_factor
    upper_y = path_y + sigma_y * bound_factor
    lower_y = path_y - sigma_y * bound_factor

    # 绘制中心虚线
    plt.plot(path_x, path_y, c=colors[i], lw=2, ls='--', zorder=10,
             label=f'Route {i + 1} (w={gmm.weights_[i]:.2f})')

    # 绘制航道 (填充区域)
    # 这里简单的用 fill_betweenx 或者 fill 模拟
    # 为了绘制任意曲线的管道，我们需要构造多边形顶点
    # 简单做法：分别画 x 轴和 y 轴的不确定性，或者只画垂直于路径的宽度
    # 这里演示：直接填充 (x+sx, y+sy) 到 (x-sx, y-sy) 的区域

    # 构造多边形顶点：先正序走上边界，再逆序走下边界
    vertices_x = np.concatenate([upper_x, lower_x[::-1]])
    vertices_y = np.concatenate([upper_y, lower_y[::-1]])

    plt.fill(vertices_x, vertices_y, c=colors[i], alpha=0.2, zorder=5, label=f'Corridor {i + 1}')

plt.title(f"Generated Flight Corridors (Mean ± 1.5$\sigma$)\nK={K_RETRIEVE}, N={N_CLUSTERS}")
plt.legend(loc='best')
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")

save_path = f'{SAVE_DIR}/gmm_corridors.png'
plt.savefig(save_path)
print(
    f"✅ 成功！\n   - 灰色细线：原始检索轨迹\n   - 彩色虚线：聚类中心\n   - 彩色色块：基于方差拓宽的航道\n   - 图片保存至: {save_path}")