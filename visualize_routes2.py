import torch, os, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from model.utils import TrajectoryDataset, TrajectoryDataset_RAG, seq_collate_with_padding
from model.Rag_embedder import TimeSeriesEmbedder

# === 配置 ===
K_RETRIEVE = 30
N_CLUSTERS = 3
TOTAL_CONTEXT_SAMPLES = 10  # 【新增】我们总共想在图上画多少条额外的上下文航线
SAVE_DIR = '/data/gjcl/lsy/trajairnet_led/traj_vis'

# 1. 加载 (保持不变)
root = os.getcwd()
loader = DataLoader(TrajectoryDataset(f'{root}/dataset/7days1/processed_data/train', 11, 120, 10), batch_size=1,
                    collate_fn=seq_collate_with_padding, shuffle=True)
rag = TrajectoryDataset_RAG(f'{root}/dataset/rag_file_7days2', 11, 120, 10).rag_system

# 2. 数据 & 检索 (保持不变)
obs = next(iter(loader))[0][0, 0].detach().cpu().numpy().transpose(1, 0)
res = rag.search_batch(TimeSeriesEmbedder().embed_batch(obs[None, :, :]).astype(np.float32), k=K_RETRIEVE)[0]
raw = np.array(
    [np.array(r['pred_data']).T if np.array(r['pred_data']).shape[0] == 3 else np.array(r['pred_data']) for r in res])

# --- 坐标归一化 ---
# 1. 计算检索轨迹的起点 (K, 1, 3)
start_points = raw[:, 0:1, :]
# 2. 减去起点，得到相对位移形状
relative_raw = raw - start_points
# 展平用于聚类 (K, 36)
flat_relative_raw = relative_raw.reshape(K_RETRIEVE, -1)

# 3. 对“形状”进行 GMM 聚类
gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag', random_state=42).fit(flat_relative_raw)
relative_routes = gmm.means_.reshape(N_CLUSTERS, 12, 3)
cluster_labels = gmm.predict(flat_relative_raw)  # 获取每一条检索轨迹属于哪个簇

# 4. 【新增核心逻辑】自适应上下文采样
# 我们将根据 cluster 的权重，决定每个 cluster 采样多少条邻居
context_trajs = []  # 存放选出来的上下文轨迹
context_colors = []  # 存放对应的颜色

colors = ['r', 'b', 'orange']  # 预定义颜色

for i in range(N_CLUSTERS):
    # A. 获取属于当前簇的所有轨迹索引
    indices_in_cluster = np.where(cluster_labels == i)[0]
    count_in_cluster = len(indices_in_cluster)

    if count_in_cluster == 0: continue

    # B. 自适应计算采样数量
    # 逻辑：总采样数 * 该簇的权重。权重越大，采得越多。
    # 至少采 0 条，至多采该簇实际拥有的数量
    n_samples = int(np.round(gmm.weights_[i] * TOTAL_CONTEXT_SAMPLES))
    n_samples = min(n_samples, count_in_cluster)

    # 如果你想让稀疏的簇至少有一条，可以取消注释下面这行：
    # n_samples = max(1, n_samples) if count_in_cluster > 0 else 0

    print(
        f"簇 {i} ({colors[i]}): 权重 {gmm.weights_[i]:.2f}, 包含 {count_in_cluster} 条, 采样 {n_samples} 条作为上下文")

    if n_samples > 0:
        # C. 选择策略：选距离簇中心最近的几条（代表性最强）
        # 计算该簇内所有点到中心的距离
        cluster_data = flat_relative_raw[indices_in_cluster]
        center = gmm.means_[i]
        dists = np.linalg.norm(cluster_data - center, axis=1)

        # 获取距离最近的前 n_samples 个索引
        nearest_indices = np.argsort(dists)[:n_samples]
        original_indices = indices_in_cluster[nearest_indices]

        # D. 保存选中的轨迹和对应的颜色
        for idx in original_indices:
            context_trajs.append(relative_raw[idx])  # 注意这里存的是相对形状
            context_colors.append(colors[i])

context_trajs = np.array(context_trajs) if len(context_trajs) > 0 else np.empty((0, 12, 3))

# 5. 还原到当前观测位置 (Re-centering)
current_end_pos = obs[-1]
absolute_routes = relative_routes + current_end_pos  # 聚类中心还原
if len(context_trajs) > 0:
    absolute_context = context_trajs + current_end_pos  # 上下文轨迹还原

# ------------------------------------------

# 6. 绘图
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
plt.figure(figsize=(8, 8));
plt.grid(True)

# 画历史
plt.plot(obs[:, 0], obs[:, 1], 'g-o', lw=3, zorder=10, label='Observed History')
plt.scatter(obs[-1, 0], obs[-1, 1], c='k', s=100, zorder=11)

# 画背景（淡灰色，所有检索结果，可选）
# for r in relative_raw:
#     line = np.concatenate([obs[-1:], r + current_end_pos], axis=0)
#     plt.plot(line[:, 0], line[:, 1], 'gray', alpha=0.05, zorder=1)

# 【新增】画自适应采样的上下文轨迹 (半透明，同色)
if len(context_trajs) > 0:
    for i, traj in enumerate(absolute_context):
        line = np.concatenate([obs[-1:], traj], axis=0)
        plt.plot(line[:, 0], line[:, 1], c=context_colors[i], lw=1, alpha=0.4, zorder=2)

# 画聚类中心 (实线，深色，最显著)
for i, (route, c) in enumerate(zip(absolute_routes, colors)):
    line = np.concatenate([obs[-1:], route], axis=0)
    plt.plot(line[:, 0], line[:, 1], c=c, lw=3, ls='--', zorder=5, label=f'Cluster {i + 1} (w={gmm.weights_[i]:.2f})')

plt.title(f"Adaptive Context Retrieval\nK={K_RETRIEVE}, Context Samples={len(context_trajs)}")
plt.legend()
save_path = f'{SAVE_DIR}/gmm_adaptive_k{K_RETRIEVE}.png'
plt.savefig(save_path)
print(f"✅ 成功！自适应采样绘图完成。\n图片保存至: {save_path}")