import torch, os, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from model.utils import TrajectoryDataset, TrajectoryDataset_RAG, seq_collate_with_padding
from model.Rag_embedder import TimeSeriesEmbedder

# === 配置 ===
K_RETRIEVE = 50
N_CLUSTERS = 3
SAVE_DIR = '/data/gjcl/lsy/trajairnet_led/traj_vis'

# 1. 加载
root = os.getcwd()
loader = DataLoader(TrajectoryDataset(f'{root}/dataset/7days1/processed_data/train', 11, 120, 10), batch_size=1, collate_fn=seq_collate_with_padding, shuffle=True)
rag = TrajectoryDataset_RAG(f'{root}/dataset/rag_file_7days2', 11, 120, 10).rag_system

# 2. 数据 & 检索
obs = next(iter(loader))[0][0, 0].detach().cpu().numpy().transpose(1, 0) # (11, 3)
res = rag.search_batch(TimeSeriesEmbedder().embed_batch(obs[None, :, :]).astype(np.float32), k=K_RETRIEVE)[0]
raw = np.array([np.array(r['pred_data']).T if np.array(r['pred_data']).shape[0]==3 else np.array(r['pred_data']) for r in res])

# --- 坐标归一化 ---
# 1. 计算检索轨迹的起点 (K, 1, 3)
start_points = raw[:, 0:1, :]
# 2. 减去起点，得到相对位移形状 (所有轨迹都从 0,0,0 开始)
relative_raw = raw - start_points

# 3. 对“形状”进行 GMM 聚类
gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag', random_state=42).fit(relative_raw.reshape(K_RETRIEVE, -1))
relative_routes = gmm.means_.reshape(N_CLUSTERS, 12, 3) # 得到的中心也是相对形状

# 4. 还原到当前观测位置 (Re-centering)
# 将相对形状加上当前观测的最后一点，拼接到当前飞机后面
current_end_pos = obs[-1] # (3,)
absolute_routes = relative_routes + current_end_pos
# ------------------------------------------

# 5. 绘图
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
plt.figure(figsize=(8, 8)); plt.grid(True)

# 画历史
plt.plot(obs[:, 0], obs[:, 1], 'g-o', lw=3, zorder=5, label='Observed History')
plt.scatter(obs[-1, 0], obs[-1, 1], c='k', s=100, zorder=6) # 当前位置

# 画聚类航线 (彩色粗线)
colors = ['r', 'b', 'orange']
for i, (route, c) in enumerate(zip(absolute_routes, colors)):
    # 拼接以便画图连贯
    line = np.concatenate([obs[-1:], route], axis=0)
    plt.plot(line[:, 0], line[:, 1], c=c, lw=3, ls='--', label=f'Cluster {i+1} (w={gmm.weights_[i]:.2f})')


# 展示聚类效果，把原始检索结果也“平移”过来作为背景
for r in relative_raw:
    line = np.concatenate([obs[-1:], r + current_end_pos], axis=0)
    plt.plot(line[:, 0], line[:, 1], 'gray', alpha=0.05)

plt.title(f"Normalized GMM Clustering\nK={K_RETRIEVE}, N={N_CLUSTERS}")
plt.legend()
save_path = f'{SAVE_DIR}/gmm_normalized_k{K_RETRIEVE}.png'
plt.savefig(save_path)
print(f"✅ 成功！航线已归一化并对齐。\n图片保存至: {save_path}")