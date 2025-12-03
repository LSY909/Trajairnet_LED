import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.Rag import TimeSeriesRAG
from model.Rag_embedder import TimeSeriesEmbedder
from model.utils import TrajectoryDataset, get_batch_route_priors


# ==========================================
# 1. 可视化函数：验证检索和聚类效果
# ==========================================
def plot_rag_results(obs, retrieved_trajs, priors, agent_idx, save_dir):
    """
    绘制观测轨迹、检索到的参考轨迹、以及生成的 Priors。
    关键：将所有轨迹的起点平移到 (0,0) 以对比形状。
    """
    plt.figure(figsize=(10, 8))

    # --- A. 观测轨迹 (蓝色) ---
    obs_np = obs.cpu().numpy()  # (Time, 3) or (Time, 2)
    # 取前两维 (x, y)
    obs_xy = obs_np[:, :2]
    start_pt = obs_xy[0]  # 以起点为基准

    plt.plot(obs_xy[:, 0] - start_pt[0], obs_xy[:, 1] - start_pt[1],
             color='blue', linewidth=3, label='Observation (Query)', marker='.', zorder=10)

    # --- B. 检索到的 RAG 轨迹 (绿色细线) ---
    # retrieved_trajs 是 list of numpy arrays
    for i, traj in enumerate(retrieved_trajs):
        # traj: (Time, D)
        traj_xy = traj[:, :2]
        traj_start = traj_xy[0]

        # 平移归零，只看形状
        plt.plot(traj_xy[:, 0] - traj_start[0], traj_xy[:, 1] - traj_start[1],
                 color='green', alpha=0.15, linewidth=1)

    # --- C. 聚类生成的 Priors (红色粗线) ---
    # priors: (Clusters, Time, 4) -> x, y, ...
    priors_np = priors.cpu().numpy()
    for i in range(priors_np.shape[0]):
        p_traj = priors_np[i]
        p_xy = p_traj[:, :2]
        p_start = p_xy[0]

        plt.plot(p_xy[:, 0] - p_start[0], p_xy[:, 1] - p_start[1],
                 color='red', linewidth=3, label=f'Cluster {i + 1}', linestyle='--', zorder=20)

    plt.title(f"Agent {agent_idx} Retrieval Analysis (Relative Shape)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    plt.axis('equal')  # 保持几何比例

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_agent_{agent_idx}.png")
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  -> Saved visualization to {save_path}")


# ==========================================
# 2. 主流程
# ==========================================
def preprocess():
    parser = argparse.ArgumentParser()
    # 数据集设置
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--set_type', type=str, default='train', help='train or test')
    # RAG 设置 (注意指向 build_rag_db 生成的文件夹)
    parser.add_argument('--rag_dir', type=str, default='7days1_rag')

    # 参数设置
    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    parser.add_argument('--k_retrieve', type=int, default=100, help='检索多少条相似轨迹')
    parser.add_argument('--n_clusters', type=int, default=3, help='生成多少个模态(Priors)')

    args = parser.parse_args()

    # 1. 路径设置
    root = os.getcwd()
    data_path = os.path.join(root, 'dataset', args.dataset_name, 'processed_data', args.set_type)
    # 指向 rag_index (不需要加 .index 后缀，faiss 会自动处理，或者代码里处理)
    rag_base_path = os.path.join(root, 'dataset', args.rag_dir, 'rag_index')
    save_path = os.path.join(root, 'dataset', args.dataset_name, f'{args.set_type}_priors.pt')
    vis_dir = os.path.join(root, 'vis_results')

    print(f"=== Preprocess & Priors Generation ===")
    print(f"Data Source: {data_path}")
    print(f"RAG Source:  {rag_base_path}")

    # 2. 加载 RAG 系统
    # 我们需要先初始化一个空的 RAG，然后 load 数据
    print("Loading RAG system...")
    try:
        # 这里先随便给个 dim，load 的时候会被覆盖
        rag = TimeSeriesRAG(embedding_dim=260)
        rag.load(rag_base_path)
        print(f"RAG Loaded. Documents: {len(rag.documents)}, Dim: {rag.embedding_dim}")
    except Exception as e:
        print(f"Error loading RAG from {rag_base_path}: {e}")
        print("Please ensure you ran 'build_rag_db.py' successfully.")
        return

    # 初始化 Embedder
    embedder = TimeSeriesEmbedder(target_length=256)

    # 3. 加载数据集
    print(f"Loading {args.set_type} dataset...")
    dataset = TrajectoryDataset(data_path, obs_len=args.obs, pred_len=args.preds, step=args.preds_step)

    def simple_collate(batch):
        return batch

    loader = DataLoader(dataset, batch_size=1, collate_fn=simple_collate, shuffle=False)

    all_priors = []
    pred_len_steps = int(args.preds / args.preds_step)

    # 计数器
    vis_count = 0
    MAX_VIS = 10  # 限制可视化数量

    print(f"Computing Priors for {len(dataset)} sequences...")

    # 4. 循环计算
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            # Batch[0][0] shape: (Agents, Time, D)
            # 可能是 (Agents, 11, 3) 或 (Agents, 3, 11)
            obs_traj = batch[0][0].cuda()

            # --- 维度修正 (自动转置) ---
            if obs_traj.shape[-1] != 3 and obs_traj.shape[-1] != 2:
                # 假设时间维在最后，转置为 (Agents, Time, D)
                if obs_traj.shape[-1] == args.obs:
                    obs_traj = obs_traj.transpose(1, 2)

            # get_batch_route_priors 需要 (Batch, Agents, Obs_Len, 3)
            # 我们这里 Batch=1，所以 unsqueeze(0)
            obs_traj_batch = obs_traj.unsqueeze(0)

            # 计算 Priors (核心函数)
            priors = get_batch_route_priors(
                obs_traj_batch,
                rag_system=rag,
                embedder=embedder,
                k_retrieve=args.k_retrieve,
                n_clusters=args.n_clusters,
                pred_len=pred_len_steps,
                device='cuda'
            )
            # priors: (1, Agents, Clusters, 12, 4)

            # 保存结果
            all_priors.append(priors.squeeze(0).cpu())

            # -----------------------------------------------------------
            # 可视化部分 (Verification)
            # -----------------------------------------------------------
            if vis_count < MAX_VIS:
                # 1. 准备 Query Embedding
                # 确保输入是 CPU numpy, shape (Agents, Time, D)
                agents_obs_np = obs_traj.cpu().numpy()
                query_emb = embedder.embed_batch(agents_obs_np).astype(np.float32)

                # 2. 检索
                # results_list: List[List[Dict]] (每个 Agent 对应一列结果)
                results_list = rag.search_batch(query_emb, k=args.k_retrieve)

                # 3. 遍历当前 Batch 的 Agent 画图
                num_agents = obs_traj.shape[0]
                for i in range(num_agents):
                    if vis_count >= MAX_VIS: break

                    # 提取检索到的未来轨迹 (Ground Truth of neighbors)
                    retrieved_raw = []
                    for res in results_list[i]:
                        r_data = np.array(res.get('pred_data', res['data']))

                        if r_data.shape[-1] != 2 and r_data.shape[-1] != 3:
                            r_data = r_data.T
                        retrieved_raw.append(r_data)

                    # 提取计算好的 Priors
                    # (Agents, Clusters, 12, 4)
                    agent_priors = priors.squeeze(0)[i]

                    # 画图
                    plot_rag_results(obs_traj[i], retrieved_raw, agent_priors, vis_count, vis_dir)
                    vis_count += 1

    # 5. 合并并保存
    final_tensor = torch.cat(all_priors, dim=0)  # (Total_Agents, Clusters, 12, 4)

    print(f"Saving priors to {save_path}")
    print(f"Final Tensor Shape: {final_tensor.shape}")
    torch.save(final_tensor, save_path)
    print("Done!")


if __name__ == '__main__':
    preprocess()