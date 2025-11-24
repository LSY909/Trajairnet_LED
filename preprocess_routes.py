import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn.mixture import GaussianMixture

# 引入项目模块
from model.utils import TrajectoryDataset, TrajectoryDataset_RAG, seq_collate_with_padding
from model.Rag_embedder import TimeSeriesEmbedder


def get_route_priors_offline(obs_traj, rag_system, embedder, k_retrieve, n_clusters):
    """
    核心计算逻辑
    """
    # 1. 维度调整 (Batch, Agent, Time, Dim)
    # 确保是 Time-Last 用于 embedder (N, T, D)
    if obs_traj.shape[2] == 3:  # (B, A, 3, 11) -> 转为 (B, A, 11, 3)
        obs_traj = obs_traj.permute(0, 1, 3, 2)

    # 展平为 (N_samples, Time, Dim)
    batch, agent, time, dim = obs_traj.shape
    flat_obs = obs_traj.reshape(-1, time, dim).numpy().astype(np.float32)
    n_total_agents = flat_obs.shape[0]

    # 2. 检索
    query_emb = embedder.embed_batch(flat_obs)
    search_res = rag_system.search_batch(query_emb, k=k_retrieve)

    # 3. 准备输出容器
    # Shape: (N_total_agents, N_Clusters, 12, 3)
    relative_routes = np.zeros((n_total_agents, n_clusters, 12, dim), dtype=np.float32)

    # 4. 逐个 Agent 进行聚类 (GMM)
    for i, items in enumerate(search_res):
        # 提取 raw data (K, T, D)
        raw = np.array(
            [np.array(item['pred_data']).T if np.array(item['pred_data']).shape[0] == 3 else np.array(item['pred_data'])
             for item in items])

        # 归一化: 减去起点
        start_point = raw[:, 0:1, :]
        rel = raw - start_point

        # 聚类 (Flatten -> GMM -> Reshape)
        traj_data = rel.reshape(k_retrieve, -1)

        try:
            # covariance_type='diag' 对应你模型中的设置，速度较快且稳定
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=0)
            gmm.fit(traj_data)
            # 获取均值中心并还原形状 (3, 12, 3)
            relative_routes[i] = gmm.means_.reshape(n_clusters, 12, dim)
        except:
            # 极少数情况如果聚类失败(如数据全0)，保持为0
            pass

    return relative_routes


def run_pre_processing(args):
    root = os.getcwd()
    # 根据你的目录结构，RAG库在 dataset/rag_file_7days2 或 similar
    rag_path = os.path.join(root, 'dataset', args.rag_dir)

    print(f"Initializing RAG System from {rag_path}...")
    # 注意：这里只初始化 RAG 系统用于查询，不需要重新构建索引
    rag_dataset = TrajectoryDataset_RAG(rag_path, obs_len=args.obs, pred_len=args.preds, step=args.preds_step)
    rag_system = rag_dataset.rag_system
    embedder = TimeSeriesEmbedder()

    splits = ['train', 'test']

    for split in splits:
        print(f"\n[Processing {split} set]")
        data_path = os.path.join(root, 'dataset', args.dataset_name, 'processed_data', split)

        # 1. 初始化 Dataset
        dataset = TrajectoryDataset(data_path, obs_len=args.obs, pred_len=args.preds, step=args.preds_step)

        # 2. 使用 DataLoader 批量处理
        # 从 dataset.obs_traj 取数据

        # 直接获取所有观测数据
        all_obs_data = dataset.obs_traj  # Tensor (Total_Agents, 11, 3)
        print(f"  Total Agents to process: {all_obs_data.shape[0]}")

        # 分批处理以防内存溢出
        batch_size = 256
        num_samples = all_obs_data.shape[0]
        all_priors_list = []

        for i in tqdm(range(0, num_samples, batch_size)):
            # 取出一个 batch 的 agent (B, 11, 3)
            batch_obs = all_obs_data[i: i + batch_size].unsqueeze(1)  # 增加一个假的 Agent 维度适配函数接口 (B, 1, 11, 3)

            # 计算 (B, 1, 3, 12, 3) -> squeeze -> (B, 3, 12, 3)
            priors = get_route_priors_offline(batch_obs, rag_system, embedder, args.k_retrieve, args.n_clusters)
            priors = priors.reshape(batch_obs.shape[0], args.n_clusters, 12, 3)

            all_priors_list.append(priors)

        full_priors = np.concatenate(all_priors_list, axis=0)

        # 保存
        save_path = os.path.join(root, 'dataset', args.dataset_name, f'{split}_route_priors.npy')
        np.save(save_path, full_priors)
        print(f"✅ Saved {split} priors to {save_path}")
        print(f"   Shape: {full_priors.shape}")

        if full_priors.shape[0] != num_samples:
            print(f"⚠️ WARNING: Size mismatch! Dataset: {num_samples}, Priors: {full_priors.shape[0]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='7days1_small')
    parser.add_argument('--rag_dir', type=str, default='rag_file_7days2')
    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    parser.add_argument('--k_retrieve', type=int, default=20)
    parser.add_argument('--n_clusters', type=int, default=3)

    args = parser.parse_args()
    run_pre_processing(args)