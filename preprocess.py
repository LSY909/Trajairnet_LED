import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.utils import TrajectoryDataset, TrajectoryDataset_RAG, get_batch_route_priors
from model.Rag_embedder import TimeSeriesEmbedder


def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--rag_dir', type=str, default='111days_rag')
    parser.add_argument('--set_type', type=str, default='train', help='train or test')

    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    parser.add_argument('--k_retrieve', type=int, default=20)
    parser.add_argument('--n_clusters', type=int, default=3)

    args = parser.parse_args()

    # 1. 路径设置
    root = os.getcwd()
    data_path = os.path.join(root, 'dataset', args.dataset_name, 'processed_data', args.set_type)
    rag_path = os.path.join(root, 'dataset', args.rag_dir)
    save_path = os.path.join(root, 'dataset', args.dataset_name, f'{args.set_type}_priors.pt')

    # 2. 初始化 RAG 和 Embedder
    print(f"Init RAG from {rag_path}...")
    rag = TrajectoryDataset_RAG(rag_path, args.obs, args.preds, args.preds_step).rag_system
    embedder = TimeSeriesEmbedder()

    # 3. 加载数据集
    print(f"Loading {args.set_type} raw data...")
    dataset = TrajectoryDataset(data_path, obs_len=args.obs, pred_len=args.preds, step=args.preds_step)
    def simple_collate(batch):
        return batch
    loader = DataLoader(dataset, batch_size=1, collate_fn=simple_collate, shuffle=False)
    all_priors = []
    print(f"Computing Priors for {len(dataset)} sequences...")
    pred_len_steps = int(args.preds / args.preds_step)
    # 4. 循环计算
    with torch.no_grad():
        for batch in tqdm(loader):
            #  obs: batch[0][0] -> Shape: (Agents, Obs_Len, 3)
            obs_traj = batch[0][0].cuda()
            # get_batch_route_priors 需要 (Batch, Agents, Obs_Len, 3)
            obs_traj_batch = obs_traj.unsqueeze(0)
            priors = get_batch_route_priors(
                obs_traj_batch,
                rag_system=rag,
                embedder=embedder,
                k_retrieve=args.k_retrieve,
                n_clusters=args.n_clusters,
                pred_len=pred_len_steps,
                device='cuda'
            )

            # priors 输出: (1, Agents, Clusters, 12, 4)
            all_priors.append(priors.squeeze(0).cpu())

    # 5. 拼接保存
    # 把所有计算出来的 priors 拼起来
    final_tensor = torch.cat(all_priors, dim=0)  # (Total_Agents_In_Dataset, Clusters, 12, 4)

    print(f"Saving priors to {save_path}")
    print(f"Final Shape: {final_tensor.shape}")
    torch.save(final_tensor, save_path)
    print("Done!")


if __name__ == '__main__':
    preprocess()