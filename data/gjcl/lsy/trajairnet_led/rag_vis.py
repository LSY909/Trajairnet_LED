import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from model.utils import TrajectoryDataset  # 确保这里能引用到你的 dataset 类


def visualize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='7days2')
    parser.add_argument('--set_type', type=str, default='train')
    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=10, help='随机可视化的样本数量')
    parser.add_argument('--save_dir', type=str, default='./vis_results', help='保存图片的文件夹')
    args = parser.parse_args()

    # 1. 路径设置 (需与 preprocess 保持一致)
    root = os.getcwd()
    data_path = os.path.join(root, 'dataset', args.dataset_name, 'processed_data', args.set_type)
    priors_path = os.path.join(root, 'dataset', args.dataset_name, f'{args.set_type}_priors.pt')

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. 加载数据
    print(f"Loading Dataset from {data_path}...")
    dataset = TrajectoryDataset(data_path, obs_len=args.obs, pred_len=args.preds, step=args.preds_step)

    print(f"Loading Priors from {priors_path}...")
    if not os.path.exists(priors_path):
        raise FileNotFoundError(f"Priors file not found at {priors_path}. Please run preprocess.py first.")

    # 加载计算好的 Priors Tensor: (Total_Agents, Clusters, Pred_Steps, 4)
    all_priors = torch.load(priors_path)
    print(f"Priors Shape: {all_priors.shape}")

    # 3. 准备遍历 (与 preprocess 逻辑保持一致以对齐索引)
    # 注意：preprocess 中使用了 flatten 逻辑 (cat dim=0)，所以我们需要维护一个全局索引
    def simple_collate(batch):
        return batch

    loader = DataLoader(dataset, batch_size=1, collate_fn=simple_collate, shuffle=False)

    global_agent_idx = 0
    vis_count = 0

    # 随机选择要画的索引，或者画前 N 个
    # 这里为了演示，我们遍历数据，随机抽取样本进行绘制
    total_agents_in_dataset = all_priors.shape[0]
    indices_to_plot = set(np.random.choice(total_agents_in_dataset, args.num_samples, replace=False))

    print(f"Starting visualization...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # 获取原始数据
            # 假设 TrajectoryDataset 返回的是 (obs, pred_gt, ...) 或者是 list of agents
            # 根据 preprocess 代码推断: batch[0][0] 是 obs, 形状 (Agents, Obs_Len, 3)
            # 假设 batch[0][1] 是 ground truth (GT), 形状 (Agents, Pred_Len, 3)

            # 注意：这里的 batch[0] 是因为 batch_size=1 且 collate 是 simple_collate
            obs_traj = batch[0][0]  # (Agents, Obs_Len, 3)

            # 尝试获取 GT，如果 dataset 返回结构不同，请根据实际情况修改
            try:
                gt_traj = batch[0][1]  # (Agents, Pred_Len, 3)
            except:
                gt_traj = None  # 如果没有 GT

            num_agents_in_scene = obs_traj.shape[0]

            for i in range(num_agents_in_scene):
                # 检查当前 agent 是否被选中要画图
                if global_agent_idx in indices_to_plot:
                    # 获取当前 Agent 的数据
                    curr_obs = obs_traj[i].cpu().numpy()  # (Obs_Len, 3)
                    curr_priors = all_priors[global_agent_idx].cpu().numpy()  # (Clusters, Steps, 4)
                    curr_gt = gt_traj[i].cpu().numpy() if gt_traj is not None else None

                    # 绘图
                    plot_one_agent(
                        curr_obs,
                        curr_priors,
                        curr_gt,
                        agent_id=global_agent_idx,
                        save_path=os.path.join(args.save_dir, f'agent_{global_agent_idx}.png')
                    )
                    vis_count += 1

                # 更新全局索引
                global_agent_idx += 1

            if vis_count >= args.num_samples:
                break

    print(f"Visualization saved to {args.save_dir}")


def plot_one_agent(obs, priors, gt, agent_id, save_path):
    """
    绘制单个 Agent 的轨迹图 (2D 平面: x, y)
    """
    plt.figure(figsize=(10, 8))

    # 1. 绘制历史观测 (Observation) - 蓝色实线
    # 假设数据格式是 (x, y, z) 或 (lat, lon, alt)，我们取前两维
    plt.plot(obs[:, 0], obs[:, 1], 'b-o', label='Observation (Past)', linewidth=2, markersize=4)
    plt.scatter(obs[-1, 0], obs[-1, 1], color='blue', s=100, marker='*', zorder=5)  # 标记当前位置

    # 2. 绘制 Ground Truth (真实未来) - 绿色实线
    if gt is not None:
        # 如果 GT 的长度和 Priors 步长不一致，需要注意对齐，这里直接画
        plt.plot(gt[:, 0], gt[:, 1], 'g-o', label='Ground Truth', linewidth=2, markersize=4, alpha=0.6)

    # 3. 绘制 Priors (RAG 检索出的聚类中心) - 红色虚线
    # priors shape: (Clusters, Steps, 4)
    num_clusters = priors.shape[0]
    for k in range(num_clusters):
        cluster_traj = priors[k]  # (Steps, 4)

        # 为了美观，把 Obs 的最后一个点作为 Prior 的起点连起来
        start_pt = obs[-1, :2]
        prior_x = np.concatenate(([start_pt[0]], cluster_traj[:, 0]))
        prior_y = np.concatenate(([start_pt[1]], cluster_traj[:, 1]))

        plt.plot(prior_x, prior_y, 'r--', alpha=0.5, linewidth=1.5, label=f'Prior Cluster {k + 1}' if k == 0 else "")
        plt.scatter(cluster_traj[-1, 0], cluster_traj[-1, 1], color='red', s=20, marker='x')  # 终点

    plt.title(f'Trajectory Visualization (Agent ID: {agent_id})')
    plt.xlabel('X / Longitude')
    plt.ylabel('Y / Latitude')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')  # 保持比例一致，这对轨迹很重要

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    visualize()