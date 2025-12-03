import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.utils import TrajectoryDataset, seq_collate_with_padding

# === 配置 ===
DATASET_NAME = '7days1'
SET_TYPE = 'train'  # 检查 train 或 test
SAVE_DIR = './traj_vis_check'
CHECK_NUM = 5  # 随机画 5 张图看看


def main():
    root = os.getcwd()
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # 1. 构造路径
    data_path = os.path.join(root, 'dataset', DATASET_NAME, 'processed_data', SET_TYPE)
    priors_path = os.path.join(root, 'dataset', DATASET_NAME, f'{SET_TYPE}_priors.pt')

    print(f"Loading data from {data_path}")
    print(f"Loading priors from {priors_path}")

    # 2. 加载数据集 (带 priors)
    # 注意：必须传入 priors_path
    dataset = TrajectoryDataset(data_path, 11, 120, 10, priors_path=priors_path)

    # 使用 batch_size=1 方便画图
    loader = DataLoader(dataset, batch_size=1, collate_fn=seq_collate_with_padding, shuffle=True)

    print(f"Total samples: {len(dataset)}")
    print(f"Generating {CHECK_NUM} visualization samples...")

    iterator = iter(loader)

    for i in range(CHECK_NUM):
        try:
            batch = next(iterator)
        except StopIteration:
            break

        # 解包 batch (注意：现在有 7 个返回值)
        # obs, pred, obs_rel, pred_rel, context, seq_start, priors
        obs_traj, pred_traj, _, _, _, _, route_priors = batch

        # 取出第一个 Agent 的数据
        # obs: (Batch, Agents, 11, 3) -> (11, 3)
        obs = obs_traj[0, 0].detach().cpu().numpy()
        current_end_pos = obs[-1]  # 当前位置

        # priors: (Batch, Agents, Clusters, 12, 4) -> (Clusters, 12, 4)
        priors = route_priors[0, 0].detach().cpu().numpy()

        # 分离 priors 中的均值和宽度
        # priors 结构: [rel_x, rel_y, rel_z, width]
        rel_routes = priors[:, :, :3]  # (3, 12, 3)
        widths = priors[:, :, 3]  # (3, 12)

        # 还原绝对坐标 (Relative -> Absolute)
        abs_routes = rel_routes + current_end_pos

        # --- 绘图 ---
        plt.figure(figsize=(10, 10), dpi=100)
        plt.grid(True, linestyle=':', alpha=0.6)

        # 1. 画观测历史
        plt.plot(obs[:, 0], obs[:, 1], 'g-o', lw=3, zorder=20, label='Observed')
        plt.scatter(obs[-1, 0], obs[-1, 1], c='k', s=120, zorder=21)

        # 2. 画预处理好的航线 (Priors)
        colors = ['r', 'b', 'orange']
        n_clusters = priors.shape[0]

        for k in range(n_clusters):
            path_x = np.concatenate(([obs[-1, 0]], abs_routes[k, :, 0]))
            path_y = np.concatenate(([obs[-1, 1]], abs_routes[k, :, 1]))

            # 宽度信息
            w = np.concatenate(([0], widths[k, :]))

            # 如果宽度全是0，说明GMM失败或没检索到数据
            if np.sum(w) == 0:
                label_str = f'Route {k + 1} (INVALID/ZERO)'
                ls = ':'
            else:
                label_str = f'Route {k + 1}'
                ls = '--'

            color = colors[k % len(colors)]

            # 画中心线
            plt.plot(path_x, path_y, c=color, lw=2, ls=ls, zorder=10, label=label_str)

            # 画宽度范围 (1.5 sigma)
            bound = 1.5
            plt.fill_between(path_x, path_y - w * bound, path_y + w * bound, color=color, alpha=0.2, zorder=5)

        plt.title(f"Sample {i + 1}: Precomputed Priors Visualization\n(Visualizing content of .pt file)")
        plt.legend()
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")

        save_file = os.path.join(SAVE_DIR, f'check_sample_{i}.png')
        plt.savefig(save_file)
        plt.close()
        print(f"Saved {save_file}")

    print("Done checking.")


if __name__ == '__main__':
    main()