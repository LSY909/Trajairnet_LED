import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.trajairnet import TrajAirNet
from model.utils import TrajectoryDataset, seq_collate_with_padding


def main():
    parser = argparse.ArgumentParser(description='Test TrajAirNet model')
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--tcn_channel_size', type=int, default=256)
    parser.add_argument('--tcn_layers', type=int, default=2)
    parser.add_argument('--tcn_kernels', type=int, default=4)
    parser.add_argument('--num_context_input_c', type=int, default=2)
    parser.add_argument('--num_context_output_c', type=int, default=7)
    parser.add_argument('--cnn_kernels', type=int, default=2)
    parser.add_argument('--gat_heads', type=int, default=16)
    parser.add_argument('--graph_hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--cvae_hidden', type=int, default=128)
    parser.add_argument('--cvae_channel_size', type=int, default=128)
    parser.add_argument('--cvae_layers', type=int, default=2)
    parser.add_argument('--mlp_layer', type=int, default=32)
    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--model_dir', type=str, default="/saved_models/")

    '''删除diffusion model 参数，添加QCNet参数'''
    # # diffusion model参数
    # parser.add_argument('--k', type=int, default=7)
    # parser.add_argument('--num_samples', type=int, default=20)
    # parser.add_argument('--traj_dim', type=int, default=3)
    # parser.add_argument('--agent_num', type=int, default=3)

    # QCNet / RAG 参数
    parser.add_argument('--k_retrieve', type=int, default=100)
    parser.add_argument('--n_clusters', type=int, default=3)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    test_priors_path = os.getcwd() + f"/dataset/{args.dataset_name}/test_priors.pt"

    print("Loading Test Data...")
    dataset_test = TrajectoryDataset(
        datapath + "test",
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim,
        priors_path=test_priors_path
    )

    loader_test = DataLoader(
        dataset_test,
        batch_size=8,
        num_workers=4,
        shuffle=False,
        collate_fn=seq_collate_with_padding
    )

    # 加载模型
    model = TrajAirNet(args).to(device)
    model_path = os.path.join(os.getcwd() + args.model_dir + f"model_{args.dataset_name}_{args.epoch}.pt")
    print(f"Loading model from {model_path}")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Model not found at {model_path}, using random weights (for debugging only).")

    # 测试
    print("Starting test...")
    test_ade_loss, test_fde_loss = test(model, loader_test, device)
    print("Test ADE Loss: ", test_ade_loss, "Test FDE Loss: ", test_fde_loss)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def test(model, loader_test, device):
    tot_ade_loss = 0
    tot_fde_loss = 0
    tot_batch = 0
    debug_printed = False
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader_test, desc="Testing"):
            tot_batch += 1
            # 只有当 tensor 不是 None 时才移动到 device，否则保持为 None
            batch = [tensor.to(device) if tensor is not None else None for tensor in batch]

            # 解包
            obs_traj, pred_traj, _, _, context, _, route_priors = batch

            batch_size = obs_traj.shape[0]
            num_agents = obs_traj.shape[1]
            adj = torch.ones((num_agents, num_agents)).to(device)

            # 推理: [B*A, K, T, 3]
            recon_y_flat = model.inference(
                obs_traj,
                pred_traj,
                adj[0],
                context.transpose(1, 2),
                route_priors=route_priors
            )

            # 恢复形状: [Batch, Agents, K, T,3]
            K = recon_y_flat.shape[1]
            T_pred = recon_y_flat.shape[2]
            recon_y = recon_y_flat.view(batch_size, num_agents, K, T_pred, 3)

            batch_ade = 0
            batch_fde = 0

            for b in range(batch_size):
                for a in range(num_agents):
                    # Ground Truth: [T_gt, 3]
                    gt = pred_traj[b, a, :, :3].cpu().numpy()

                    # 如果 Ground Truth 全是 0 (或者极小)，说明这可能是 Padding 数据
                    if np.sum(np.abs(gt)) < 1e-4:
                        continue

                    # Predictions: [K, T_pred, 3]
                    preds = recon_y[b, a].cpu().numpy()

                    # [关键修复] 长度对齐
                    min_len = min(gt.shape[0], preds.shape[1])
                    if min_len == 0: continue

                    gt = gt[:min_len]  # [min_len, 3]
                    preds = preds[:, :min_len]  # [K, min_len, 3]

                    # [DEBUG] 打印第一条数据看看数值范围
                    if not debug_printed:
                        print(f"\n[DEBUG] Ground Truth Sample (First 3 pts):\n{gt[:3]}")
                        print(f"[DEBUG] Prediction Sample (First mode, First 3 pts):\n{preds[0, :3]}")
                        print(f"[DEBUG] Difference:\n{preds[0, :3] - gt[:3]}")
                        debug_printed = True

                    # 计算 K 个模态与 GT 的距离
                    # diff: [K, min_len, 3]
                    diff = preds - gt[None, :, :]
                    l2_dist = np.linalg.norm(diff, axis=-1)  # [K, min_len]

                    ade_k = np.mean(l2_dist, axis=1)  # [K]
                    fde_k = l2_dist[:, -1]  # [K]

                    # 取 minADE / minFDE
                    batch_ade += np.min(ade_k)
                    batch_fde += np.min(fde_k)

            tot_ade_loss += batch_ade / (batch_size * num_agents)
            tot_fde_loss += batch_fde / (batch_size * num_agents)

    return tot_ade_loss / tot_batch, tot_fde_loss / tot_batch


if __name__ == '__main__':
    main()