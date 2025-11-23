#
# import argparse
# import os
# from tqdm import tqdm
# import numpy as np
# import csv
#
# import torch
# from torch.utils.data import DataLoader
#
# from model.trajairnet import TrajAirNet
# from model.utils import ade, fde, TrajectoryDataset, seq_collate
# from model.utils import TrajectoryDataset, seq_collate, loss_func, TrajectoryDataset_RAG, seq_collate_with_padding
# from model.Rag_embedder import TimeSeriesEmbedder
# import matplotlib.pyplot as plt
#
#
# def main():
#     parser = argparse.ArgumentParser(description='Test TrajAirNet model')
#     parser.add_argument('--dataset_folder', type=str, default='/dataset/')
#     parser.add_argument('--dataset_name', type=str, default='7days1')
#     parser.add_argument('--epoch', type=int, default=20)
#     parser.add_argument('--obs', type=int, default=11)
#     parser.add_argument('--preds', type=int, default=120)
#     parser.add_argument('--preds_step', type=int, default=10)
#
#     ##Network params
#     parser.add_argument('--input_channels', type=int, default=3)
#     parser.add_argument('--tcn_channel_size', type=int, default=256)
#     parser.add_argument('--tcn_layers', type=int, default=2)
#     parser.add_argument('--tcn_kernels', type=int, default=4)
#     parser.add_argument('--num_context_input_c', type=int, default=2)
#     parser.add_argument('--num_context_output_c', type=int, default=7)
#     parser.add_argument('--cnn_kernels', type=int, default=2)
#     parser.add_argument('--gat_heads', type=int, default=16)
#     parser.add_argument('--graph_hidden', type=int, default=256)
#     parser.add_argument('--dropout', type=float, default=0.05)
#     parser.add_argument('--alpha', type=float, default=0.2)
#     parser.add_argument('--cvae_hidden', type=int, default=128)
#     parser.add_argument('--cvae_channel_size', type=int, default=128)
#     parser.add_argument('--cvae_layers', type=int, default=2)
#     parser.add_argument('--mlp_layer', type=int, default=32)
#     parser.add_argument('--delim', type=str, default=' ')
#     parser.add_argument('--model_dir', type=str, default="/saved_models/")
#
#     # diffusion model参数
#     parser.add_argument('--k', type=int, default=4)
#     parser.add_argument('--num_samples', type=int, default=20)
#     parser.add_argument('--traj_dim', type=int, default=3)
#     parser.add_argument('--agent_num', type=int, default=3)
#
#     # RAG 参数
#     parser.add_argument('--k_retrieve', type=int, default=100)
#     parser.add_argument('--n_clusters', type=int, default=3)
#
#     args = parser.parse_args()
#
#     ##Select device
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     ##Load data
#     datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
#     print("Loading Test Data from ", datapath + "test")
#     dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step,
#                                      delim=args.delim)
#     loader_test = DataLoader(dataset_test, batch_size=8, num_workers=4, shuffle=True,
#                              collate_fn=seq_collate_with_padding)
#
#     # 初始化 RAG
#     print("Initializing RAG System for Testing...")
#     rag = TrajectoryDataset_RAG("./dataset/rag_files", obs_len=args.obs, pred_len=args.preds, step=args.preds_step,
#                                 delim=args.delim).rag_system
#     embedder = TimeSeriesEmbedder()
#
#     ##Load model
#     model = TrajAirNet(args)
#     model.to(device)
#
#     model_path = os.path.join(os.getcwd() + args.model_dir + f"model_{args.dataset_name}_{args.epoch}.pt")
#     print(f"Loading model from {model_path}")
#
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#
#     test_ade_loss, test_fde_loss = test(model, loader_test, device, rag, embedder)
#
#     print("Test ADE Loss: ", test_ade_loss, "Test FDE Loss: ", test_fde_loss)
#
#
# # Metrics helpers
# def ade(y1, y2):
#     y1 = np.transpose(y1, (1, 0))
#     y2 = np.transpose(y2, (1, 0))
#     loss = y1 - y2
#     loss = loss ** 2
#     loss = np.sqrt(np.sum(loss, 1))
#     return np.mean(loss)
#
#
# def fde(y1, y2):
#     loss = (y1[:, -1] - y2[:, -1]) ** 2
#     return np.sqrt(np.sum(loss))
#
#
# # [核心修改] test 函数签名，增加 rag 和 embedder
# def test(model, loader_test, device, rag, embedder):
#     tot_ade_loss = 0
#     tot_fde_loss = 0
#     tot_batch = 0
#
#     for batch in tqdm(loader_test):
#         tot_batch += 1
#         batch = [tensor.to(device) for tensor in batch]
#         obs_traj_all, pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, seq_start = batch
#         batch_size = obs_traj_all.shape[0]
#         num_agents = obs_traj_all.shape[1]
#         adj = torch.ones((num_agents, num_agents))
#
#         # [核心修改] 调用 inference 时传入工具
#         recon_y_all = model.inference(
#             obs_traj_all,
#             pred_traj_all,
#             adj[0],
#             torch.transpose(context, 1, 2),
#             rag_system=rag,  # <--- 传入
#             embedder=embedder  # <--- 传入
#         )
#
#         recon_y_all = torch.reshape(recon_y_all, (batch_size,
#                                                   num_agents,
#                                                   recon_y_all.shape[1],
#                                                   recon_y_all.shape[2],
#                                                   recon_y_all.shape[3]
#                                                   ))
#
#         ade_loss = 0
#         fde_loss = 0
#
#         for bs in range(batch_size):
#             scene_ade_loss = 0
#             scene_fde_loss = 0
#
#             # 简单的去除 padding 逻辑
#             new_num_agents = 1
#             new_obs_traj_all = obs_traj_all[bs].clone().cpu().numpy()
#             for dup in range(1, num_agents):
#                 if new_obs_traj_all[0][0][0] == new_obs_traj_all[dup][0][0]:
#                     new_num_agents = dup
#                     break
#
#             for agent in range(new_num_agents):
#                 pred_traj = np.squeeze(pred_traj_all[bs, agent, :, :].cpu().numpy())
#
#                 recon_pred = recon_y_all[bs, agent].detach().cpu().numpy().transpose(2, 1, 0)
#
#                 min_ade_loss = float('inf')
#                 min_fde_loss = float('inf')
#
#                 # 遍历所有采样样本，找最佳
#                 for k in range(recon_pred.shape[2]):
#                     single_ade_loss = ade(recon_pred[:3, :, k], pred_traj[:3, :])
#                     single_fde_loss = fde((recon_pred[:3, :, k]), (pred_traj[:3, :]))
#
#                     if single_ade_loss <= min_ade_loss:
#                         min_ade_loss = single_ade_loss
#                         min_fde_loss = single_fde_loss
#
#                 scene_ade_loss += min_ade_loss
#                 scene_fde_loss += min_fde_loss
#
#             ade_loss += scene_ade_loss / new_num_agents
#             fde_loss += scene_fde_loss / new_num_agents
#
#         tot_ade_loss += ade_loss / batch_size
#         tot_fde_loss += fde_loss / batch_size
#
#     return tot_ade_loss / (tot_batch), tot_fde_loss / (tot_batch)
#
#
# if __name__ == '__main__':
#     main()


import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from model.trajairnet import TrajAirNet
from model.utils import ade, fde, TrajectoryDataset, seq_collate_with_padding


def test(model, loader_test, device):
    tot_ade_loss = 0
    tot_fde_loss = 0
    tot_batch = 0

    for batch in tqdm(loader_test):
        tot_batch += 1
        batch = [tensor.to(device) if tensor is not None else None for tensor in batch]
        # 解包 (包含 route_priors)
        obs_traj_all, pred_traj_all, _, _, context, _, route_priors = batch

        batch_size = obs_traj_all.shape[0]
        num_agents = obs_traj_all.shape[1]
        adj = torch.ones((num_agents, num_agents))

        # 直接传入 route_priors
        recon_y_all = model.inference(
            obs_traj_all,
            pred_traj_all,
            adj[0],
            torch.transpose(context, 1, 2),
            route_priors=route_priors
        )

        recon_y_all = torch.reshape(recon_y_all, (batch_size, num_agents, recon_y_all.shape[1], recon_y_all.shape[2],
                                                  recon_y_all.shape[3]))

        ade_loss = 0
        fde_loss = 0
        for bs in range(batch_size):
            scene_ade_loss = 0
            scene_fde_loss = 0
            new_num_agents = 1
            new_obs_traj_all = obs_traj_all[bs].clone().cpu().numpy()
            for dup in range(1, num_agents):
                if new_obs_traj_all[0][0][0] == new_obs_traj_all[dup][0][0]:
                    new_num_agents = dup
                    break

            for agent in range(new_num_agents):
                pred_traj = np.squeeze(pred_traj_all[bs, agent, :, :].cpu().numpy())
                recon_pred = recon_y_all[bs, agent].detach().cpu().numpy().transpose(2, 1, 0)
                min_ade_loss = float('inf')
                min_fde_loss = float('inf')
                for k in range(recon_pred.shape[2]):
                    single_ade_loss = ade(recon_pred[:3, :, k], pred_traj[:3, :])
                    single_fde_loss = fde((recon_pred[:3, :, k]), (pred_traj[:3, :]))
                    if single_ade_loss <= min_ade_loss:
                        min_ade_loss = single_ade_loss
                        min_fde_loss = single_fde_loss
                scene_ade_loss += min_ade_loss
                scene_fde_loss += min_fde_loss

            ade_loss += scene_ade_loss / new_num_agents
            fde_loss += scene_fde_loss / new_num_agents

        tot_ade_loss += ade_loss / batch_size
        tot_fde_loss += fde_loss / batch_size

    return tot_ade_loss / (tot_batch), tot_fde_loss / (tot_batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='111_days')
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
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--traj_dim', type=int, default=3)
    parser.add_argument('--agent_num', type=int, default=3)
    parser.add_argument('--k_retrieve', type=int, default=100)
    parser.add_argument('--n_clusters', type=int, default=3)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    print("Loading Test Data from ", datapath + "test")

    # [修改] 传入 priors_path
    test_priors_path = f'./dataset/{args.dataset_name}/test_route_priors.npy'
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step,
                                     delim=args.delim, priors_path=test_priors_path)
    loader_test = DataLoader(dataset_test, batch_size=64, num_workers=4, shuffle=True,
                             collate_fn=seq_collate_with_padding)

    model = TrajAirNet(args)
    model.to(device)

    model_path = os.path.join(os.getcwd() + args.model_dir + f"model_{args.dataset_name}_{args.epoch}.pt")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_ade_loss, test_fde_loss = test(model, loader_test, device)
    print("Test ADE Loss: ", test_ade_loss, "Test FDE Loss: ", test_fde_loss)