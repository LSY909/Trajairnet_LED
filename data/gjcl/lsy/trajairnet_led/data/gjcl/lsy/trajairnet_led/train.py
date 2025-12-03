# import argparse
# import os
# from datetime import datetime
# from sklearn.mixture import GaussianMixture
# import numpy as np
# from tqdm import tqdm
# import torch
# from torch.utils.data import DataLoader
# from torch import optim
#
# from model.Rag_embedder import TimeSeriesEmbedder
# from model.trajairnet import TrajAirNet
# from model.utils import TrajectoryDataset, seq_collate, loss_func, loss_func_MSE, TrajectoryDataset_RAG, \
#     seq_collate_with_padding
# from test import test
#
# import time
#
#
# def train():
#     parser = argparse.ArgumentParser(description='Train TrajAirNet model')
#     # parser.add_argument('--dataset_folder', type=str, default='/dataset/')
#     # parser.add_argument('--dataset_name', type=str, default='111_days')
#
#     # [修改 1] 将默认数据集改为 7days1_small
#     parser.add_argument('--dataset_name', type=str, default='7days1_small')
#     # [新增] 增加 RAG 目录参数，默认为对应的小库
#     parser.add_argument('--rag_dir', type=str, default='7days1_small_rag')
#
#     parser.add_argument('--obs', type=int, default=11)
#     parser.add_argument('--preds', type=int, default=120)
#     parser.add_argument('--preds_step', type=int, default=10)
#
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
#     parser.add_argument('--lr', type=float, default=0.001)
#
#     # parser.add_argument('--total_epochs', type=int, default=50)
#     parser.add_argument('--total_epochs', type=int, default=50)
#     parser.add_argument('--delim', type=str, default=' ')
#     parser.add_argument('--evaluate', type=bool, default=True)
#     parser.add_argument('--save_model', type=bool, default=True)
#     parser.add_argument('--model_pth', type=str, default="/saved_models/")
#
#     parser.add_argument('--k', type=int, default=4)
#     parser.add_argument('--num_samples', type=int, default=15)
#     parser.add_argument('--traj_dim', type=int, default=3)
#     parser.add_argument('--agent_num', type=int, default=3)
#
#     # RAG 参数
#     parser.add_argument('--k_retrieve', type=int, default=20)
#     parser.add_argument('--n_clusters', type=int, default=3)
#
#     args = parser.parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
#
#     print("Loading Train Data from ", datapath + "train")
#     dataset_train = TrajectoryDataset(datapath + "train", obs_len=args.obs, pred_len=args.preds, step=args.preds_step,
#                                       delim=args.delim)
#
#     print("Loading Test Data from ", datapath + "test")
#     dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step,
#                                      delim=args.delim)
#
#     print("Initializing RAG System...")
#     rag = TrajectoryDataset_RAG("./dataset/rag_file_7days2", obs_len=args.obs, pred_len=args.preds,
#                                 step=args.preds_step, delim=args.delim).rag_system
#     embedder = TimeSeriesEmbedder()
#     loader_train = DataLoader(dataset_train, batch_size=16, num_workers=4, shuffle=True,
#                               collate_fn=seq_collate_with_padding)
#     loader_test = DataLoader(dataset_test, batch_size=16, num_workers=4, shuffle=True,
#                              collate_fn=seq_collate_with_padding)
#
#     model = TrajAirNet(args)
#     model.to(device)
#
#     print(f"torch.cuda.is_available:{torch.cuda.is_available()}")
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#
#     print("Starting Training....")
#
#     for epoch in range(1, args.total_epochs + 1):
#         model.train()
#         loss_total, loss_dt, loss_dc, count = 0, 0, 0, 0
#
#         for batch in tqdm(loader_train):
#             batch_count = 1
#             tot_batch_count = 1
#             batch = [tensor.to(device) for tensor in batch]
#
#             obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch
#             num_agents = obs_traj.shape[1]
#             adj = torch.ones((num_agents, num_agents))
#
#             optimizer.zero_grad()
#
#             #  调用模型时，传入 rag 和 embedder
#             loss_dist, loss_uncertainty = model(obs_traj, pred_traj, adj[0], torch.transpose(context, 1, 2),
#                                                 rag_system=rag, embedder=embedder)
#
#             alpha = 100
#             loss = loss_dist * alpha + loss_uncertainty
#
#             loss_total += loss.item()
#             loss_dt += loss_dist.item() * alpha
#             loss_dc += loss_uncertainty.item()
#
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.model_initializer.parameters(), 1.)
#             optimizer.step()
#             count += 1
#             tot_batch_count += 1
#
#         print('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
#             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
#             epoch, loss_total / count, loss_dt / count, loss_dc / count))
#
#         if args.save_model:
#             loss = loss_total / count
#             model_path = os.getcwd() + args.model_pth + "model_" + args.dataset_name + "_" + str(epoch) + ".pt"
#             print("Saving model at", model_path)
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': loss,
#             }, model_path)
#
#         if args.evaluate and epoch % 5 == 0:
#             print("Starting Testing....")
#             model.eval()
#             # 传递 rag 和 embedder 给 test 函数
#             test_ade_loss, test_fde_loss = test(model, loader_test, device, rag, embedder)
#             print("EPOCH: ", epoch, "Train Loss: ", loss_total / count, "Test ADE Loss: ", test_ade_loss,
#                   "Test FDE Loss: ", test_fde_loss)
#
#
# if __name__ == '__main__':
#     train()


import argparse
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim

from model.Rag_embedder import TimeSeriesEmbedder
from model.trajairnet import TrajAirNet
from model.utils import TrajectoryDataset, seq_collate_with_padding, TrajectoryDataset_RAG
from test import test
import time


def train():
    parser = argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')


    ''' 在这里改数据集'''
    # [修改 1] 默认数据集改为 7days1_small
    parser.add_argument('--dataset_name', type=str, default='7days1_small')
    # [修改 2] RAG 目录改为 7days1_small_rag
    parser.add_argument('--rag_dir', type=str, default='7days1_small_rag')


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
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--model_pth', type=str, default="/saved_models/")

    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=15)
    parser.add_argument('--traj_dim', type=int, default=3)
    parser.add_argument('--agent_num', type=int, default=3)

    # RAG 参数
    parser.add_argument('--k_retrieve', type=int, default=20)
    parser.add_argument('--n_clusters', type=int, default=3)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据路径
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Train Data from ", datapath + "train")
    # [修改] 不再传入 priors_path，回退到原始数据加载
    dataset_train = TrajectoryDataset(
        datapath + "train",
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim
    )

    print("Loading Test Data from ", datapath + "test")
    dataset_test = TrajectoryDataset(
        datapath + "test",
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim
    )

    # [恢复] 初始化 RAG 系统 (因为要在线计算)
    rag_path = f"./dataset/{args.rag_dir}"
    print(f"Initializing RAG System from {rag_path}...")
    rag = TrajectoryDataset_RAG(rag_path, obs_len=args.obs, pred_len=args.preds,
                                step=args.preds_step, delim=args.delim).rag_system
    embedder = TimeSeriesEmbedder()

    # DataLoader (小数据集 Batch Size 保持 16 即可，因为在线计算 CPU 压力大)
    loader_train = DataLoader(dataset_train, batch_size=16, num_workers=4, shuffle=True,
                              collate_fn=seq_collate_with_padding)
    loader_test = DataLoader(dataset_test, batch_size=16, num_workers=4, shuffle=True,
                             collate_fn=seq_collate_with_padding)

    model = TrajAirNet(args)
    model.to(device)

    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting Training....")

    for epoch in range(1, args.total_epochs + 1):
        model.train()
        loss_total, loss_dt, loss_dc, count = 0, 0, 0, 0

        for batch in tqdm(loader_train):
            batch = [tensor.to(device) for tensor in batch]

            # 解包数据
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch[:6]  # 只取前6个，忽略可能的None

            num_agents = obs_traj.shape[1]
            adj = torch.ones((num_agents, num_agents))

            optimizer.zero_grad()

            # [核心] 将 rag 和 embedder 传给模型，让模型自己算
            loss_dist, loss_uncertainty = model(
                obs_traj,
                pred_traj,
                adj[0],
                context.transpose(1, 2),
                rag_system=rag,
                embedder=embedder
            )

            alpha = 100
            loss = loss_dist * alpha + loss_uncertainty

            loss_total += loss.item()
            loss_dt += loss_dist.item() * alpha
            loss_dc += loss_uncertainty.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model_initializer.parameters(), 1.)
            optimizer.step()
            count += 1

        print('[{}] Epoch: {}\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            epoch, loss_total / count, loss_dt / count, loss_dc / count))

        if args.save_model:
            if not os.path.exists(os.getcwd() + args.model_pth):
                os.makedirs(os.getcwd() + args.model_pth)
            model_path = os.getcwd() + args.model_pth + "model_" + args.dataset_name + "_" + str(epoch) + ".pt"
            print("Saving model at", model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_total / count,
            }, model_path)

        if epoch % 5 == 0:
            print("Starting Testing....")
            model.eval()
            # 测试也需要传工具
            test_ade_loss, test_fde_loss = test(model, loader_test, device, rag, embedder)
            print("EPOCH: ", epoch, "Train Loss: ", loss_total / count, "Test ADE Loss: ", test_ade_loss,
                  "Test FDE Loss: ", test_fde_loss)


if __name__ == '__main__':
    train()