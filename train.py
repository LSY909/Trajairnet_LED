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
#     parser.add_argument('--dataset_folder', type=str, default='/dataset/')
#     parser.add_argument('--dataset_name', type=str, default='111_days')
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
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

from model.trajairnet import TrajAirNet
from model.utils import TrajectoryDataset, seq_collate_with_padding
from test import test
import time


def train():
    parser = argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='111_days')
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

    parser.add_argument('--total_epochs', type=int, default=50)
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # [1] 检查预处理
    train_prior_path = f'./dataset/{args.dataset_name}/train_route_priors.npy'
    test_prior_path = f'./dataset/{args.dataset_name}/test_route_priors.npy'
    if not os.path.exists(train_prior_path):
        print("⏳ Detecting missing priors. Running Offline Pre-processing...")
        os.system(f"python preprocess_routes.py --dataset_name {args.dataset_name}")

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Train Data...")
    dataset_train = TrajectoryDataset(
        datapath + "train", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim,
        priors_path=train_prior_path
    )
    print("Loading Test Data...")
    dataset_test = TrajectoryDataset(
        datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim,
        priors_path=test_prior_path
    )

    # [注意] 保持 Batch Size 32，先看看 AMP 能降多少显存
    BATCH_SIZE = 16
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=8, shuffle=True,
                              collate_fn=seq_collate_with_padding, pin_memory=True)
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=8, shuffle=True,
                             collate_fn=seq_collate_with_padding, pin_memory=True)

    model = TrajAirNet(args)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # [核心优化] 初始化 GradScaler
    scaler = torch.cuda.amp.GradScaler()

    print("Starting Training (with AMP)....")

    for epoch in range(1, args.total_epochs + 1):
        model.train()
        loss_total, loss_dt, loss_dc, count = 0, 0, 0, 0

        for batch in tqdm(loader_train):
            batch = [tensor.to(device) if tensor is not None else None for tensor in batch]
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start, route_priors = batch

            num_agents = obs_traj.shape[1]
            adj = torch.ones((num_agents, num_agents))

            optimizer.zero_grad()

            # [核心优化] 开启混合精度上下文
            with torch.cuda.amp.autocast():
                loss_dist, loss_uncertainty = model(
                    obs_traj, pred_traj, adj[0], context.transpose(1, 2),
                    route_priors=route_priors
                )
                alpha = 100
                loss = loss_dist * alpha + loss_uncertainty

            # [核心优化] 使用 scaler 缩放损失并反向传播
            scaler.scale(loss).backward()

            # Unscale 梯度以便裁剪 (可选但推荐)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.model_initializer.parameters(), 1.)

            # 这里的 step 和 update 替代原来的 optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            count += 1
            loss_total += loss.item()  # 注意：loss 可能是 scaled tensor，但在 autocast 外取 item 通常没问题

        print(f'Epoch: {epoch}\tLoss: {loss_total / count:.6f}')

        if args.save_model:
            model_path = os.getcwd() + args.model_pth + "model_" + args.dataset_name + "_" + str(epoch) + ".pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_total / count,
            }, model_path)

        if epoch % 5 == 0:
            print("Starting Testing....")
            model.eval()
            test_ade_loss, test_fde_loss = test(model, loader_test, device)
            print(f"EPOCH: {epoch} Test ADE: {test_ade_loss} Test FDE: {test_fde_loss}")


if __name__ == '__main__':
    train()