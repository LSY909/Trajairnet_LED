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
    # 参数设置
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
    parser.add_argument('--k_retrieve', type=int, default=20)  # 离线预处理使用的检索数
    parser.add_argument('--n_clusters', type=int, default=3)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # [步骤 1] 检查离线预处理文件
    # ==========================================
    # 定义预处理文件路径
    train_priors_path = f'./dataset/{args.dataset_name}/train_route_priors.npy'
    test_priors_path = f'./dataset/{args.dataset_name}/test_route_priors.npy'

    # 如果文件不存在，自动运行预处理脚本
    if not os.path.exists(train_priors_path):
        print(f"⏳ Route priors not found at {train_priors_path}")
        print("   Running offline preprocessing (this happens only once)...")
        # 调用我们刚才写的 preprocess_routes.py
        exit_code = os.system(
            f"python preprocess_routes.py --dataset_name {args.dataset_name} --k_retrieve {args.k_retrieve} --n_clusters {args.n_clusters}")
        if exit_code != 0:
            raise RuntimeError("Preprocessing failed!")

    # ==========================================
    # [步骤 2] 加载数据 (传入 priors_path)
    # ==========================================
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    print("Loading Train Data...")
    dataset_train = TrajectoryDataset(
        datapath + "train",
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim,
        priors_path=train_priors_path  # <--- 传入预处理文件路径
    )

    print("Loading Test Data...")
    dataset_test = TrajectoryDataset(
        datapath + "test",
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim,
        priors_path=test_priors_path  # <--- 传入预处理文件路径
    )

    # [优化] 增大 Batch Size (例如 64) 以利用 GPU 性能
    BATCH_SIZE = 64
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True,
                              collate_fn=seq_collate_with_padding)
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=4, shuffle=True,
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
            batch = [tensor.to(device) if tensor is not None else None for tensor in batch]

            # [步骤 3] 解包数据 (现在包含 route_priors)
            # utils.py 中的 collate_fn 会返回 7 个元素
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start, route_priors = batch

            num_agents = obs_traj.shape[1]
            adj = torch.ones((num_agents, num_agents))

            optimizer.zero_grad()

            # [步骤 4] 调用模型 (直接传入 route_priors)
            # 不再需要 rag_system 和 embedder
            loss_dist, loss_uncertainty = model(
                obs_traj,
                pred_traj,
                adj[0],
                context.transpose(1, 2),
                route_priors=route_priors  # <--- 直接使用离线数据
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
            loss = loss_total / count
            model_path = os.getcwd() + args.model_pth + "model_" + args.dataset_name + "_" + str(epoch) + ".pt"
            print("Saving model at", model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_path)

        if epoch % 5 == 0:
            print("Starting Testing....")
            model.eval()
            # test 函数也需要同步修改为接收 route_priors (离线模式)
            test_ade_loss, test_fde_loss = test(model, loader_test, device)
            print("EPOCH: ", epoch, "Train Loss: ", loss, "Test ADE Loss: ", test_ade_loss, "Test FDE Loss: ",
                  test_fde_loss)


if __name__ == '__main__':
    train()