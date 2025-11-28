# import argparse
# import os
# from datetime import datetime
# import numpy as np
# from tqdm import tqdm
# import torch
# from torch.utils.data import DataLoader
# from torch import optim
# import time
# from test import test
#
# from model.trajairnet import TrajAirNet
# from model.utils import TrajectoryDataset, seq_collate_with_padding
#
#
# def train():
#     parser = argparse.ArgumentParser(description='Train TrajAirNet model')
#     parser.add_argument('--dataset_folder', type=str, default='/dataset/')
#
#     # [配置] 数据集与RAG库
#     parser.add_argument('--dataset_name', type=str, default='7days1')
#     parser.add_argument('--rag_dir', type=str, default='111days_rag')
#
#     # 轨迹参数
#     parser.add_argument('--obs', type=int, default=11)
#     parser.add_argument('--preds', type=int, default=120)
#     parser.add_argument('--preds_step', type=int, default=10)
#
#     # 网络参数
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
#     # 训练参数
#     parser.add_argument('--total_epochs', type=int, default=10)
#     parser.add_argument('--delim', type=str, default=' ')
#     parser.add_argument('--evaluate', type=bool, default=True)
#     parser.add_argument('--save_model', type=bool, default=True)
#     parser.add_argument('--model_pth', type=str, default="/saved_models/")
#
#     # Diffusion 参数
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
#
#     # 显卡设置
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 1. 加载数据集
#     datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
#     train_priors_path = os.getcwd() + f"/dataset/{args.dataset_name}/train_priors.pt"
#     test_priors_path = os.getcwd() + f"/dataset/{args.dataset_name}/test_priors.pt"
#
#     print(f"Loading Train Data from {datapath}train with priors: {train_priors_path}")
#
#     dataset_train = TrajectoryDataset(
#         datapath + "train",
#         obs_len=args.obs,
#         pred_len=args.preds,
#         step=args.preds_step,
#         delim=args.delim,
#         priors_path=train_priors_path
#     )
#
#
#     dataset_test = TrajectoryDataset(
#         datapath + "test",
#         obs_len=args.obs,
#         pred_len=args.preds,
#         step=args.preds_step,
#         delim=args.delim,
#         priors_path=test_priors_path
#     )
#
#     # 3. DataLoader 配置
#     BATCH_SIZE = 16
#
#     loader_train = DataLoader(
#         dataset_train,
#         batch_size=BATCH_SIZE,
#         num_workers=4,
#         shuffle=True,
#         collate_fn=seq_collate_with_padding
#     )
#
#     loader_test = DataLoader(
#         dataset_test,
#         batch_size=BATCH_SIZE,
#         num_workers=4,
#         shuffle=True,
#         collate_fn=seq_collate_with_padding
#     )
#
#     # 4. 模型初始化
#     model = TrajAirNet(args)
#     model.to(device)
#
#     print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#
#     print("Starting Training (Offline Mode)....")
#
#     # 5. 训练循环
#     for epoch in range(1, args.total_epochs + 1):
#         model.train()
#         loss_total, loss_dt, loss_dc, count = 0, 0, 0, 0
#
#         for batch in tqdm(loader_train):
#             batch = [tensor.to(device) for tensor in batch]
#
#             # 解包 7 个变量
#             obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start, route_priors = batch
#
#             num_agents = obs_traj.shape[1]
#             adj = torch.ones((num_agents, num_agents))
#
#             optimizer.zero_grad()
#
#             loss_dist, loss_uncertainty = model(
#                 obs_traj,
#                 pred_traj,
#                 adj[0],
#                 context.transpose(1, 2),
#                 route_priors=route_priors
#             )
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
#
#
#         print('[{}] Epoch: {}\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
#             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
#             epoch, loss_total / count, loss_dt / count, loss_dc / count))
#
#         if args.save_model:
#             if not os.path.exists(os.getcwd() + args.model_pth):
#                 os.makedirs(os.getcwd() + args.model_pth)
#             model_path = os.getcwd() + args.model_pth + "model_" + args.dataset_name + "_" + str(epoch) + ".pt"
#             print("Saving model at", model_path)
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': loss_total / count,
#             }, model_path)
#
#             if epoch % 1 == 0:
#                 print("Starting Testing....")
#
#                 model.eval()
#                 test_ade_loss, test_fde_loss = test(model, loader_test, device)
#
#                 print("EPOCH: ", epoch, "Train Loss: ", loss, "Test ADE Loss: ", test_ade_loss, "Test FDE Loss: ",
#                       test_fde_loss)
#
#
# if __name__ == '__main__':
#     train()


import argparse
import os
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.trajairnet import TrajAirNet
from model.utils import TrajectoryDataset, seq_collate_with_padding
from test import test


def train():
    parser = argparse.ArgumentParser(description='Train TrajAirNet model (QCNet Style)')
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1')
    parser.add_argument('--rag_dir', type=str, default='111days_rag')  # 仅作路径参考

    # 轨迹参数
    parser.add_argument('--obs', type=int, default=11)
    parser.add_argument('--preds', type=int, default=120)
    parser.add_argument('--preds_step', type=int, default=10)

    # 网络参数
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

    # 训练超参
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--total_epochs', type=int, default=50)
    parser.add_argument('--delim', type=str, default=' ')
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--model_pth', type=str, default="/saved_models/")

    # QCNet / RAG 参数
    parser.add_argument('--k_retrieve', type=int, default=20)
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--traj_dim', type=int, default=3)

    args = parser.parse_args()

    # 设备配置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 数据集路径配置 (离线模式)
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    train_priors_path = os.getcwd() + f"/dataset/{args.dataset_name}/train_priors.pt"
    test_priors_path = os.getcwd() + f"/dataset/{args.dataset_name}/test_priors.pt"

    print(f"Loading Train Data from {datapath}train")
    print(f"Loading Train Priors from {train_priors_path}")

    # 初始化训练集
    dataset_train = TrajectoryDataset(
        datapath + "train",
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim,
        priors_path=train_priors_path
    )

    # 初始化测试集
    dataset_test = TrajectoryDataset(
        datapath + "test",
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim,
        priors_path=test_priors_path
    )

    loader_train = DataLoader(dataset_train, batch_size=16, num_workers=4, shuffle=True,
                              collate_fn=seq_collate_with_padding)
    loader_test = DataLoader(dataset_test, batch_size=16, num_workers=4, shuffle=False,
                             collate_fn=seq_collate_with_padding)

    # 2. 模型初始化
    model = TrajAirNet(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting Training on {device}...")

    # 3. 训练循环
    for epoch in range(1, args.total_epochs + 1):
        model.train()
        loss_total = 0
        loss_reg_sum = 0
        loss_cls_sum = 0
        count = 0

        for batch in tqdm(loader_train, desc=f"Epoch {epoch}"):
            batch = [tensor.to(device) for tensor in batch]

            # 解包 7 个变量
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start, route_priors = batch

            # 构造邻接矩阵
            batch_size = obs_traj.shape[0]
            num_agents = obs_traj.shape[1]
            adj = torch.ones((num_agents, num_agents)).to(device)

            optimizer.zero_grad()

            # 前向传播 (返回 Regression Loss 和 Classification Loss)
            reg_loss, cls_loss = model(
                obs_traj,
                pred_traj,
                adj[0],
                context.transpose(1, 2),
                route_priors=route_priors
            )

            # 总 Loss (可以加权)
            loss = reg_loss + 0.5 * cls_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_total += loss.item()
            loss_reg_sum += reg_loss.item()
            loss_cls_sum += cls_loss.item()
            count += 1

        # 打印日志
        avg_loss = loss_total / count
        avg_reg = loss_reg_sum / count
        avg_cls = loss_cls_sum / count

        print('[{}] Epoch: {}\tLoss: {:.6f} (Reg: {:.6f}, Cls: {:.6f})'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            epoch, avg_loss, avg_reg, avg_cls))

        # 保存模型
        if args.save_model:
            if not os.path.exists(os.getcwd() + args.model_pth):
                os.makedirs(os.getcwd() + args.model_pth)
            model_path = os.getcwd() + args.model_pth + "model_" + args.dataset_name + "_" + str(epoch) + ".pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, model_path)
            print("Saved model to", model_path)

        # 评估
        if args.evaluate and epoch % 1 == 0:
            print("Starting test...")
            # [修改] 只需要传入 model, loader, device
            test_ade, test_fde = test(model, loader_test, device)
            print(f"Epoch {epoch} Result -> ADE: {test_ade:.4f}, FDE: {test_fde:.4f}")


if __name__ == '__main__':
    train()