import argparse
import os
from datetime import datetime
from sklearn.mixture import GaussianMixture
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import optim

from model.Rag_embedder import TimeSeriesEmbedder
from model.trajairnet import TrajAirNet
from model.utils import TrajectoryDataset, seq_collate, loss_func, loss_func_MSE, TrajectoryDataset_RAG, seq_collate_with_padding
from test import test

import time

def train():
    ##Dataset params
    # 创建参数解析器，用于解析命令行参数
    parser=argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')
    # 观测轨迹长度
    parser.add_argument('--obs',type=int,default=11)
    # 预测轨迹长度
    parser.add_argument('--preds',type=int,default=120)
    # 预测轨迹长度的时间步长（默认10）
    parser.add_argument('--preds_step',type=int,default=10)
    # 预测点为120/10=12个



    ##Network params
    # 输入通道数（默认3，分别为x、y、v）
    parser.add_argument('--input_channels',type=int,default=3)
    # TCN层的通道数（默认256）
    parser.add_argument('--tcn_channel_size',type=int,default=256)
    # TCN层的层数（默认2）
    parser.add_argument('--tcn_layers',type=int,default=2)
    # TCN层的卷积核大小（默认4）
    parser.add_argument('--tcn_kernels',type=int,default=4)

    # 上下文输入通道数（默认2，分别为v、d）
    parser.add_argument('--num_context_input_c',type=int,default=2)
    # 上下文输出通道数（默认7，分别为v、d、x、y、v_x、v_y、d_x、d_y）
    parser.add_argument('--num_context_output_c',type=int,default=7)
    # CNN层的卷积核大小（默认2）
    parser.add_argument('--cnn_kernels',type=int,default=2)

    # GAT（图注意力网络）注意力头数
    parser.add_argument('--gat_heads',type=int, default=16)
    # GAT层的隐藏层通道数（默认256）
    parser.add_argument('--graph_hidden',type=int,default=256)
    # Dropout概率（默认0.05）
    parser.add_argument('--dropout',type=float,default=0.05)
    # LeakyReLU负斜率
    parser.add_argument('--alpha',type=float,default=0.2)
    # cave的隐藏层
    parser.add_argument('--cvae_hidden',type=int,default=128)
    # 通道大小
    parser.add_argument('--cvae_channel_size',type=int,default=128)
    parser.add_argument('--cvae_layers',type=int,default=2)
    parser.add_argument('--mlp_layer',type=int,default=32)
    # 学习率
    parser.add_argument('--lr',type=float,default=0.001)


    # 训练总轮次
    parser.add_argument('--total_epochs',type=int, default=20)
    # 数据分隔符
    parser.add_argument('--delim',type=str,default=' ')
    # 在训练过程中是否进行评估
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--model_pth', type=str , default="/saved_models/")


    # diffusion model参数
    # 采样参数
    parser.add_argument('--k', type=int , default=4)
    # 采样数15
    parser.add_argument('--num_samples', type=int , default=15)
    # 轨迹维度=3   B*N*T*C
    parser.add_argument('--traj_dim', type=int , default=3)
    # 智能体数量为3
    parser.add_argument('--agent_num', type=int , default=3)

    '''
    添加的RAG参数
    '''
    # RAG 参数
    parser.add_argument('--k_retrieve', type=int, default=100)
    parser.add_argument('--n_clusters', type=int, default=3)
    # DenseFuturePredictor 先验融合权重（0 表示关闭）
    parser.add_argument('--dense_prior_weight', type=float, default=0.5)
    # 航线先验生成模式：per_agent(最慢/最细), per_batch(快), topk(最快), none(关闭)
    parser.add_argument('--route_prior_mode', type=str, default='per_batch',
                        choices=['per_agent', 'per_batch', 'topk', 'none'])
    # 使用离线预计算的航线先验（会跳过在线检索/聚类，显著提速）
    parser.add_argument('--route_priors_train', type=str, default='')
    parser.add_argument('--route_priors_test', type=str, default='')


    # 解析命令行参数
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Train Data from ",datapath + "train")
    route_priors_train = args.route_priors_train if args.route_priors_train else None
    dataset_train = TrajectoryDataset(
        datapath + "train",
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim,
        route_priors_path=route_priors_train,
    )

    print("Loading Test Data from ",datapath + "test")
    route_priors_test = args.route_priors_test if args.route_priors_test else None
    dataset_test = TrajectoryDataset(
        datapath + "test",
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim,
        route_priors_path=route_priors_test,
    )
    # 初始化RAG（检索增强生成）系统
    #rag = TrajectoryDataset_RAG("./dataset/rag_file_7days2", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim).rag_system
    '''添加的初始化检索系统'''
    rag = None
    embedder = None
    if route_priors_train is None:
        print("Initializing RAG System (online route priors)...")
        rag = TrajectoryDataset_RAG("./dataset/rag_file_7days2", obs_len=args.obs, pred_len=args.preds,
                                    step=args.preds_step, delim=args.delim).rag_system
        embedder = TimeSeriesEmbedder()
    loader_train = DataLoader(
        dataset_train,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        collate_fn=seq_collate_with_padding,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        collate_fn=seq_collate_with_padding,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


    model = TrajAirNet(args)
    model.to(device)

    #Resume继续训练
    print(f"torch.cuda.is_available:{torch.cuda.is_available()}")


    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    num_batches = len(loader_train)

    print("Starting Training....")

    for epoch in range(1, args.total_epochs+1):

        model.train()
        loss_batch = 0
        batch_count = 0
        tot_batch_count = 0
        tot_loss = 0
        embedder = TimeSeriesEmbedder()
        loss_total, loss_dt, loss_dc, count = 0, 0, 0, 0
        for batch in tqdm(loader_train):
            batch_count += 1
            tot_batch_count += 1
            batch = [tensor.to(device) for tensor in batch]

            if len(batch) == 7:
                obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start, route_priors = batch
            else:
                obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch
                route_priors = None
            # 这个也要改，改为在之前就检索，或者之后检索
            all_obs_traj_search_results = []
            all_pred_traj_search_results = []

            num_agents = obs_traj.shape[1]
            adj = torch.ones((num_agents,num_agents))



            # ####################################DiffusionLOSS######################
            optimizer.zero_grad()
            loss_dist, loss_uncertainty = model(
                obs_traj,
                pred_traj,
                adj[0],
                torch.transpose(context, 1, 2),
                route_priors=route_priors,
                rag_system=rag,
                embedder=embedder,
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

            # break

        print('[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            epoch, loss_total / count, loss_dt / count, loss_dc / count))
        if args.save_model:
            loss = tot_loss/tot_batch_count
            model_path = os.getcwd() + args.model_pth + "model_" + args.dataset_name + "_" + str(epoch) + ".pt"
            print("Saving model at",model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, model_path)

        # if args.evaluate:
        if epoch % 5 == 0:
            print("Starting Testing....")

            model.eval()
            test_ade_loss, test_fde_loss = test(model,loader_test,device,rag)

            print("EPOCH: ",epoch,"Train Loss: ",loss,"Test ADE Loss: ",test_ade_loss,"Test FDE Loss: ",test_fde_loss)

if __name__=='__main__':

    train()


