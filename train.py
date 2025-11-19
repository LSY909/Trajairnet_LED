
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
    # import pydevd_pycharm
    # pydevd_pycharm.settrace(
    #     'localhost',
    #     port=9022,
    #     suspend=True
    # )

    ##Dataset params
    # 创建参数解析器，用于解析命令行参数
    parser=argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    # parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--dataset_name',type=str,default='111_days')
    #parser.add_argument('--dataset_name',type=str,default='7days1_small')
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
    parser.add_argument('--total_epochs',type=int, default=50)
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


    # 解析命令行参数
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    ##Select device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##Load test and train data加载数据
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Train Data from ",datapath + "train")
    dataset_train = TrajectoryDataset(datapath + "train", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)

    print("Loading Test Data from ",datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)
    # 初始化RAG（检索增强生成）系统
    rag = TrajectoryDataset_RAG("./dataset/rag_file_7days2", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim).rag_system

    loader_train = DataLoader(dataset_train,batch_size=16,num_workers=4,shuffle=True,collate_fn=seq_collate_with_padding)
    loader_test = DataLoader(dataset_test,batch_size=16,num_workers=4,shuffle=True,collate_fn=seq_collate_with_padding)


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

            obs_traj , pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch
            # 这个也要改，改为在之前就检索，或者之后检索
            all_obs_traj_search_results = []
            all_pred_traj_search_results = []

            # for bs in range(obs_traj.shape[0]):
            #     # embedding
            #     obs_traj_embed_batch = embedder.embed_batch(np.transpose(obs_traj[bs].cpu().numpy(),(0, 2, 1)))
            #     pred_traj_embed_batch = embedder.embed_batch(np.transpose(pred_traj[bs].cpu().numpy(),(0, 2, 1)))
            #     # retriver
            #     obs_traj_search_results = rag.search_batch(obs_traj_embed_batch.astype(np.float32))
            #     pred_traj_search_results = rag.search_batch(pred_traj_embed_batch.astype(np.float32))
            #
            #     all_obs_traj_search_results.append(obs_traj_search_results)
            #     all_pred_traj_search_results.append(pred_traj_search_results)
            # 检索
            for bs in range(obs_traj.shape[0]):
                emb=embedder.embed(obs_traj[bs].detach().cpu().numpy())
                all_obs_traj_search_results.append(rag.search_batch(emb.astype(np.float32),k=20))
            # 聚类
            raw = np.array([[np.array(i['pred_data']).T if np.array(i['pred_data']).shape[0] == 3 else np.array(
                i['pred_data']) for i in a] for s in all_obs_traj_search_results for a in s])
            rel = raw - raw[:, :, 0:1, :]
            ctrs = np.array(
                [GaussianMixture(3, 'diag', random_state=0).fit(r.reshape(20, -1)).means_.reshape(3, 12, 3) for r in
                 rel])
            route_priors = torch.tensor(
                ctrs.reshape(*obs_traj.shape[:2], 3, 12, 3) + obs_traj.detach().cpu().numpy()[:, :, -1, None,
                                                              None]).float().to(device)


            num_agents = obs_traj.shape[1]
            adj = torch.ones((num_agents,num_agents))

            # optimizer.zero_grad()
            # recon_y,m,var = model(torch.transpose(obs_traj,1,2),pred_traj, adj[0],torch.transpose(context,1,2),
            #                       obs_traj_search_results, pred_traj_search_results)
            # loss = 0
            #
            # for agent in range(num_agents):
            #     loss += loss_func(recon_y[agent],torch.transpose(pred_traj[:,:,agent],0,1).unsqueeze(0),m[agent],var[agent])



            # ####################################DiffusionLOSS######################
            '''这里也要改'''
            # optimizer.zero_grad()
            # loss_dist, loss_uncertainty = model(obs_traj,pred_traj, adj[0],torch.transpose(context,1,2),
            #                       all_obs_traj_search_results, all_pred_traj_search_results)
            optimizer.zero_grad()
            loss_dist, loss_uncertainty = model(obs_traj, pred_traj, adj[0], context.transpose(1, 2), [], [],
                                        route_priors=route_priors)

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


