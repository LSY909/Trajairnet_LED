import argparse
import os
from tqdm import tqdm
import numpy as np
import csv

import torch
from torch.utils.data import DataLoader

from model.trajairnet import TrajAirNet
from model.utils import ade, fde, TrajectoryDataset, seq_collate
from model.utils import TrajectoryDataset, seq_collate, loss_func, TrajectoryDataset_RAG,seq_collate_with_padding
from model.Rag_embedder import TimeSeriesEmbedder
import matplotlib.pyplot as plt
def main():

    parser=argparse.ArgumentParser(description='Test TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')

    parser.add_argument('--epoch',type=int,default=20)

    parser.add_argument('--obs',type=int,default=11)
    parser.add_argument('--preds',type=int,default=120)
    parser.add_argument('--preds_step',type=int,default=10)

    ##Network params
    parser.add_argument('--input_channels',type=int,default=3)
    parser.add_argument('--tcn_channel_size',type=int,default=256)
    parser.add_argument('--tcn_layers',type=int,default=2)
    parser.add_argument('--tcn_kernels',type=int,default=4)

    parser.add_argument('--num_context_input_c',type=int,default=2)
    parser.add_argument('--num_context_output_c',type=int,default=7)
    parser.add_argument('--cnn_kernels',type=int,default=2)

    parser.add_argument('--gat_heads',type=int, default=16)
    parser.add_argument('--graph_hidden',type=int,default=256)
    parser.add_argument('--dropout',type=float,default=0.05)
    parser.add_argument('--alpha',type=float,default=0.2)
    parser.add_argument('--cvae_hidden',type=int,default=128)
    parser.add_argument('--cvae_channel_size',type=int,default=128)
    parser.add_argument('--cvae_layers',type=int,default=2)
    parser.add_argument('--mlp_layer',type=int,default=32)

    parser.add_argument('--delim',type=str,default=' ')

    parser.add_argument('--model_dir', type=str , default="/saved_models/")



    # diffusion model参数
    parser.add_argument('--k', type=int , default=7)
    parser.add_argument('--num_samples', type=int , default=20)
    parser.add_argument('--traj_dim', type=int , default=3)
    parser.add_argument('--agent_num', type=int , default=3)

    # RAG 参数
    parser.add_argument('--k_retrieve', type=int, default=100)
    parser.add_argument('--n_clusters', type=int, default=3)

    args=parser.parse_args()


    ##Select device
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##Load data
    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    print("Loading Test Data from ",datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)
    # loader_test = DataLoader(dataset_test,batch_size=64,num_workers=4,shuffle=True,collate_fn=seq_collate)
    loader_test = DataLoader(dataset_test,batch_size=8,num_workers=4,shuffle=True,collate_fn=seq_collate_with_padding)

    # 初始化 RAG
    print("Initializing RAG System for Testing...")
    rag = TrajectoryDataset_RAG("./dataset/rag_files", obs_len=args.obs, pred_len=args.preds, step=args.preds_step,
                                delim=args.delim).rag_system
    embedder = TimeSeriesEmbedder()  # 新增：初始化 Embedder

    ##Load model
    model = TrajAirNet(args)
    model.to(device)

    model_path = os.path.join(os.getcwd() + args.model_dir + f"model_{args.dataset_name}_{args.epoch}.pt")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])


    test_ade_loss, test_fde_loss = test(model,loader_test,device, rag, embedder)

    print("Test ADE Loss: ",test_ade_loss,"Test FDE Loss: ",test_fde_loss)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))
def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def test(model,loader_test,device,rag, embedder):
    tot_ade_loss = 0
    tot_fde_loss = 0
    tot_batch = 0
    embedder = TimeSeriesEmbedder()
    for batch in tqdm(loader_test):
        tot_batch += 1
        batch = [tensor.to(device) for tensor in batch]
        obs_traj_all , pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, seq_start  = batch
        batch_size = obs_traj_all.shape[0]

        # all_obs_traj_search_results = []
        # all_pred_traj_search_results = []

        num_agents = obs_traj_all.shape[1]
        adj = torch.ones((num_agents, num_agents))

        best_ade_loss = float('inf')
        best_fde_loss = float('inf')


        recon_y_all = model.inference(obs_traj_all, pred_traj_all, adj[0], torch.transpose(context, 1, 2),
                                      rag_system=rag, embedder=embedder)

        recon_y_all = torch.reshape(recon_y_all,(batch_size,
                                                 num_agents,
                                                 recon_y_all.shape[1],
                                                 recon_y_all.shape[2],
                                                 recon_y_all.shape[3]
                                                 ))
        # recon_y_all = recon_y_all.permute(2,1,0)

        ade_loss = 0
        fde_loss = 0

        '''
        绘制预测结果的曲线，根据智能体数量进行绘制
        需要拿到数据：
        所有智能体的观测轨迹
        所有智能体的真实轨迹
        所有智能体的预测轨迹【由于我们是多预测，分别绘制最好的、所有】
        '''
        all_agent_num = 0


        for bs in range(batch_size):
            # Padding的结果不重复计算损失
            scene_ade_loss = 0
            scene_fde_loss = 0

            new_num_agents = 1
            new_obs_traj_all = obs_traj_all[bs].clone().cpu().numpy()
            for dup in range(1,num_agents):
                if new_obs_traj_all[0][0][0] == new_obs_traj_all[dup][0][0]:
                    new_num_agents = dup
                    break
            #all_agent_num+=new_num_agents
            # 绘制整个场景的飞机目标
            plt.figure(figsize=(10, 8), dpi=150)
            ade_list =[]
            fde_list =[]
            for agent in range(new_num_agents):
                # 要记录下每次的观测轨迹有多少，吧padding的去除！

                obs_traj = np.squeeze(obs_traj_all[bs,agent,:,:].cpu().numpy())
                pred_traj = np.squeeze(pred_traj_all[bs,agent,:,:].cpu().numpy())

                # recon_pred的维度 20 12 3
                recon_pred = recon_y_all[bs,agent,:,:,:].detach().cpu().numpy().transpose()

                #最小误差
                min_ade_loss = float('inf')
                min_fde_loss = float('inf')

                n = 0
                for k in range(recon_pred.shape[2]):
                    single_ade_loss = ade(recon_pred[:3,:,k], pred_traj[:3,:])
                    single_fde_loss = fde((recon_pred[:3,:,k]), (pred_traj[:3,:]))
                    # if single_ade_loss <= min_ade_loss and single_fde_loss <= min_fde_loss:
                    if single_ade_loss <= min_ade_loss:
                        min_ade_loss = single_ade_loss
                        min_fde_loss = single_fde_loss
                        n = k

                scene_ade_loss += min_ade_loss
                scene_fde_loss += min_fde_loss

                ade_list.append(min_ade_loss)
                fde_list.append(min_fde_loss)
                # if min_ade_loss <= 0.6:
                recon_pred = recon_pred[:,:,n]
                plt.scatter(obs_traj[0, :], obs_traj[1, :], label='Obe', color='green', s=50, alpha=0.6)
                plt.scatter(pred_traj[0, :], pred_traj[1, :], label='True', color='blue', s=50, alpha=0.6)
                plt.scatter(recon_pred[0, :], recon_pred[1, :], label='Pred', color='red', s=50, alpha=0.6)
                plt.xticks([-2, -1, 0, 1, 2, 3])  # 指定刻度位置
                plt.yticks([-2, -1, 0, 1, 2, 3])
            plt.grid(True)
            plt.legend()
            plt.savefig(f"./images/1009/{new_num_agents}_"
                        f"{sum(ade_list) / len(ade_list)}_"
                        f"{sum(fde_list) / len(fde_list)}.png")
            plt.close()
            # print("Successfully Saving An Image!")

            scene_average_ade_loss = scene_ade_loss/new_num_agents
            scene_average_fde_loss = scene_fde_loss/new_num_agents
            ade_loss += scene_average_ade_loss
            fde_loss += scene_average_fde_loss
        tot_ade_loss += ade_loss/batch_size
        tot_fde_loss += fde_loss/batch_size

    return tot_ade_loss/(tot_batch),tot_fde_loss/(tot_batch)



if __name__=='__main__':
    main()

