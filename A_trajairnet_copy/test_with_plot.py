import argparse
import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from model.trajairnet import TrajAirNet
from model.utils import ade, fde, TrajectoryDataset, seq_collate
from model.utils import TrajectoryDataset, seq_collate, loss_func, TrajectoryDataset_RAG
from model.Rag_embedder import TimeSeriesEmbedder
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
import os

# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
def main():
    
    parser=argparse.ArgumentParser(description='Test TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    # parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--dataset_name',type=str,default='7days1_small')
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
    parser.add_argument('--k', type=int , default=4)
    parser.add_argument('--num_samples', type=int , default=15)
    parser.add_argument('--traj_dim', type=int , default=3)
    parser.add_argument('--agent_num', type=int , default=3)

    args=parser.parse_args()


    ##Select device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##Load data

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"

    print("Loading Test Data from ",datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)
    loader_test = DataLoader(dataset_test,batch_size=1,num_workers=4,shuffle=True,collate_fn=seq_collate)

    rag = TrajectoryDataset_RAG("./dataset/rag_files", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim).rag_system
    # rag = TrajectoryDataset_RAG("./dataset/7days1_small_rag", obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim).rag_system

    ##Load model
    model = TrajAirNet(args)
    model.to(device)

    # model_path =  os.getcwd() + args.model_dir + "model_" + args.dataset_name + "_" + "6" + ".pt"
    model_path =  os.getcwd() + args.model_dir + "model_7days1_small_5.pt"
    print(f"Loading model from:{model_path}")


    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_ade_loss, test_fde_loss = test(model,loader_test,device, rag)

    print("Test ADE Loss: ",test_ade_loss,"Test FDE Loss: ",test_fde_loss)

def test(model,loader_test,device,rag):
    tot_ade_loss = 0
    tot_fde_loss = 0
    tot_batch = 0
    embedder = TimeSeriesEmbedder()
    for batch in tqdm(loader_test):
        batch = [tensor.to(device) for tensor in batch]
        obs_traj_all , pred_traj_all, obs_traj_rel_all, pred_traj_rel_all, context, seq_start  = batch

        obs_traj_embed_batch = embedder.embed_batch(np.transpose(obs_traj_all.cpu().numpy(), (1, 0, 2)))
        pred_traj_embed_batch = embedder.embed_batch(np.transpose(pred_traj_all.cpu().numpy(), (1, 0, 2)))
        obs_traj_search_results = rag.search_batch(obs_traj_embed_batch.astype(np.float32))
        pred_traj_search_results = rag.search_batch(pred_traj_embed_batch.astype(np.float32))

        num_agents = obs_traj_all.shape[1]
        best_ade_loss = float('inf')
        best_fde_loss = float('inf')

        # 只考虑特定数量的agent
        if num_agents == 3:
            tot_batch += 1
            pass
        else:
            continue
        
        for i in range(5):
            z = torch.randn([1,1 ,128]).to(device)
            
            adj = torch.ones((num_agents,num_agents))
            recon_y_all = model.inference(torch.transpose(obs_traj_all,1,2),z,adj,torch.transpose(context,1,2),
            obs_traj_search_results, pred_traj_search_results)
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
            plt.figure(figsize=(10, 8), dpi=150)

            for agent in range(num_agents):
                obs_traj = np.squeeze(obs_traj_all[:,agent,:].cpu().numpy())
                pred_traj = np.squeeze(pred_traj_all[:,agent,:].cpu().numpy())
                recon_pred = np.squeeze(recon_y_all[agent].permute(2,1,0).detach().cpu().numpy()).transpose()

                plt.scatter(obs_traj[:,0], obs_traj[:,1], label='观测轨迹', color='blue', s=50, alpha=0.6)
                plt.scatter(pred_traj[:,0], pred_traj[:,1], label='真实轨迹', color='red', s=50, alpha=0.6)
                # ade_loss += ade(recon_pred, pred_traj)
                # fde_loss += fde((recon_pred), (pred_traj))

                ####################找最小误差#######################
                min_ade_loss = float('inf')
                min_fde_loss = float('inf')
                best_pred = None
                for k in range(recon_y_all[agent].shape[2]):
                    single_ade_loss = ade(recon_pred[:,:,k], pred_traj)
                    single_fde_loss = fde((recon_pred[:,:,k]), (pred_traj))
                    if single_ade_loss <= min_ade_loss and single_fde_loss <= min_fde_loss:
                        min_ade_loss = single_ade_loss
                        min_fde_loss = single_fde_loss
                        best_pred = recon_pred[:,:,k]
                        
                plt.scatter(best_pred[:,0], best_pred[:,1], label='预测轨迹', color='green', s=50, alpha=0.6)

                ade_loss += min_ade_loss
                fde_loss += min_fde_loss
                ##################################################

            ade_total_loss = ade_loss/num_agents
            fde_total_loss = fde_loss/num_agents
            if ade_total_loss<best_ade_loss:
                best_ade_loss = ade_total_loss
                best_fde_loss = fde_total_loss

            plt.grid(True)
            plt.legend()
            plt.savefig(f"./images/0814_concat/{ade_total_loss}.png")
            # plt.show()

        tot_ade_loss += best_ade_loss
        tot_fde_loss += best_fde_loss
    # return tot_ade_loss/(tot_batch),tot_fde_loss/(tot_batch)
    return tot_ade_loss/(tot_batch),tot_fde_loss/(tot_batch)


if __name__=='__main__':
    main()

