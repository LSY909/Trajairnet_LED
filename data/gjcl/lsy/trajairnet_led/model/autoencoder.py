import torch
from torch.nn import Module
import torch.nn as nn
from .encoders.trajectron import Trajectron
from .encoders import dynamics as dynamic_module
import models.diffusion as diffusion
from models.diffusion import DiffusionTraj,VarianceSchedule
import pdb

'''
这段代码实现了一个**基于条件扩散模型（Conditional Diffusion Model）**的轨迹预测框架：

编码器（Encoder）： 将历史和社会信息压缩为条件向量。

扩散模型（Diffusion）： 在训练时，利用条件向量指导对真实未来轨迹的去噪过程（学习如何将噪声变回轨迹）；在预测时，从随机噪声开始，逐步去噪生成新的、多样的未来轨迹。

运动学（Dynamics）： 负责在速度和位置之间进行转换，以确保轨迹的物理合理性。

'''




class AutoEncoder(Module):

    def __init__(self, config, encoder):
        super().__init__()
        ## 保存配置对象，包含模型的各种超参数设置
        self.config = config
        ## 保存编码器组件，用于将输入数据编码为潜在表示
        self.encoder = encoder
        self.diffnet = getattr(diffusion, config.diffnet)

        self.diffusion = DiffusionTraj(
            net = self.diffnet(point_dim=3, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'

            )
        )

    def encode(self, batch,node_type):
        z = self.encoder.get_latent(batch, node_type)
        return z
    
    def generate(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        #print(f"Using {sampling}")
        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type)
        # 用于预测，为什么是vel，是因为config里面
        predicted_y_vel =  self.diffusion.sample(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj, sampling=sampling, step=step)
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)


        return predicted_y_pos.cpu().detach().numpy()

    def get_loss(self, batch, node_type):
        # batch拿数据
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch
        # self.encode就是图中的a temporal-social encoder
        # feat_x_encoded经过X提特征后，作为状态信息输入到diffusion中，然后进行加噪去噪
        feat_x_encoded = self.encode(batch,node_type) # B * 64
        loss = self.diffusion.get_loss(y_t.cuda(), feat_x_encoded)
        return loss
