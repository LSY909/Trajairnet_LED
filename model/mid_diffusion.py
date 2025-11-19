import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
import pdb

class VarianceSchedule(Module):

    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

class DiffusionTraj(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):

        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1)

        e_rand = torch.randn_like(x_0).cuda()  # (B, N, d)

        # e_theta是diffusion输出的内容，他的维度应该是 # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)
        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    # 评估的时候会调用该方法，用于生成真实轨迹 point_dim=3最开始的时候是
    ## num_points: 需要生成的轨迹点数量 sample: 采样次数，即生成多少个轨迹样本 bestof: 是否使用随机初始化噪声（True）或零初始化（False）
    ## point_dim: 每个轨迹点的维度，默认为3（可能代表经纬度和高度）
    ## flexibility: 灵活性参数，用于调整采样过程中的噪声量
    ## ret_traj: 是否返回整个采样过程的轨迹（所有中间步骤结果）
    ## sampling: 采样方法，可选"ddpm"（默认）或"ddim"
    ## step: 采样步长，决定了从噪声到干净样本的迭代步数
    def sample(self, num_points, context, sample, bestof, point_dim=3, flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        traj_list = []  # 存储生成的轨迹样本列表
        for i in range(sample):  # 循环生成指定数量的样本
            batch_size = context.size(0)  # 获取批次大小
            if bestof:  # 如果使用随机初始化
                x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)  # 生成标准正态分布的噪声
            else:  # 如果使用零初始化
                x_T = torch.zeros([batch_size, num_points, point_dim]).to(context.device)  # 初始化为全零张量
            traj = {self.var_sched.num_steps: x_T}  # 创建轨迹字典，键为时间步，值为对应状态
            stride = step  # 设置采样步长
            # stride = int(100/stride)  # 被注释掉的备选步长计算方式

            # 从最大时间步开始，逐步回退到时间步0（从噪声到干净样本）
            for t in range(self.var_sched.num_steps, 0, -stride):
                # 生成噪声：对于最后一步不添加噪声
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                # 获取当前时间步的扩散参数
                alpha = self.var_sched.alphas[t]  # 扩散过程的alpha值
                alpha_bar = self.var_sched.alpha_bars[t]  # 扩散过程的累积alpha值
                alpha_bar_next = self.var_sched.alpha_bars[t - stride]  # 下一步的累积alpha值
                # pdb.set_trace()  # 调试断点（已注释）
                sigma = self.var_sched.get_sigmas(t, flexibility)  # 根据时间步和灵活性参数获取sigma值

                # 计算DDPM采样所需的系数
                c0 = 1.0 / torch.sqrt(alpha)  # 系数c0
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)  # 系数c1

                x_t = traj[t]  # 获取当前时间步的状态
                beta = self.var_sched.betas[[t] * batch_size]  # 获取当前时间步的beta值，并扩展到批次大小
                # 使用网络预测噪声
                e_theta = self.net(x_t, beta=beta, context=context)

                # 根据选择的采样方法执行不同的采样步骤
                if sampling == "ddpm":  # 使用DDPM（Denoising Diffusion Probabilistic Models）方法
                    x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                elif sampling == "ddim":  # 使用DDIM（Denoising Diffusion Implicit Models）方法
                    x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * e_theta
                else:
                    pdb.set_trace()  # 其他情况进入调试

                # 保存下一步的状态，并停止梯度传播
                traj[t - stride] = x_next.detach()  # Stop gradient and save trajectory.
                # 将当前状态移至CPU以节省GPU内存
                traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
                # 如果不需要保存完整轨迹，则删除当前时间步的状态
                if not ret_traj:
                    del traj[t]

            # 根据是否需要返回完整轨迹，将最终结果或整个轨迹添加到列表中
            if ret_traj:
                traj_list.append(traj)  # 保存完整轨迹
            else:
                traj_list.append(traj[0])  # 只保存最终结果（时间步0的状态）
        # 将所有样本堆叠成一个张量并返回
        return torch.stack(traj_list)

class TrajNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(2, 128, context_dim+3), ##升维
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3), ##降维
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 2, context_dim+3),

        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        #pdb.set_trace()
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


# 走的这个网络
## 条件信息处理且经历了从2D到3D的升级（注释掉的2D版本 vs 当前3D版本）
class TransformerConcatLinear(Module):

    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual
        ## 位置编码
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        # 原始的维度只用x,y,现在我考虑高度是x,y,z
        # self.concat1 = ConcatSquashLinear(2,2*context_dim,context_dim+3)
        ## 线性连接层
        self.concat1 = ConcatSquashLinear(3,2*context_dim,context_dim+3)
        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        # self.linear = ConcatSquashLinear(context_dim//2, 2, context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, 3, context_dim+3)
        #self.linear = nn.Linear(128,2)


    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
        x = self.concat1(ctx_emb,x)
        final_emb = x.permute(1,0,2)
        final_emb = self.pos_emb(final_emb)


        trans = self.transformer_encoder(final_emb).permute(1,0,2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)

class TransformerLinear(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.residual = residual

        self.pos_emb = PositionalEncoding(d_model=128, dropout=0.1, max_len=24)
        self.y_up = nn.Linear(2, 128)
        self.ctx_up = nn.Linear(context_dim+3, 128)
        self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.linear = nn.Linear(128, point_dim)

    def forward(self, x, beta, context):

        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        ctx_emb = self.ctx_up(ctx_emb)
        emb = self.y_up(x)
        final_emb = torch.cat([ctx_emb, emb], dim=1).permute(1,0,2)
        #pdb.set_trace()
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb)  # 13 * b * 128
        trans = trans[1:].permute(1,0,2)   # B * 12 * 128, drop the first one which is the z
        return self.linear(trans)


## 线性解码器
class LinearDecoder(Module):
    def __init__(self):
            super().__init__()
            self.act = F.leaky_relu
            self.layers = ModuleList([
                #nn.Linear(2, 64),
                nn.Linear(32, 64),
                nn.Linear(64, 128),
                nn.Linear(128, 256),
                nn.Linear(256, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 12)
                #nn.Linear(2, 64),
                #nn.Linear(2, 64),
            ])
    def forward(self, code):

        out = code
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        return out
