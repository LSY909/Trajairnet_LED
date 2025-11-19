from datetime import datetime
import random
import torch
from sklearn.cluster import KMeans
from torch import nn

from model.tcn_model import TemporalConvNet
from model.batch_gat import GAT
from model.cvae_base import CVAE
from model.utils import acc_to_abs, DotDict
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import pdb

from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel

## 去噪步数
NUM_Tau = 5
class TrajAirNet(nn.Module):
    def __init__(self, args):
        super(TrajAirNet, self).__init__()

        input_size = args.input_channels
        n_classes = int(args.preds / args.preds_step)
        num_channels = [args.tcn_channel_size] * args.tcn_layers
        num_channels.append(n_classes)
        tcn_kernel_size = args.tcn_kernels
        dropout = args.dropout

        graph_hidden = args.graph_hidden

        # 原始参数
        # gat_in = n_classes*args.obs
        # gat_out = n_classes*args.obs

        gat_in = n_classes * args.obs + n_classes ** 2
        gat_out = n_classes * args.obs + n_classes ** 2

        n_heads = args.gat_heads
        alpha = args.alpha

        cvae_encoder = [n_classes * n_classes]
        for layer in range(args.cvae_layers):
            cvae_encoder.append(args.cvae_channel_size)
        cvae_decoder = [args.cvae_channel_size] * args.cvae_layers
        cvae_decoder.append(input_size * args.mlp_layer)

        self.tcn_encoder_x = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
        self.tcn_encoder_similarest = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size,
                                                      dropout=dropout)



        self.tcn_encoder_y = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
        self.tcn_encoder_search = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
        self.cvae = CVAE(encoder_layer_sizes=cvae_encoder, latent_size=args.cvae_hidden,
                         decoder_layer_sizes=cvae_decoder, conditional=True, num_labels=gat_out + gat_in)
        # 原始的GAT和批处理的GAT
        # self.gat = GAT(nin=gat_in, nhid=graph_hidden, nout=gat_out, alpha=alpha, nheads=n_heads)
        self.gat = GAT(in_feature=gat_in, hidden_feature=graph_hidden, out_feature=gat_out,
                       attention_layers=3,dropout=0.1, alpha=alpha)

        self.linear_decoder = nn.Linear(args.mlp_layer, n_classes)

        cfg = DotDict({'scheduler': 'ddim', 'steps': 10, 'beta_start': 1.e-4, 'beta_end': 5.e-2, 'beta_schedule': 'linear',
                       'k': args.k, 's': args.num_samples})
        # self.diffuison = CoreDenoisingModel(cfg=cfg).cuda()
        # eps_theta = self.model(cur_y, beta, x, mask)

        self.context_conv = nn.Conv1d(in_channels=5, out_channels=4, kernel_size=args.cnn_kernels)
        self.context_linear = nn.Linear(11, args.num_context_output_c)

        self.relu = nn.ReLU()
        self.init_weights()

        self.k = args.k
        self.s = args.num_samples

        self.mlp = FutureDistributionAggregator(future_steps=5, h=12, w=12, hidden_dim=256)

        # 这个是借鉴MID的代码部分内容
        self.model = CoreDenoisingModel().cuda()
        self.model_initializer = InitializationModel(t_h=args.obs, d_h=3, t_f=n_classes, d_f=3, k_pred=20).cuda()
        self.betas = self.make_beta_schedule(
            ## 总扩散步数：100
            schedule='linear', n_timesteps=100,
            start=1.e-4, end=5.e-2).cuda()

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        # '''这里是聚类的数量'''
        # n_clusters = 3
        # self.kmeans = KMeans(n_clusters=n_clusters)

    def init_weights(self):
        self.linear_decoder.weight.data.normal_(0, 0.05)
        self.context_linear.weight.data.normal_(0, 0.05)
        self.context_conv.weight.data.normal_(0, 0.1)


    # def calculate_mask(self, obs_traj):
    #     static_dist = 0.03
    #     if obs_traj.size(0) <= 2:
    #         mask = (obs_traj[:, -1] - obs_traj[:, -2]).div(1).norm(p=2, dim=-1) > static_dist
    #     else:
    #         mask = (obs_traj[:, -1] - obs_traj[:, -3]).div(2).norm(p=2, dim=-1) > static_dist
    #     return mask

    # 根据目标最后时刻的距离计算他们之家的边权重
    def calculate_mask(self, obs_traj):
        relation_dist = 8
        last_points = obs_traj[:, -1, :]
        diff = last_points[:, None, :] - last_points[None, :, :]
        dist_matrix = diff.norm(p=2, dim=-1)
        adj = F.softmax(-dist_matrix/1, dim=0)
        mask = dist_matrix < relation_dist
        # mask.fill_diagonal_(False)    # 用处不大
        return adj,mask


    def forward(self, x, y, adj, context, obs_traj_search_results, pred_traj_search_results, sort=False):

        batch_size = x.shape[0]
        agent_num = x.shape[1]
        topk = 5

        '''
        1、找到每个batch所检索的各个最相似的，比如x的维度是 256 * 7 * 3 * 11，那么找完之后的维度应该是 256* 7 * 5【最相似】 *3 * 11
        2、考虑每个检索结果如何融合，256 * 7 * 5 * 3 *11，冲突消解？通过聚类
        3、每个场景都可视化以下看看结果，是不是存在冲突的情况
        '''

        '''
        1、拿到检索的数据进行聚类

        '''
        # var_num = x.shape[2]
        # ob_len = x.shape[3]
        # topk = obs_traj_search_results.shape[2]
        # pre_len = obs_traj_search_results.shape[3]
        #
        # # 聚类
        # obs_traj_search_results = torch.FloatTensor(obs_traj_search_results).to(x.device)
        # searched_for_cluster = obs_traj_search_results.clone()
        # end_points = torch.reshape(searched_for_cluster, (batch_size * agent_num, topk, pre_len, var_num))
        # end_points = end_points[:, :, -1, :]
        # centers = self.kmeans(end_points).centers
        # centers_embed = self.centers_proj(centers)
        # centers_embed_list = torch.split(centers_embed, 1, dim=1)
        # 对观测进行编码
        # x1 = torch.reshape(x, (batch_size * agent_num, var_num, ob_len))

        '''到这里聚类的中心已经拿到了'''

        # # # 把每个agent最相似的结果拿到
        # batch_input_searched_traj = []
        # batch_pred_searched_traj = []
        # for i in range(batch_size):
        #     # 每个场景，有agent_num个目标
        #     scene = x[i].clone()        # 1 * 7 * 3 * 11
        #     searched_scene = obs_traj_search_results[i]
        #     scene_y = y[i].detach().clone()
        #     # # 可视化场景,以检索的为主
        #     # plt.figure(figsize=(10, 8))
        #     # colors = ["Red", "Blue", "Green", "Orange", "Purple","Magenta","Cyan"]
        #     scene_input_searched_input_traj = []
        #     scene_pred_searched_input_traj = []
        #     for agent in range(agent_num):
        #         # 单个目标的轨迹和检索结果
        #         # input_x = scene[agent].clone().cpu().numpy()
        #         input_x = scene[agent].clone().cpu().numpy().transpose(1,0)
        #         searched_x = searched_scene[agent]
        #         # 检索最相似的未来轨迹
        #         distance = []
        #         all_searched_input_traj = []
        #         all_searched_future_traj = []
        #         # 用于存储满足角度条件的轨迹
        #         index = []
        #         for k in range(20):
        #             searched_input_traj = np.array(searched_x[k]["data"])
        #             traj1_dir = input_x[-1, :2] - input_x[0, :2]
        #             traj2_dir = searched_input_traj[-1, :2] - searched_input_traj[0, :2]
        #             cos_sim = np.dot(traj1_dir, traj2_dir) / (
        #                     np.linalg.norm(traj1_dir) * np.linalg.norm(traj2_dir)
        #             )
        #             if cos_sim > 0.9:
        #                 # 检索距离分数
        #                 distance.append(searched_x[k]["distance"])
        #                 # 检索最相似的输入
        #                 searched_input_traj = torch.tensor(searched_input_traj).to(x.device)
        #                 all_searched_input_traj.append(searched_input_traj)
        #                 # 检索最相似的预测
        #                 searched_future_traj = np.array(searched_x[k]["pred_data"])
        #                 searched_future_traj = torch.tensor(searched_future_traj).to(x.device)
        #                 all_searched_future_traj.append(searched_future_traj)
        #                 index.append(k)
        #             else:
        #                 continue
        #             # 可能为0
        #             if len(all_searched_future_traj) == 5:
        #                 break
        #
        #         if len(all_searched_future_traj) < 5:
        #             for extra in range (5 - len(all_searched_future_traj)):
        #                 searched_future_traj = np.array(searched_x[extra]["pred_data"])
        #                 searched_future_traj = torch.tensor(searched_future_traj).to(x.device)
        #                 all_searched_future_traj.append(searched_future_traj)
        #         scene_pred_searched_input_traj.append(torch.stack(all_searched_future_traj))
        #     # 暂时只使用检测的未来数据
        #     batch_pred_searched_traj.append(torch.stack(scene_pred_searched_input_traj))
        # batch_pred_searched = torch.stack(batch_pred_searched_traj)
        # batch_pred_searched = torch.reshape(batch_pred_searched, (batch_size * agent_num * topk,
        #                                                           batch_pred_searched.shape[3],
        #                                                           batch_pred_searched.shape[4]))
        # batch_pred_searched = batch_pred_searched.permute(0,2,1)
        # batch_pred_encoded = self.tcn_encoder_similarest(batch_pred_searched.to(torch.float32))
        # batch_pred_encoded = torch.reshape(batch_pred_encoded, (batch_size,
        #                                                         agent_num,
        #                                                         topk,
        #                                                         batch_pred_encoded.shape[1],
        #                                                         batch_pred_encoded.shape[2]))
        # f_agg = self.mlp(batch_pred_encoded).squeeze()
        #
        # # 调整输入的维度为（batch_size * agent_num） * 变量数 * 轨迹点长度
        # x1 = torch.reshape(x,(batch_size * agent_num, x.shape[2],x.shape[3]))
        # encoded_x = self.tcn_encoder_x(x1)
        # encoded_x = torch.reshape(encoded_x,(batch_size, agent_num, encoded_x.shape[1],encoded_x.shape[2]))
        # f = torch.cat((encoded_x, f_agg),dim=-1)
        # # gat_out的维度是 32 * 7 * 27
        # gat_output = self.gat(torch.reshape(f,(batch_size,agent_num,f.shape[2] * f.shape[3])),adj.to(x.device))


        # 后面仿照LED方法调整数据

        #pdb.set_trace()
        fut_traj = torch.reshape(y,(batch_size * agent_num, y.shape[2],y.shape[3]))
        fut_traj = fut_traj.permute(0,2,1)
        past_traj = torch.reshape(x,(batch_size * agent_num, x.shape[2],x.shape[3]))
        past_traj = past_traj.permute(0,2,1)
        traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).cuda()
        for i in range(batch_size):
            traj_mask[i * agent_num:(i + 1) * agent_num, i * agent_num:(i + 1) * agent_num] = 1.
        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
        # 对应论文框架图相乘部分内容
        sample_prediction = torch.exp(variance_estimation / 2)[
                                ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:,
                                                                       None, None, None]
        # 对应论文相加部分
        loc = sample_prediction + mean_estimation[:, None]
        generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
        #pdb.set_trace()
        loss_dist = (generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=-1).min(dim=1)[0].mean()
        loss_uncertainty = (torch.exp(-variance_estimation)
                            *
                            (generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2))
                            +
                            variance_estimation
                            ).mean()

        return loss_dist, loss_uncertainty


    def inference(self, x, y, adj, context, obs_traj_search_results, pred_traj_search_results, ):
        # 智能体数量
        batch_size = x.shape[0]
        agent_num = x.shape[1]
        #topk = 5

        '''
        1、找到每个batch所检索的各个最相似的，比如x的维度是 256 * 7 * 3 * 11，那么找完之后的维度应该是 256* 7 * 5【最相似】 *3 * 11
        2、考虑每个检索结果如何融合，256 * 7 * 5 * 3 *11，冲突消解？通过聚类
        3、每个场景都可视化以下看看结果，是不是存在冲突的情况

        '''
        # 后面仿照LED方法调整数据
        fut_traj = torch.reshape(y, (batch_size * agent_num, y.shape[2], y.shape[3]))
        fut_traj = fut_traj.permute(0, 2, 1)
        past_traj = torch.reshape(x, (batch_size * agent_num, x.shape[2], x.shape[3]))
        past_traj = past_traj.permute(0, 2, 1)
        traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).cuda()
        for i in range(batch_size):
            traj_mask[i * agent_num:(i + 1) * agent_num, i * agent_num:(i + 1) * agent_num] = 1.
        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask)
        # 对应论文框架图相乘部分内容
        sample_prediction = torch.exp(variance_estimation / 2)[
                                ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:,
                                                                       None, None, None]
        # 对应论文相加部分
        loc = sample_prediction + mean_estimation[:, None]

        # generated_y的维度是(batch_size * agent) * 20 * 12 * 3
        generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
        return generated_y


    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def make_beta_schedule(self, schedule: str = 'linear',
                           n_timesteps: int = 1000,
                           start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
        '''
        Make beta schedule.

        Parameters
        ----
        schedule: str, in ['linear', 'quad', 'sigmoid'],
        n_timesteps: int, diffusion steps,
        start: float, beta start, `start<end`,
        end: float, beta end,

        Returns
        ----
        betas: Tensor with the shape of (n_timesteps)

        '''
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas
    def noise_estimation_loss(self, x, y_0, mask):
        batch_size = x.shape[0]
        # Select a random step for each example
        t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,)).to(x.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size]
        # x0 multiplier
        a = self.extract(self.alphas_bar_sqrt, t, y_0)
        beta = self.extract(self.betas, t, y_0)
        # eps multiplier
        am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
        e = torch.randn_like(y_0)
        # model input
        y = y_0 * a + e * am1
        output = self.model(y, beta, x, mask)
        # batch_size, 20, 2
        return (e - output).square().mean()

    def p_sample(self, x, mask, cur_y, t):
        if t == 0:
            z = torch.zeros_like(cur_y).to(x.device)
        else:
            z = torch.randn_like(cur_y).to(x.device)
        t = torch.tensor([t]).cuda()
        # Factor to the model output
        eps_factor = (
                    (1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
        # Model output
        beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
        eps_theta = self.model(cur_y, beta, x, mask)
        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        # Generate z
        z = torch.randn_like(cur_y).to(x.device)
        # Fixed sigma
        sigma_t = self.extract(self.betas, t, cur_y).sqrt()
        sample = mean + sigma_t * z
        return (sample)

    def p_sample_accelerate(self, x, mask, cur_y, t):
        if t == 0:
            z = torch.zeros_like(cur_y).to(x.device)
        else:
            z = torch.randn_like(cur_y).to(x.device)
        t = torch.tensor([t]).cuda()
        # Factor to the model output
        eps_factor = (
                    (1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
        # Model output
        beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
        eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask)
        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        # Generate z
        z = torch.randn_like(cur_y).to(x.device)
        # Fixed sigma
        sigma_t = self.extract(self.betas, t, cur_y).sqrt()
        sample = mean + sigma_t * z * 0.00001
        return (sample)

    def p_sample_loop(self, x, mask, shape):
        self.model.eval()
        prediction_total = torch.Tensor().cuda()
        for _ in range(20):
            cur_y = torch.randn(shape).to(x.device)
            for i in reversed(range(self.n_steps)):
                cur_y = self.p_sample(x, mask, cur_y, i)
            prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
        return prediction_total

    def p_sample_loop_mean(self, x, mask, loc):
        prediction_total = torch.Tensor().cuda()
        for loc_i in range(1):
            cur_y = loc
            for i in reversed(range(NUM_Tau)):
                cur_y = self.p_sample(x, mask, cur_y, i)
            prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
        return prediction_total

    def p_sample_loop_accelerate(self, x, mask, loc):
        '''
        Batch operation to accelerate the denoising process.

        x: [11, 10, 6]
        mask: [11, 11]
        cur_y: [11, 10, 20, 2]
        '''
        #pdb.set_trace()
        prediction_total = torch.Tensor().cuda()
        cur_y = loc[:, :10]
        for i in reversed(range(NUM_Tau)):
            cur_y = self.p_sample_accelerate(x, mask, cur_y, i)
        cur_y_ = loc[:, 10:]
        for i in reversed(range(NUM_Tau)):
            cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i)
        # shape: B=b*n, K=10, T, 2
        prediction_total = torch.cat((cur_y_, cur_y), dim=1)
        return prediction_total


class FutureDistributionAggregator(nn.Module):
    def __init__(self, future_steps=5, h=12, w=12, hidden_dim=256):
        super().__init__()
        self.in_dim = future_steps * h * w   # 5*12*12=720
        self.out_dim = h * w                 # 12*12=144

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim)
        )

    def forward(self, x):
        # x: (B, C, F, H, W) = (256, 7, 5, 12, 12)
        b, c, f, h, w = x.shape
        x = x.view(b * c, -1)      # (256*7, 5*12*12)
        x = self.mlp(x)            # (256*7, 12*12)
        x = x.view(b, c, 1, h, w) # (256, 7, 1, 12, 12)
        return x


