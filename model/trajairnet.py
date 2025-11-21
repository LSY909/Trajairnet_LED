from datetime import datetime
import random
import torch
from sklearn.cluster import KMeans
from torch import nn
from sklearn.mixture import GaussianMixture  # 引入 GMM
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pdb

from model.tcn_model import TemporalConvNet
from model.batch_gat import GAT
from model.cvae_base import CVAE
from model.utils import acc_to_abs, DotDict
from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel
from model.Rag_embedder import TimeSeriesEmbedder  # 引入 Embedder

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
        self.tcn_encoder_search = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size,
                                                  dropout=dropout)
        self.cvae = CVAE(encoder_layer_sizes=cvae_encoder, latent_size=args.cvae_hidden,
                         decoder_layer_sizes=cvae_decoder, conditional=True, num_labels=gat_out + gat_in)

        self.gat = GAT(in_feature=gat_in, hidden_feature=graph_hidden, out_feature=gat_out,
                       attention_layers=3, dropout=0.1, alpha=alpha)

        self.linear_decoder = nn.Linear(args.mlp_layer, n_classes)

        self.context_conv = nn.Conv1d(in_channels=5, out_channels=4, kernel_size=args.cnn_kernels)
        self.context_linear = nn.Linear(11, args.num_context_output_c)

        self.relu = nn.ReLU()

        self.k = args.k
        self.s = args.num_samples

        self.mlp = FutureDistributionAggregator(future_steps=5, h=12, w=12, hidden_dim=256)

        # LED Modules
        self.model = CoreDenoisingModel().cuda()
        self.model_initializer = InitializationModel(t_h=args.obs, d_h=3, t_f=n_classes, d_f=3, k_pred=20).cuda()

        self.betas = self.make_beta_schedule(
            schedule='linear', n_timesteps=100,
            start=1.e-4, end=5.e-2).cuda()

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        # ============================================================
        #  Agent-Map 交互 (航线特征融合)
        # ============================================================
        self.k_retrieve = getattr(args, 'k_retrieve', 20)
        self.n_clusters = getattr(args, 'n_clusters', 3)
        self.traj_dim = getattr(args, 'traj_dim', 3)

        # Map Encoder 输入维度: 3条航线 * 12个点 * 3维 = 108
        self.map_input_dim = self.n_clusters * n_classes * self.traj_dim
        self.map_feature_dim = 64

        # Map Encoder
        self.map_encoder = nn.Sequential(
            nn.Linear(self.map_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.map_feature_dim),
            nn.ReLU()
        )

        self.context_dim = 256
        # Context Fusion: 将 Social Context (256) 与 Map 特征 (64) 融合
        self.context_fusion = nn.Linear(self.context_dim + self.map_feature_dim, self.context_dim)
        self.act_fusion = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.linear_decoder.weight.data.normal_(0, 0.05)
        self.context_linear.weight.data.normal_(0, 0.05)
        self.context_conv.weight.data.normal_(0, 0.1)
        # 初始化新层
        for m in self.map_encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.context_fusion.weight)

    def calculate_mask(self, obs_traj):
        relation_dist = 8
        last_points = obs_traj[:, -1, :]
        diff = last_points[:, None, :] - last_points[None, :, :]
        dist_matrix = diff.norm(p=2, dim=-1)
        adj = F.softmax(-dist_matrix / 1, dim=0)
        mask = dist_matrix < relation_dist
        return adj, mask

    def _get_route_priors(self, obs_traj, rag_system, embedder):
        def _get_route_priors(self, obs_traj, rag_system, embedder):
            if rag_system is None or embedder is None: return None
            # 1.  (B, N, 11, 3)
            if obs_traj.shape[2] == 3:
                # 如果是 (B, N, 3, 11) -> 转为 (B, N, 11, 3)
                obs_perm = obs_traj.permute(0, 1, 3, 2)
            else:
                obs_perm = obs_traj  # 已经是 (B, N, 11, 3)
            # 2. [数据准备] 提取 numpy 数据
            obs_np = obs_perm.detach().cpu().numpy()
            last_pos_gpu = obs_perm[:, :, -1, :].unsqueeze(2).unsqueeze(2)
            # 3. [Embedding]
            # Reshape to (Total_Agents, Time, Dim)
            flat_obs = obs_np.reshape(-1, 11, 3).astype(np.float32)
            query_emb = embedder.embed_batch(flat_obs)
            # 4. [检索]
            search_res = rag_system.search_batch(query_emb, k=self.k_retrieve)
            # 5. [提取与归一化]
            raw = np.array([[np.array(i['pred_data']).T if np.array(i['pred_data']).shape[0] == 3 else np.array(
                i['pred_data']) for i in s] for s in search_res])
            rel = raw - raw[:, :, 0:1, :]
            # 6. [聚类] 使用 GaussianMixture (GMM)
            ctrs = np.array([
                GaussianMixture(n_components=self.n_clusters, covariance_type='diag', random_state=0).fit(
                    r.reshape(self.k_retrieve, -1)
                ).means_.reshape(self.n_clusters, 12, 3)
                for r in rel
            ])
            # 7. [还原]
            relative_routes = ctrs.reshape(obs_traj.shape[0], obs_traj.shape[1], self.n_clusters, 12, 3)
            absolute_routes = relative_routes + last_pos_gpu.detach().cpu().numpy()
            return torch.tensor(absolute_routes).float().to(obs_traj.device)

    def forward(self, x, y, adj, context, rag_system=None, embedder=None, sort=False):
        batch_size = x.shape[0]
        agent_num = x.shape[1]

        # 1. 获取航线先验
        route_priors = self._get_route_priors(x, rag_system, embedder)

        # 2. 编码航线特征
        if route_priors is not None:
            flat_routes = route_priors.view(batch_size * agent_num, -1)
            map_features = self.map_encoder(flat_routes)  # [B*A, 64]
        else:
            map_features = torch.zeros(batch_size * agent_num, self.map_feature_dim).to(x.device)

        # 3. 数据准备
        fut_traj = torch.reshape(y, (batch_size * agent_num, y.shape[2], y.shape[3]))
        fut_traj = fut_traj.permute(0, 2, 1)
        past_traj = torch.reshape(x, (batch_size * agent_num, x.shape[2], x.shape[3]))
        past_traj = past_traj.permute(0, 2, 1)
        traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).cuda()
        for i in range(batch_size):
            traj_mask[i * agent_num:(i + 1) * agent_num, i * agent_num:(i + 1) * agent_num] = 1.

        # 4. LED 初始化
        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask,
                                                                                         map_features)

        sample_prediction = torch.exp(variance_estimation / 2)[
                                ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:,
                                                                       None, None, None]
        loc = sample_prediction + mean_estimation[:, None]


        social_context = self.model.encoder_context(past_traj, traj_mask)  # [B*A, 1, 256]
        social_context = social_context.squeeze(1)  # [B*A, 256]

        # 融合 Map 和 Social 特征
        combined_context = torch.cat([social_context, map_features], dim=-1)
        final_context = self.act_fusion(self.context_fusion(combined_context))

        generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc, context_features=final_context)

        loss_dist = (generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=-1).min(dim=1)[0].mean()
        loss_uncertainty = (torch.exp(-variance_estimation)
                            *
                            (generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2))
                            +
                            variance_estimation
                            ).mean()

        return loss_dist, loss_uncertainty


    def inference(self, x, y, adj, context, rag_system=None, embedder=None):
        batch_size = x.shape[0]
        agent_num = x.shape[1]

        # 1. 获取航线先验
        route_priors = self._get_route_priors(x, rag_system, embedder)

        if route_priors is not None:
            flat_routes = route_priors.view(batch_size * agent_num, -1)
            map_features = self.map_encoder(flat_routes)
        else:
            map_features = torch.zeros(batch_size * agent_num, self.map_feature_dim).to(x.device)

        past_traj = torch.reshape(x, (batch_size * agent_num, x.shape[2], x.shape[3]))
        past_traj = past_traj.permute(0, 2, 1)
        traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).cuda()
        for i in range(batch_size):
            traj_mask[i * agent_num:(i + 1) * agent_num, i * agent_num:(i + 1) * agent_num] = 1.

        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask,
                                                                                         map_features)

        sample_prediction = torch.exp(variance_estimation / 2)[
                                ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:,
                                                                       None, None, None]
        loc = sample_prediction + mean_estimation[:, None]

        # 5. Context Fusion
        social_context = self.model.encoder_context(past_traj, traj_mask).squeeze(1)
        combined_context = torch.cat([social_context, map_features], dim=-1)
        final_context = self.act_fusion(self.context_fusion(combined_context))

        generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc, context_features=final_context)
        return generated_y

    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def make_beta_schedule(self, schedule: str = 'linear',
                           n_timesteps: int = 1000,
                           start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    def p_sample_accelerate(self, x, mask, cur_y, t, context_features=None):
        if t == 0:
            z = torch.zeros_like(cur_y).to(x.device)
        else:
            z = torch.randn_like(cur_y).to(x.device)
        t = torch.tensor([t]).cuda()

        eps_factor = (
                    (1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
        beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)

        # 传入融合后的 context_features
        eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask, map_features=context_features)

        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        sigma_t = self.extract(self.betas, t, cur_y).sqrt()
        sample = mean + sigma_t * z * 0.00001
        return sample

    def p_sample_loop_accelerate(self, x, mask, loc, context_features=None):
        prediction_total = torch.Tensor().cuda()
        cur_y = loc[:, :10]
        for i in reversed(range(NUM_Tau)):
            cur_y = self.p_sample_accelerate(x, mask, cur_y, i, context_features)
        cur_y_ = loc[:, 10:]
        for i in reversed(range(NUM_Tau)):
            cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i, context_features)
        prediction_total = torch.cat((cur_y_, cur_y), dim=1)
        return prediction_total


class FutureDistributionAggregator(nn.Module):
    def __init__(self, future_steps=5, h=12, w=12, hidden_dim=256):
        super().__init__()
        self.in_dim = future_steps * h * w
        self.out_dim = h * w
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim)
        )

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = x.view(b * c, -1)
        x = self.mlp(x)
        x = x.view(b, c, 1, h, w)
        return x





#
# from datetime import datetime
# import random
# import torch
# from torch import nn
# import numpy as np
# # [优化] 移除 sklearn，改用 PyTorch 原生实现以提速
# # from sklearn.cluster import KMeans
#
# from model.tcn_model import TemporalConvNet
# from model.batch_gat import GAT
# from model.cvae_base import CVAE
# from model.utils import acc_to_abs, DotDict
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
#
# import pdb
#
# from models.model_led_initializer import LEDInitializer as InitializationModel
# from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel
# from model.Rag_embedder import TimeSeriesEmbedder
#
# ## 去噪步数
# NUM_Tau = 5
#
#
# # ============================================================
# # [核心优化] PyTorch 并行 Batched KMeans (GPU加速)
# # ============================================================
# def batched_kmeans(X, num_clusters, num_iters=10):
#     """
#     输入 X: [Batch_Size, Num_Points, Dim]
#     输出 Centers: [Batch_Size, Num_Clusters, Dim]
#     说明: 这是一个极简的 KMeans 实现，支持 Batch 并行，完全在 GPU 上运行。
#     """
#     batch_size, num_points, dim = X.shape
#
#     # 1. 初始化中心: 简单地选择前 K 个点作为初始中心 (对于轨迹聚类足够有效且最快)
#     # 也可以随机选择，但在 Batch 维度下稍微麻烦一点，固定选择更稳健
#     centers = X[:, :num_clusters, :].clone()
#
#     for _ in range(num_iters):
#         # 2. 计算距离: [B, N, 1, D] - [B, 1, K, D] -> [B, N, K] (自动广播)
#         # 使用 cdist 计算欧氏距离
#         dists = torch.cdist(X, centers)
#
#         # 3. 分配标签: [B, N]
#         labels = torch.argmin(dists, dim=2)
#
#         # 4. 更新中心: 计算每个簇的均值
#         # 使用 one_hot 编码进行并行求和
#         # mask: [B, N, K]
#         mask = F.one_hot(labels, num_clusters).float()
#
#         # sum_data: [B, K, D] = sum(mask * X)
#         # b:batch, n:points, k:clusters, d:dim
#         sum_data = torch.einsum('bnk,bnd->bkd', mask, X)
#
#         # counts: [B, K, 1]
#         counts = mask.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-6  # 防止除零
#
#         new_centers = sum_data / counts
#
#         # 简单的收敛检测 (可选，为了速度直接跑完固定步数)
#         # if torch.norm(new_centers - centers) < 1e-4: break
#         centers = new_centers
#
#     return centers
#
#
# class TrajAirNet(nn.Module):
#     def __init__(self, args):
#         super(TrajAirNet, self).__init__()
#
#         input_size = args.input_channels
#         n_classes = int(args.preds / args.preds_step)
#         num_channels = [args.tcn_channel_size] * args.tcn_layers
#         num_channels.append(n_classes)
#         tcn_kernel_size = args.tcn_kernels
#         dropout = args.dropout
#
#         graph_hidden = args.graph_hidden
#
#         gat_in = n_classes * args.obs + n_classes ** 2
#         gat_out = n_classes * args.obs + n_classes ** 2
#
#         n_heads = args.gat_heads
#         alpha = args.alpha
#
#         cvae_encoder = [n_classes * n_classes]
#         for layer in range(args.cvae_layers):
#             cvae_encoder.append(args.cvae_channel_size)
#         cvae_decoder = [args.cvae_channel_size] * args.cvae_layers
#         cvae_decoder.append(input_size * args.mlp_layer)
#
#         self.tcn_encoder_x = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
#         self.tcn_encoder_similarest = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size,
#                                                       dropout=dropout)
#         self.tcn_encoder_y = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
#         self.tcn_encoder_search = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size,
#                                                   dropout=dropout)
#         self.cvae = CVAE(encoder_layer_sizes=cvae_encoder, latent_size=args.cvae_hidden,
#                          decoder_layer_sizes=cvae_decoder, conditional=True, num_labels=gat_out + gat_in)
#
#         self.gat = GAT(in_feature=gat_in, hidden_feature=graph_hidden, out_feature=gat_out,
#                        attention_layers=3, dropout=0.1, alpha=alpha)
#
#         self.linear_decoder = nn.Linear(args.mlp_layer, n_classes)
#
#         self.context_conv = nn.Conv1d(in_channels=5, out_channels=4, kernel_size=args.cnn_kernels)
#         self.context_linear = nn.Linear(11, args.num_context_output_c)
#
#         self.relu = nn.ReLU()
#
#         self.k = args.k
#         self.s = args.num_samples
#
#         self.mlp = FutureDistributionAggregator(future_steps=5, h=12, w=12, hidden_dim=256)
#
#         # LED Modules
#         self.model = CoreDenoisingModel().cuda()
#         self.model_initializer = InitializationModel(t_h=args.obs, d_h=3, t_f=n_classes, d_f=3, k_pred=20).cuda()
#
#         self.betas = self.make_beta_schedule(
#             schedule='linear', n_timesteps=100,
#             start=1.e-4, end=5.e-2).cuda()
#
#         self.alphas = 1 - self.betas
#         self.alphas_prod = torch.cumprod(self.alphas, 0)
#         self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
#         self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
#
#         # ============================================================
#         # [新增模块] Agent-Map 交互
#         # ============================================================
#         self.k_retrieve = getattr(args, 'k_retrieve', 20)  # 建议保持较小值，如20
#         self.n_clusters = getattr(args, 'n_clusters', 3)
#         self.traj_dim = getattr(args, 'traj_dim', 3)
#
#         # Map Encoder
#         self.map_input_dim = self.n_clusters * n_classes * self.traj_dim
#         self.map_feature_dim = 64
#
#         self.map_encoder = nn.Sequential(
#             nn.Linear(self.map_input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.map_feature_dim),
#             nn.ReLU()
#         )
#
#         self.context_dim = 256
#         self.context_fusion = nn.Linear(self.context_dim + self.map_feature_dim, self.context_dim)
#         self.act_fusion = nn.ReLU()
#
#         self.init_weights()
#
#     def init_weights(self):
#         self.linear_decoder.weight.data.normal_(0, 0.05)
#         self.context_linear.weight.data.normal_(0, 0.05)
#         self.context_conv.weight.data.normal_(0, 0.1)
#         for m in self.map_encoder:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#         nn.init.xavier_uniform_(self.context_fusion.weight)
#
#     def calculate_mask(self, obs_traj):
#         relation_dist = 8
#         last_points = obs_traj[:, -1, :]
#         diff = last_points[:, None, :] - last_points[None, :, :]
#         dist_matrix = diff.norm(p=2, dim=-1)
#         adj = F.softmax(-dist_matrix / 1, dim=0)
#         mask = dist_matrix < relation_dist
#         return adj, mask
#
#     # ============================================================
#     # [核心优化] 极速版 RAG 检索与聚类
#     # ============================================================
#     def _get_route_priors(self, obs_traj, rag_system, embedder):
#         if rag_system is None or embedder is None: return None
#         # 1.  (B, N, 11, 3)
#         if obs_traj.shape[2] == 3:
#             obs_perm = obs_traj.permute(0, 1, 3, 2)  # (B, N, 11, 3)
#         else:
#             obs_perm = obs_traj  # (B, N, 11, 3)
#         # 2. [数据准备] 提取 numpy 和 GPU 位置信息
#         obs_np = obs_perm.detach().cpu().numpy()
#         last_pos_gpu = obs_perm[:, :, -1, :].unsqueeze(2).unsqueeze(2)
#         # 3. [Embedding]
#         flat_obs = obs_np.reshape(-1, 11, 3).astype(np.float32)
#         query_emb = embedder.embed_batch(flat_obs)  # Embedder 应该能处理 (N, T, D) 格式
#         # 4. [检索] CPU 操作
#         search_res = rag_system.search_batch(query_emb, k=self.k_retrieve)
#         # 5. [提取与归一化]
#         raw = np.array([[np.array(i['pred_data']).T if np.array(i['pred_data']).shape[0] == 3 else np.array(
#             i['pred_data']) for i in s] for s in search_res])
#         rel = raw - raw[:, :, 0:1, :]  # (B*N, K, 12, 3)
#         # 6. [聚类] GPU Batched KMeans
#         X = torch.from_numpy(rel).float().to(obs_traj.device).reshape(rel.shape[0], self.k_retrieve, -1)
#         ctrs_flat = batched_kmeans(X, self.n_clusters, num_iters=5)  # [B*N, N_Clusters, 36]
#         # 7. [还原]
#         relative_routes = ctrs_flat.view(obs_traj.shape[0], obs_traj.shape[1], self.n_clusters, 12, 3)
#         absolute_routes = relative_routes + last_pos_gpu
#         return absolute_routes
#
#     def forward(self, x, y, adj, context, rag_system=None, embedder=None, sort=False):
#         batch_size = x.shape[0]
#         agent_num = x.shape[1]
#         # 1. 获取航线先验 (GPU加速版)
#         route_priors = self._get_route_priors(x, rag_system, embedder)
#         # 2. 编码航线特征
#         if route_priors is not None:
#             flat_routes = route_priors.view(batch_size * agent_num, -1)
#             map_features = self.map_encoder(flat_routes)
#         else:
#             map_features = torch.zeros(batch_size * agent_num, self.map_feature_dim).to(x.device)
#         # 3. 数据准备
#         fut_traj = torch.reshape(y, (batch_size * agent_num, y.shape[2], y.shape[3]))
#         fut_traj = fut_traj.permute(0, 2, 1)
#         past_traj = torch.reshape(x, (batch_size * agent_num, x.shape[2], x.shape[3]))
#         past_traj = past_traj.permute(0, 2, 1)
#         traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).cuda()
#         for i in range(batch_size):
#             traj_mask[i * agent_num:(i + 1) * agent_num, i * agent_num:(i + 1) * agent_num] = 1.
#         # 4. LED 初始化
#         sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask,
#                                                                                          map_features)
#         sample_prediction = torch.exp(variance_estimation / 2)[
#                                 ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:,
#                                                                        None, None, None]
#         loc = sample_prediction + mean_estimation[:, None]
#         # 5. Fusion
#         social_context = self.model.encoder_context(past_traj, traj_mask).squeeze(1)
#         combined_context = torch.cat([social_context, map_features], dim=-1)
#         final_context = self.act_fusion(self.context_fusion(combined_context))
#
#         # 6. Generation
#         generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc, context_features=final_context)
#
#         # 7. Loss
#         loss_dist = (generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=-1).min(dim=1)[0].mean()
#         loss_uncertainty = (torch.exp(-variance_estimation)
#                             *
#                             (generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2))
#                             +
#                             variance_estimation
#                             ).mean()
#
#         return loss_dist, loss_uncertainty
#
#     def inference(self, x, y, adj, context, rag_system=None, embedder=None):
#         batch_size = x.shape[0]
#         agent_num = x.shape[1]
#
#         route_priors = self._get_route_priors(x, rag_system, embedder)
#
#         if route_priors is not None:
#             flat_routes = route_priors.view(batch_size * agent_num, -1)
#             map_features = self.map_encoder(flat_routes)
#         else:
#             map_features = torch.zeros(batch_size * agent_num, self.map_feature_dim).to(x.device)
#
#         past_traj = torch.reshape(x, (batch_size * agent_num, x.shape[2], x.shape[3]))
#         past_traj = past_traj.permute(0, 2, 1)
#         traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).cuda()
#         for i in range(batch_size):
#             traj_mask[i * agent_num:(i + 1) * agent_num, i * agent_num:(i + 1) * agent_num] = 1.
#
#         sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask,
#                                                                                          map_features)
#         sample_prediction = torch.exp(variance_estimation / 2)[
#                                 ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:,
#                                                                        None, None, None]
#         loc = sample_prediction + mean_estimation[:, None]
#
#         social_context = self.model.encoder_context(past_traj, traj_mask).squeeze(1)
#         combined_context = torch.cat([social_context, map_features], dim=-1)
#         final_context = self.act_fusion(self.context_fusion(combined_context))
#
#         generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc, context_features=final_context)
#         return generated_y
#
#     def extract(self, input, t, x):
#         shape = x.shape
#         out = torch.gather(input, 0, t.to(input.device))
#         reshape = [t.shape[0]] + [1] * (len(shape) - 1)
#         return out.reshape(*reshape)
#
#     def make_beta_schedule(self, schedule: str = 'linear',
#                            n_timesteps: int = 1000,
#                            start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
#         if schedule == 'linear':
#             betas = torch.linspace(start, end, n_timesteps)
#         elif schedule == "quad":
#             betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
#         elif schedule == "sigmoid":
#             betas = torch.linspace(-6, 6, n_timesteps)
#             betas = torch.sigmoid(betas) * (end - start) + start
#         return betas
#
#     def p_sample_accelerate(self, x, mask, cur_y, t, context_features=None):
#         if t == 0:
#             z = torch.zeros_like(cur_y).to(x.device)
#         else:
#             z = torch.randn_like(cur_y).to(x.device)
#         t = torch.tensor([t]).cuda()
#         eps_factor = (
#                     (1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
#         beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
#         eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask, map_features=context_features)
#         mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
#         sigma_t = self.extract(self.betas, t, cur_y).sqrt()
#         sample = mean + sigma_t * z * 0.00001
#         return sample
#
#     def p_sample_loop_accelerate(self, x, mask, loc, context_features=None):
#         prediction_total = torch.Tensor().cuda()
#         cur_y = loc[:, :10]
#         for i in reversed(range(NUM_Tau)):
#             cur_y = self.p_sample_accelerate(x, mask, cur_y, i, context_features)
#         cur_y_ = loc[:, 10:]
#         for i in reversed(range(NUM_Tau)):
#             cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i, context_features)
#         prediction_total = torch.cat((cur_y_, cur_y), dim=1)
#         return prediction_total
#
#
# class FutureDistributionAggregator(nn.Module):
#     def __init__(self, future_steps=5, h=12, w=12, hidden_dim=256):
#         super().__init__()
#         self.in_dim = future_steps * h * w
#         self.out_dim = h * w
#         self.mlp = nn.Sequential(
#             nn.Linear(self.in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, self.out_dim)
#         )
#
#     def forward(self, x):
#         b, c, f, h, w = x.shape
#         x = x.view(b * c, -1)
#         x = self.mlp(x)
#         x = x.view(b, c, 1, h, w)
#         return x