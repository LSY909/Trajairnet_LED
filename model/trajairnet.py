# # from datetime import datetime
# # import random
# # import torch
# # from torch import nn
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import torch.nn.functional as F
# # import pdb
# #
# # from model.tcn_model import TemporalConvNet
# # from model.batch_gat import GAT
# # from model.cvae_base import CVAE
# # from model.utils import acc_to_abs, DotDict
# # from models.model_led_initializer import LEDInitializer as InitializationModel
# # #from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel
# #
# # ## 去噪步数
# # #NUM_Tau = 5
# #
# # # 引入QCNet组件
# # from model.qcnet_modules import QCNetStyleRouteEncoder, QCNetAgentMapAttention, QCNetRefinementDecoder
# #
# # # 不需要这个分布聚合器了
# # # class FutureDistributionAggregator(nn.Module):
# # #     def __init__(self, future_steps=5, h=12, w=12, hidden_dim=256):
# # #         super().__init__()
# # #         self.in_dim = future_steps * h * w
# # #         self.out_dim = h * w
# # #         self.mlp = nn.Sequential(
# # #             nn.Linear(self.in_dim, hidden_dim),
# # #             nn.ReLU(),
# # #             nn.Linear(hidden_dim, self.out_dim)
# # #         )
# # #
# # #     def forward(self, x):
# # #         b, c, f, h, w = x.shape
# # #         x = x.view(b * c, -1)
# # #         x = self.mlp(x)
# # #         x = x.view(b, c, 1, h, w)
# # #         return x
# #
# #
# # class TrajAirNet(nn.Module):
# #     def __init__(self, args):
# #         super(TrajAirNet, self).__init__()
# #
# #         input_size = args.input_channels
# #         n_classes = int(args.preds / args.preds_step)
# #         num_channels = [args.tcn_channel_size] * args.tcn_layers
# #         num_channels.append(n_classes)
# #         tcn_kernel_size = args.tcn_kernels
# #         dropout = args.dropout
# #         graph_hidden = args.graph_hidden
# #         gat_in = n_classes * args.obs + n_classes ** 2
# #         gat_out = n_classes * args.obs + n_classes ** 2
# #         n_heads = args.gat_heads
# #         alpha = args.alpha
# #
# #         cvae_encoder = [n_classes * n_classes]
# #         for layer in range(args.cvae_layers):
# #             cvae_encoder.append(args.cvae_channel_size)
# #         cvae_decoder = [args.cvae_channel_size] * args.cvae_layers
# #         cvae_decoder.append(input_size * args.mlp_layer)
# #
# #         #保留TCN编码器
# #         self.tcn_encoder_x = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
# #         self.tcn_encoder_similarest = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size,
# #                                                       dropout=dropout)
# #         self.tcn_encoder_y = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
# #         self.tcn_encoder_search = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size,
# #                                                   dropout=dropout)
# #
# #         self.cvae = CVAE(encoder_layer_sizes=cvae_encoder, latent_size=args.cvae_hidden,
# #                          decoder_layer_sizes=cvae_decoder, conditional=True, num_labels=gat_out + gat_in)
# #
# #         self.gat = GAT(in_feature=gat_in, hidden_feature=graph_hidden, out_feature=gat_out,
# #                        attention_layers=3, dropout=0.1, alpha=alpha)
# #
# #         ### [新增] 增加一个线性层，把 TCN 输出适配到 QCNet 的 256 维
# #         self.agent_proj = nn.Linear(num_channels[-1], 256)
# #
# #         self.linear_decoder = nn.Linear(args.mlp_layer, n_classes)
# #
# #         self.context_conv = nn.Conv1d(in_channels=5, out_channels=4, kernel_size=args.cnn_kernels)
# #         self.context_linear = nn.Linear(11, args.num_context_output_c)
# #
# #         self.relu = nn.ReLU()
# #
# #         self.k = args.k
# #         self.s = args.num_samples
# #
# #         #self.mlp = FutureDistributionAggregator(future_steps=5, h=12, w=12, hidden_dim=256)
# #
# #         # LED Modules
# #         #self.model = CoreDenoisingModel().cuda()
# #         self.model_initializer = InitializationModel(t_h=args.obs, d_h=3, t_f=n_classes, d_f=3, k_pred=20).cuda()
# #
# #         self.betas = self.make_beta_schedule(
# #             schedule='linear', n_timesteps=100,
# #             start=1.e-4, end=5.e-2).cuda()
# #
# #         self.alphas = 1 - self.betas
# #         self.alphas_prod = torch.cumprod(self.alphas, 0)
# #         self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
# #         self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
# #
# #         # ============================================================
# #         #  Agent-Map 交互 (适配 4D 航道数据: x,y,z,width)
# #         # ============================================================
# #         self.k_retrieve = getattr(args, 'k_retrieve', 20)
# #         self.n_clusters = getattr(args, 'n_clusters', 3)
# #         self.traj_dim = getattr(args, 'traj_dim', 3)
# #
# #         # [修改] 输入维度: 3条航线 * 12个点 * 4维 (x,y,z,w)
# #         # n_classes 是预测步长 (例如 12)
# #         self.map_input_dim = self.n_clusters * n_classes * 4
# #         self.map_feature_dim = 64
# #
# #         # Map Encoder (简单的 MLP)
# #         self.map_encoder = nn.Sequential(
# #             nn.Linear(self.map_input_dim, 128),
# #             nn.ReLU(),
# #             nn.Linear(128, self.map_feature_dim),
# #             nn.ReLU()
# #         )
# #
# #         self.context_dim = 256
# #         # Context Fusion: 将 Social Context (256) 与 Map 特征 (64) 融合
# #         self.context_fusion = nn.Linear(self.context_dim + self.map_feature_dim, self.context_dim)
# #         self.act_fusion = nn.ReLU()
# #
# #         self.init_weights()
# #
# #     def init_weights(self):
# #         self.linear_decoder.weight.data.normal_(0, 0.05)
# #         self.context_linear.weight.data.normal_(0, 0.05)
# #         self.context_conv.weight.data.normal_(0, 0.1)
# #         # 初始化新层
# #         for m in self.map_encoder:
# #             if isinstance(m, nn.Linear):
# #                 nn.init.xavier_uniform_(m.weight)
# #         nn.init.xavier_uniform_(self.context_fusion.weight)
# #
# #     def calculate_mask(self, obs_traj):
# #         relation_dist = 8
# #         last_points = obs_traj[:, -1, :]
# #         diff = last_points[:, None, :] - last_points[None, :, :]
# #         dist_matrix = diff.norm(p=2, dim=-1)
# #         adj = F.softmax(-dist_matrix / 1, dim=0)
# #         mask = dist_matrix < relation_dist
# #         return adj, mask
# #
# #     # forward 只接收 route_priors
# #     def forward(self, x, y, adj, context, route_priors=None, sort=False):
# #         batch_size = x.shape[0]
# #         agent_num = x.shape[1]
# #
# #         # 1. 处理航线先验
# #         if route_priors is not None:
# #             # route_priors: (B, N, 3, 12, 4) [相对坐标 + 宽度]
# #
# #             # A. 拆分
# #             rel_coords = route_priors[..., :3]  # 前3维 (x, y, z)
# #             widths = route_priors[..., 3:]  # 第4维 (width)
# #
# #             # B. 还原坐标绝对位置
# #             if x.shape[2] == 3:
# #                 last_pos = x.permute(0, 1, 3, 2)[:, :, -1, :].unsqueeze(2).unsqueeze(2)
# #             else:
# #                 last_pos = x[:, :, -1, :].unsqueeze(2).unsqueeze(2)
# #
# #             abs_coords = rel_coords.to(x.device) + last_pos
# #
# #             # C. 重新拼接 (4D)
# #             abs_priors = torch.cat([abs_coords, widths.to(x.device)], dim=-1)
# #
# #             # D. 展平并编码
# #             flat_routes = abs_priors.view(batch_size * agent_num, -1)
# #             map_features = self.map_encoder(flat_routes)  # [B*A, 64]
# #         else:
# #             # 兼容无数据情况
# #             map_features = torch.zeros(batch_size * agent_num, self.map_feature_dim).to(x.device)
# #
# #         # 2. 数据准备
# #         fut_traj = torch.reshape(y, (batch_size * agent_num, y.shape[2], y.shape[3])).permute(0, 2, 1)
# #         past_traj = torch.reshape(x, (batch_size * agent_num, x.shape[2], x.shape[3])).permute(0, 2, 1)
# #         traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).cuda()
# #         for i in range(batch_size):
# #             traj_mask[i * agent_num:(i + 1) * agent_num, i * agent_num:(i + 1) * agent_num] = 1.
# #
# #         # 3. LED 初始化
# #         sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask,
# #                                                                                          map_features)
# #         sample_prediction = torch.exp(variance_estimation / 2)[
# #                                 ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:,
# #                                                                        None, None, None]
# #         loc = sample_prediction + mean_estimation[:, None]
# #
# #         # 4. Fusion
# #         social_context = self.model.encoder_context(past_traj, traj_mask).squeeze(1)  # [B*A, 256]
# #         combined_context = torch.cat([social_context, map_features], dim=-1)
# #         final_context = self.act_fusion(self.context_fusion(combined_context))
# #
# #         # 5. Generation
# #         generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc, context_features=final_context)
# #
# #         # 6. Loss
# #         loss_dist = (generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=-1).min(dim=1)[0].mean()
# #         loss_uncertainty = (torch.exp(-variance_estimation)
# #                             *
# #                             (generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2))
# #                             +
# #                             variance_estimation
# #                             ).mean()
# #
# #         return loss_dist, loss_uncertainty
# #
# #     #  inference 同步
# #     def inference(self, x, y, adj, context, route_priors=None):
# #         batch_size = x.shape[0]
# #         agent_num = x.shape[1]
# #
# #         if route_priors is not None:
# #             rel_coords = route_priors[..., :3]
# #             widths = route_priors[..., 3:]
# #             if x.shape[2] == 3:
# #                 last_pos = x.permute(0, 1, 3, 2)[:, :, -1, :].unsqueeze(2).unsqueeze(2)
# #             else:
# #                 last_pos = x[:, :, -1, :].unsqueeze(2).unsqueeze(2)
# #             abs_coords = rel_coords.to(x.device) + last_pos
# #             abs_priors = torch.cat([abs_coords, widths.to(x.device)], dim=-1)
# #
# #             flat_routes = abs_priors.view(batch_size * agent_num, -1)
# #             map_features = self.map_encoder(flat_routes)
# #         else:
# #             map_features = torch.zeros(batch_size * agent_num, self.map_feature_dim).to(x.device)
# #
# #         past_traj = torch.reshape(x, (batch_size * agent_num, x.shape[2], x.shape[3])).permute(0, 2, 1)
# #         traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).cuda()
# #         for i in range(batch_size):
# #             traj_mask[i * agent_num:(i + 1) * agent_num, i * agent_num:(i + 1) * agent_num] = 1.
# #
# #         sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask,
# #                                                                                          map_features)
# #
# #         sample_prediction = torch.exp(variance_estimation / 2)[
# #                                 ..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:,
# #                                                                        None, None, None]
# #         loc = sample_prediction + mean_estimation[:, None]
# #
# #         social_context = self.model.encoder_context(past_traj, traj_mask).squeeze(1)
# #         combined_context = torch.cat([social_context, map_features], dim=-1)
# #         final_context = self.act_fusion(self.context_fusion(combined_context))
# #
# #         generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc, context_features=final_context)
# #         return generated_y
# #
# #     def extract(self, input, t, x):
# #         shape = x.shape
# #         out = torch.gather(input, 0, t.to(input.device))
# #         reshape = [t.shape[0]] + [1] * (len(shape) - 1)
# #         return out.reshape(*reshape)
# #
# #     def make_beta_schedule(self, schedule: str = 'linear',
# #                            n_timesteps: int = 1000,
# #                            start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
# #         if schedule == 'linear':
# #             betas = torch.linspace(start, end, n_timesteps)
# #         elif schedule == "quad":
# #             betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
# #         elif schedule == "sigmoid":
# #             betas = torch.linspace(-6, 6, n_timesteps)
# #             betas = torch.sigmoid(betas) * (end - start) + start
# #         return betas
# #
# #     def p_sample_accelerate(self, x, mask, cur_y, t, context_features=None):
# #         if t == 0:
# #             z = torch.zeros_like(cur_y).to(x.device)
# #         else:
# #             z = torch.randn_like(cur_y).to(x.device)
# #         t = torch.tensor([t]).cuda()
# #
# #         eps_factor = (
# #                 (1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
# #         beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
# #
# #         eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask, map_features=context_features)
# #
# #         mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
# #         sigma_t = self.extract(self.betas, t, cur_y).sqrt()
# #         sample = mean + sigma_t * z * 0.00001
# #         return sample
# #
# #     def p_sample_loop_accelerate(self, x, mask, loc, context_features=None):
# #         prediction_total = torch.Tensor().cuda()
# #         cur_y = loc[:, :10]
# #         for i in reversed(range(NUM_Tau)):
# #             cur_y = self.p_sample_accelerate(x, mask, cur_y, i, context_features)
# #         cur_y_ = loc[:, 10:]
# #         for i in reversed(range(NUM_Tau)):
# #             cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i, context_features)
# #         prediction_total = torch.cat((cur_y_, cur_y), dim=1)
# #         return prediction_total
#
#
# from datetime import datetime
# import random
# import torch
# from torch import nn
# import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# import pdb
#
# from model.tcn_model import TemporalConvNet
# from model.batch_gat import GAT
# from model.cvae_base import CVAE
# from model.utils import acc_to_abs, DotDict
# from models.model_led_initializer import LEDInitializer as InitializationModel
#
# # [新增] 引入 QCNet 组件
# # 请确保 model/qcnet_modules.py 包含: QCNetStyleRouteEncoder, QCNetAgentMapAttention, QCNetRefinementDecoder
# from model.qcnet_modules import QCNetStyleRouteEncoder, QCNetAgentMapAttention, QCNetRefinementDecoder
#
#
# class TrajAirNet(nn.Module):
#     def __init__(self, args):
#         super(TrajAirNet, self).__init__()
#
#         # --- 参数配置 (保持不变) ---
#         input_size = args.input_channels
#         n_classes = int(args.preds / args.preds_step)  # 预测步长 (例如12)
#         num_channels = [args.tcn_channel_size] * args.tcn_layers
#         num_channels.append(n_classes)
#         tcn_kernel_size = args.tcn_kernels
#         dropout = args.dropout
#
#         graph_hidden = args.graph_hidden
#         gat_in = n_classes * args.obs + n_classes ** 2
#         gat_out = n_classes * args.obs + n_classes ** 2
#         n_heads = args.gat_heads
#         alpha = args.alpha
#
#         # --- TCN 历史编码器 (保持不变) ---
#         self.tcn_encoder_x = TemporalConvNet(input_size, num_channels, kernel_size=tcn_kernel_size, dropout=dropout)
#
#         # [新增] 线性投影层：将 TCN 输出的特征维度映射到 QCNet 需要的 256 维
#         self.agent_proj = nn.Linear(num_channels[-1], 256)
#
#         # --- 辅助模块 (保持不变，即便暂时不用也保留以防报错) ---
#         cvae_encoder = [n_classes * n_classes]
#         for layer in range(args.cvae_layers):
#             cvae_encoder.append(args.cvae_channel_size)
#         cvae_decoder = [args.cvae_channel_size] * args.cvae_layers
#         cvae_decoder.append(input_size * args.mlp_layer)
#
#         self.cvae = CVAE(encoder_layer_sizes=cvae_encoder, latent_size=args.cvae_hidden,
#                          decoder_layer_sizes=cvae_decoder, conditional=True, num_labels=gat_out + gat_in)
#
#         self.gat = GAT(in_feature=gat_in, hidden_feature=graph_hidden, out_feature=gat_out,
#                        attention_layers=3, dropout=0.1, alpha=alpha)
#
#         # --- LED 初始化模块 (保留) ---
#         # [关键] k_pred 设置为 6，对应 QCNet 的 6 个模态
#         self.model_initializer = InitializationModel(t_h=args.obs, d_h=3, t_f=n_classes, d_f=3, k_pred=6).cuda()
#
#         # ============================================================
#         #  [新增/修改] QCNet 核心组件 (替代 Diffusion)
#         # ============================================================
#         self.n_clusters = getattr(args, 'n_clusters', 3)
#         self.traj_dim = getattr(args, 'traj_dim', 3)
#
#         # 1. Map Encoder: 提取航线几何特征
#         self.route_encoder = QCNetStyleRouteEncoder(
#             in_channels=7,  # 输入: (x,y,z,w, cos,sin, mag)
#             hidden_dim=64,
#             out_dim=128  # Map Feature Dimension
#         )
#
#         # 2. Attention: 融合 Agent 和 Map 特征
#         self.agent_map_attn = QCNetAgentMapAttention(
#             agent_dim=256,
#             map_dim=128,
#             hidden_dim=256
#         )
#
#         # 3. Refinement Decoder: 轨迹精修生成
#         self.refinement_decoder = QCNetRefinementDecoder(
#             embed_dim=256,
#             future_steps=n_classes,
#             num_modes=6  # 必须与 k_pred 一致
#         )
#
#         # 初始化权重
#         self.init_weights()
#
#     def init_weights(self):
#         # 简单初始化，可以根据需要调整
#         nn.init.xavier_uniform_(self.agent_proj.weight)
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
#     def forward(self, x, y, adj, context, route_priors=None, sort=False):
#         batch_size = x.shape[0]
#         agent_num = x.shape[1]
#
#         # -----------------------------------------------------------
#         # Step 1: 处理航线数据 (Map Encoding)
#         # -----------------------------------------------------------
#         if route_priors is not None:
#             # route_priors: (B, N, 3, 12, 4) -> [rel_x, rel_y, rel_z, width]
#             rel_coords = route_priors[..., :3]
#             widths = route_priors[..., 3:]
#
#             # 还原绝对坐标 (保留你原有的逻辑)
#             if x.shape[2] == 3:
#                 last_pos = x.permute(0, 1, 3, 2)[:, :, -1, :].unsqueeze(2).unsqueeze(2)
#             else:
#                 last_pos = x[:, :, -1, :].unsqueeze(2).unsqueeze(2)
#
#             abs_coords = rel_coords.to(x.device) + last_pos
#
#             # 拼接输入: [B, A, Clusters, 12, 4]
#             map_input = torch.cat([abs_coords, widths.to(x.device)], dim=-1)
#
#             # 编码: [B*A, Clusters, 128]
#             map_features = self.route_encoder(map_input)
#         else:
#             # 无地图数据时的零填充
#             map_features = torch.zeros(batch_size * agent_num, self.n_clusters, 128).to(x.device)
#
#         # -----------------------------------------------------------
#         # Step 2: 处理 Agent 历史特征 (Agent Encoding)
#         # -----------------------------------------------------------
#         past_traj = torch.reshape(x, (batch_size * agent_num, x.shape[2], x.shape[3])).permute(0, 2, 1)
#         fut_traj = torch.reshape(y, (batch_size * agent_num, y.shape[2], y.shape[3])).permute(0, 2, 1)
#
#         # TCN 编码
#         tcn_out = self.tcn_encoder_x(past_traj)  # [B*A, Channels, Obs_Len]
#         # 取最后一帧特征并投影到 256 维
#         agent_feats = self.agent_proj(tcn_out[:, :, -1])  # [B*A, 256]
#
#         # -----------------------------------------------------------
#         # Step 3: 特征融合 (Context Fusion)
#         # -----------------------------------------------------------
#         # 使用 QCNet Attention 模块
#         # context_feat: [B*A, 256] -> 将作为 Decoder 的 Memory
#         context_feat, _ = self.agent_map_attn(agent_feats, map_features)
#
#         # -----------------------------------------------------------
#         # Step 4: 生成 Anchors (利用 LED Initialization)
#         # -----------------------------------------------------------
#         traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).cuda()
#         for i in range(batch_size):
#             traj_mask[i * agent_num:(i + 1) * agent_num, i * agent_num:(i + 1) * agent_num] = 1.
#
#         # 为了适配 LED 接口，传入 Map 特征的均值
#         map_feat_mean = map_features.mean(dim=1)  # [B*A, 128]
#
#         # ===> 你的核心 LED 代码 (保留) <===
#         sample_prediction, mean_estimation, variance_estimation = self.model_initializer(
#             past_traj, traj_mask, map_feat_mean
#         )
#
#         # 标准化处理
#         std_val = sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
#         std_val = torch.clamp(std_val, min=1e-6)  # 防止除零
#
#         sample_prediction = torch.exp(variance_estimation / 2)[..., None, None] * \
#                             (sample_prediction / std_val)
#
#         # loc: [B*A, K, T, 3] -> 这里的 loc 就是粗轨迹 (Anchors)
#         loc = sample_prediction + mean_estimation[:, None]
#
#         # -----------------------------------------------------------
#         # Step 5: 轨迹精修 (QCNet Refinement)
#         # -----------------------------------------------------------
#         # 截取 (x, y) 坐标作为 Anchor 输入
#         anchors = loc[..., :2]
#
#         # Decoder 前向传播
#         # refined_traj: [B*A, K, T, 2]
#         # scores: [B*A, K]
#         refined_traj, scores = self.refinement_decoder(anchors, context_feat)
#
#         # -----------------------------------------------------------
#         # Step 6: Loss 计算 (Winner-Takes-All)
#         # -----------------------------------------------------------
#         gt_traj_xy = fut_traj[..., :2]
#
#         # 计算所有 Mode 与 GT 的欧氏距离
#         norm_diff = torch.norm(refined_traj - gt_traj_xy.unsqueeze(1), p=2, dim=-1).mean(dim=-1)  # [N, K]
#
#         # 找到最佳 Mode (minADE)
#         min_ade, best_mode_idx = torch.min(norm_diff, dim=1)
#
#         # 6.1 Regression Loss: 只计算 Best Mode 的损失
#         best_mode_mask = torch.zeros_like(norm_diff).scatter_(1, best_mode_idx.unsqueeze(1), 1.0)
#         reg_loss = (norm_diff * best_mode_mask).sum() / batch_size
#
#         # 6.2 Classification Loss: 交叉熵 (让模型预测哪个是 Best Mode)
#         cls_loss = F.cross_entropy(scores, best_mode_idx)
#
#         # 总 Loss
#         total_loss = reg_loss + cls_loss
#
#         return total_loss, min_ade.mean()
#
#     # [修改] inference 函数
#     def inference(self, x, y, adj, context, route_priors=None):
#         batch_size = x.shape[0]
#         agent_num = x.shape[1]
#
#         # --- 1. Map Encoding ---
#         if route_priors is not None:
#             rel_coords = route_priors[..., :3]
#             widths = route_priors[..., 3:]
#             if x.shape[2] == 3:
#                 last_pos = x.permute(0, 1, 3, 2)[:, :, -1, :].unsqueeze(2).unsqueeze(2)
#             else:
#                 last_pos = x[:, :, -1, :].unsqueeze(2).unsqueeze(2)
#             abs_coords = rel_coords.to(x.device) + last_pos
#             map_input = torch.cat([abs_coords, widths.to(x.device)], dim=-1)
#             map_features = self.route_encoder(map_input)
#         else:
#             map_features = torch.zeros(batch_size * agent_num, self.n_clusters, 128).to(x.device)
#
#         # --- 2. Agent Encoding ---
#         past_traj = torch.reshape(x, (batch_size * agent_num, x.shape[2], x.shape[3])).permute(0, 2, 1)
#         tcn_out = self.tcn_encoder_x(past_traj)
#         agent_feats = self.agent_proj(tcn_out[:, :, -1])
#
#         # --- 3. Context Fusion ---
#         context_feat, _ = self.agent_map_attn(agent_feats, map_features)
#
#         # --- 4. Anchors Generation (LED) ---
#         traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).cuda()
#         for i in range(batch_size):
#             traj_mask[i * agent_num:(i + 1) * agent_num, i * agent_num:(i + 1) * agent_num] = 1.
#
#         map_feat_mean = map_features.mean(dim=1)
#         sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask,
#                                                                                          map_feat_mean)
#
#         std_val = sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
#         std_val = torch.clamp(std_val, min=1e-6)
#         sample_prediction = torch.exp(variance_estimation / 2)[..., None, None] * (sample_prediction / std_val)
#
#         loc = sample_prediction + mean_estimation[:, None]
#         anchors = loc[..., :2]
#
#         # --- 5. Refinement ---
#         refined_traj, scores = self.refinement_decoder(anchors, context_feat)
#
#         # --- 6. Selection ---
#         # 选择分数最高的轨迹作为最终输出
#         best_mode_idx = torch.argmax(scores, dim=1)  # [B*A]
#         best_traj = refined_traj[torch.arange(refined_traj.shape[0]), best_mode_idx]  # [B*A, T, 2]
#
#         return best_traj


import torch
from torch import nn
import torch.nn.functional as F
from models.model_led_initializer import LEDInitializer as InitializationModel
from model.qcnet_modules import QCNetStyleRouteEncoder, QCNetAgentMapAttention, QCNetRefinementDecoder


# ==========================================
# SocialEncoder (保持不变)
# ==========================================
class SocialEncoder(nn.Module):
    def __init__(self, obs_len):
        super().__init__()
        self.encode_past = nn.Linear(obs_len * 3, 256, bias=False)
        self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

    def forward(self, h, mask=None):
        h_flat = h.reshape(h.size(0), -1)
        h_feat = self.encode_past(h_flat).unsqueeze(1)
        h_feat_ = self.transformer_encoder(h_feat)
        return (h_feat + h_feat_)


class TrajAirNet(nn.Module):
    def __init__(self, args):
        super(TrajAirNet, self).__init__()

        n_classes = int(args.preds / args.preds_step)

        self.n_clusters = getattr(args, 'n_clusters', 3)
        self.map_feature_dim = 128
        self.context_dim = 256
        self.k_pred = 6

        # 1. 历史轨迹编码器
        self.context_encoder = SocialEncoder(obs_len=args.obs)

        # 2. 航线编码器
        self.route_encoder = QCNetStyleRouteEncoder(
            in_channels=7,
            out_dim=self.map_feature_dim
        )

        # 3. Agent-Map 交互模块
        self.agent_map_attn = QCNetAgentMapAttention(
            agent_dim=self.context_dim,
            map_dim=self.map_feature_dim,
            hidden_dim=self.context_dim,
            num_heads=4
        )

        # 4. Proposal 阶段
        self.map_to_led = nn.Linear(self.context_dim, 64)
        self.model_initializer = InitializationModel(
            t_h=args.obs,
            d_h=3,
            t_f=n_classes,
            d_f=3,
            k_pred=self.k_pred
        ).cuda()

        # 5. Refinement 阶段
        self.refinement_module = QCNetRefinementDecoder(
            embed_dim=self.context_dim,
            future_steps=n_classes,
            num_modes=self.k_pred
        )

    def forward(self, x, y, adj, context, route_priors=None, sort=False):
        """
        x: [Batch, Agents, 3, Obs_Len]
        y: [Batch, Agents, 3, Pred_Len]
        """
        batch_size = x.shape[0]
        agent_num = x.shape[1]

        # ==========================================
        # 0. 坐标系归一化 (关键步骤！)
        # ==========================================
        # 提取最后时刻的绝对位置作为参考点
        # x shape: [B, A, 3, 11] -> last_pos: [B, A, 3, 1]
        if x.shape[2] == 3 and x.shape[3] != 3:
            last_pos = x[:, :, :, -1].unsqueeze(-1)  # [B, A, 3, 1]
            # 顺便把 x 转为 [B, A, 11, 3] 方便后续处理
            x_perm = x.permute(0, 1, 3, 2)  # [B, A, 11, 3]
            last_pos_flat = last_pos.squeeze(-1).unsqueeze(2)  # [B, A, 1, 3] 用于广播
        else:
            # 假设已经是 [B, A, 11, 3]
            x_perm = x
            last_pos_flat = x[:, :, -1, :].unsqueeze(2)  # [B, A, 1, 3]

        # 计算相对历史轨迹 (Relative History)
        # [B, A, 11, 3] - [B, A, 1, 3] = [B, A, 11, 3]
        norm_past_traj = x_perm - last_pos_flat

        # 准备输入: [B*A, 11, 3]
        past_traj_input = norm_past_traj.reshape(batch_size * agent_num, -1, 3)

        # -------------------------------------------------------
        # A. 航线特征编码 (使用相对坐标)
        # -------------------------------------------------------
        if route_priors is not None:
            # route_priors: [B, A, C, T, 4]
            # 这里的 rel_coords 本身就是相对于检索起点的偏移，
            # 而检索起点接近 last_pos，所以直接用 rel_coords 就是很好的归一化特征
            rel_coords = route_priors[..., :3]
            widths = route_priors[..., 3:]

            # [修改] 不再加 last_pos 还原绝对坐标，直接用相对坐标编码
            # 这样模型学习的是“形状”，而不是“地图上的绝对位置”
            norm_priors = torch.cat([rel_coords.to(x.device), widths.to(x.device)], dim=-1)

            map_features = self.route_encoder(norm_priors)
        else:
            map_features = torch.zeros(batch_size * agent_num, self.n_clusters, self.map_feature_dim).to(x.device)

        # -------------------------------------------------------
        # B. 历史轨迹编码
        # -------------------------------------------------------
        traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).to(x.device)
        for i in range(batch_size):
            traj_mask[i * agent_num:(i + 1) * agent_num, i * agent_num:(i + 1) * agent_num] = 1.

        # 输入归一化后的轨迹
        social_context = self.context_encoder(past_traj_input, traj_mask).squeeze(1)

        # -------------------------------------------------------
        # C. 交互 & D. Proposal
        # -------------------------------------------------------
        fused_context, _ = self.agent_map_attn(social_context, map_features)
        led_input = self.map_to_led(fused_context)

        # LED 预测的也是相对位移
        _, guess_mean, _ = self.model_initializer(past_traj_input, traj_mask, led_input)

        # -------------------------------------------------------
        # E. Refinement
        # -------------------------------------------------------
        # 输出的 refined_traj 是 [B*A, K, T, 2] (相对坐标)
        refined_traj_rel, scores = self.refinement_module(guess_mean, fused_context)

        # -------------------------------------------------------
        # F. Loss 计算 (全部在相对坐标系下进行)
        # -------------------------------------------------------
        # 1. 处理真值 y: [B, A, 3, T] -> [B, A, T, 3]
        if y.shape[2] == 3:
            y_perm = y.permute(0, 1, 3, 2)
        else:
            y_perm = y

        # 2. 计算真值的相对坐标
        # gt_abs: [B, A, T, 3]
        # last_pos_flat: [B, A, 1, 3]
        # gt_rel: [B, A, T, 3] -> [B*A, T, 2] (只取xy)
        gt_traj_rel = (y_perm - last_pos_flat.to(y.device)).reshape(batch_size * agent_num, -1, 3)[..., :2]

        # 3. 计算距离 (Pred Rel vs GT Rel)
        diff = refined_traj_rel - gt_traj_rel.unsqueeze(1)
        dist = torch.norm(diff, p=2, dim=-1).mean(dim=-1)
        best_mode_idx = torch.argmin(dist, dim=-1)

        batch_indices = torch.arange(refined_traj_rel.shape[0], device=x.device)
        best_traj = refined_traj_rel[batch_indices, best_mode_idx]

        reg_loss = F.smooth_l1_loss(best_traj, gt_traj_rel)
        cls_loss = F.cross_entropy(scores, best_mode_idx)

        return reg_loss, cls_loss

    def inference(self, x, y, adj, context, route_priors=None):
        batch_size = x.shape[0]
        agent_num = x.shape[1]

        # 0. 获取参考点
        if x.shape[2] == 3 and x.shape[3] != 3:
            # [B, A, 3, T] -> [B, A, 3, 1] -> [B, A, 1, 3] (flat)
            last_pos_flat = x[:, :, :, -1].unsqueeze(-1).permute(0, 1, 3, 2).reshape(batch_size * agent_num, 1, 3)
            x_perm = x.permute(0, 1, 3, 2)
        else:
            last_pos_flat = x[:, :, -1, :].unsqueeze(2).reshape(batch_size * agent_num, 1, 3)
            x_perm = x

        # 1. 归一化输入
        norm_past_traj = (x_perm.reshape(batch_size * agent_num, -1, 3) - last_pos_flat)

        # 2. 航线编码 (直接用相对特征)
        if route_priors is not None:
            rel_coords = route_priors[..., :3]
            widths = route_priors[..., 3:]
            norm_priors = torch.cat([rel_coords.to(x.device), widths.to(x.device)], dim=-1)
            map_features = self.route_encoder(norm_priors)
        else:
            map_features = torch.zeros(batch_size * agent_num, self.n_clusters, self.map_feature_dim).to(x.device)

        # 3. 网络前向
        traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).to(x.device)
        for i in range(batch_size):
            traj_mask[i * agent_num:(i + 1) * agent_num, i * agent_num:(i + 1) * agent_num] = 1.

        social_context = self.context_encoder(norm_past_traj, traj_mask).squeeze(1)
        fused_context, _ = self.agent_map_attn(social_context, map_features)
        led_input = self.map_to_led(fused_context)
        _, guess_mean, _ = self.model_initializer(norm_past_traj, traj_mask, led_input)

        # 得到相对坐标预测 [B*A, K, T, 2]
        refined_traj_rel, scores = self.refinement_module(guess_mean, fused_context)

        # 4. 反归一化 (Relative -> Absolute)
        # [B*A, 1, 1, 2]
        last_pos_xy = last_pos_flat[..., :2].unsqueeze(1)
        refined_traj_abs = refined_traj_rel + last_pos_xy

        return refined_traj_abs