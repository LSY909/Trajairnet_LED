import torch
from torch import nn
import torch.nn.functional as F
from models.model_led_initializer import LEDInitializer as InitializationModel
from model.qcnet_modules import QCNetStyleRouteEncoder, QCNetAgentMapAttention, QCNetRefinementDecoder


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
        return h_feat + h_feat_


class TrajAirNet(nn.Module):
    def __init__(self, args):
        super(TrajAirNet, self).__init__()

        n_classes = int(args.preds / args.preds_step)

        '''修改为QCNet的部分'''
        self.n_clusters = getattr(args, 'n_clusters', 3)
        self.map_feature_dim = 128
        self.context_dim = 256
        self.k_pred = 6   # 预测模态数

        self.coord_dim = 3  #坐标维度
        self.output_dim = self.coord_dim*2

        # 历史轨迹编码器
        self.context_encoder = SocialEncoder(obs_len=args.obs)
        # 航线编码器
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

        # 4. Proposal 阶段(LED)
        self.map_to_led = nn.Linear(self.context_dim, 64)
        self.model_initializer = InitializationModel(
            t_h=args.obs,
            d_h=3,
            t_f=n_classes,
            d_f=3,
            k_pred=self.k_pred
        ).cuda()

        # 5. Refinement 阶段 (QCNet)
        self.refinement_module = QCNetRefinementDecoder(
            embed_dim=self.context_dim,
            future_steps=n_classes,
            num_modes=self.k_pred,
            output_dim=self.output_dim
        )

    def forward(self, x, y, adj, context, route_priors=None, sort=False):
        """
        x: [Batch, Agents, 3, Obs_Len]
        y: [Batch, Agents, 3, Pred_Len]
        """
        batch_size = x.shape[0]
        agent_num = x.shape[1]

        # 提取最后时刻位置作为原点: [B, A, 3, 11] -> [B, A, 3, 1]
        if x.shape[2] == 3 and x.shape[3] != 3:
            last_pos = x[:, :, :, -1].unsqueeze(-1)
            x_perm = x.permute(0, 1, 3, 2)  # [B, A, 11, 3]
            # [B, A, 1, 3] 用于广播
            last_pos_flat = last_pos.squeeze(-1).unsqueeze(2)
        else:
            x_perm = x
            last_pos_flat = x[:, :, -1, :].unsqueeze(2)

        # 输入归一化
        norm_past_traj = x_perm - last_pos_flat
        # 输入给 Encoder: [B*A, 11, 3]
        past_traj_input = norm_past_traj.reshape(batch_size * agent_num, -1, 3)


        # -------------------------------------------------------
        #  航线特征编码 (使用相对坐标)
        # -------------------------------------------------------
        if route_priors is not None:
            # route_priors: [B, A, C, T, 4]
            rel_coords = route_priors[..., :3]
            widths = route_priors[..., 3:]
            # 相对坐标编码
            norm_priors = torch.cat([rel_coords.to(x.device), widths.to(x.device)], dim=-1)
            map_features = self.route_encoder(norm_priors)
        else:
            map_features = torch.zeros(batch_size * agent_num, self.n_clusters, self.map_feature_dim).to(x.device)

        # -------------------------------------------------------
        # 历史轨迹编码
        # -------------------------------------------------------
        traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).to(x.device) + 1.0
        # 转为 [B*A, 3, 11] 传入
        social_context = self.context_encoder(past_traj_input, traj_mask).squeeze(1)
        fused_context, _ = self.agent_map_attn(social_context, map_features)
        led_input = self.map_to_led(fused_context)

        # LED 输入相对轨迹
        _, guess_mean, _ = self.model_initializer(past_traj_input, traj_mask, led_input)

        # 预测结果为6通道 [B*A, Modes, Time, 6]
        refined_traj_rel, scores = self.refinement_module(guess_mean, fused_context)
        pred_loc = refined_traj_rel[..., :3]
        pred_scale = F.softplus(refined_traj_rel[..., 3:]) + 1e-6

        # ==========================================
        # Loss 计算 (在相对坐标系下计算)
        # ==========================================
        if y.shape[2] == 3:
            y_perm = y.permute(0, 1, 3, 2)
        else:
            y_perm = y

        # 真值的相对坐标
        gt_traj_rel = (y_perm - last_pos_flat.to(y.device)).reshape(batch_size * agent_num, -1, 3)[..., :3]

        # 1. Find Best Mode
        # ------------------------------------------------
        diff = pred_loc - gt_traj_rel.unsqueeze(1)
        dist = torch.norm(diff, p=2, dim=-1).mean(dim=-1)  # [Batch, Modes]
        best_mode_idx = torch.argmin(dist, dim=-1)

        # 2. 提取最佳轨迹的 Mu 和 Scale
        batch_indices = torch.arange(refined_traj_rel.shape[0], device=x.device)

        best_traj_loc = pred_loc[batch_indices, best_mode_idx]  # [B*A, Time, 3]
        best_traj_scale = pred_scale[batch_indices, best_mode_idx]  # [B*A, Time, 3]

        # 3. 计算回归 Loss (Laplace NLL)
        l1_diff = torch.abs(best_traj_loc - gt_traj_rel)
        # 计算 NLL
        nll_loss = l1_diff / best_traj_scale + torch.log(best_traj_scale)
        # 求均值 (平均每个样本、每个时间步、每个坐标轴)
        reg_loss = nll_loss.mean()

        # 4. 计算分类 Loss
        cls_loss = F.cross_entropy(scores, best_mode_idx)
        return reg_loss, cls_loss

    def inference(self, x, y, adj, context, route_priors=None):
        batch_size = x.shape[0]
        agent_num = x.shape[1]

        # 0. 准备坐标
        if x.shape[2] == 3 and x.shape[3] != 3:
            last_pos_val = x[:, :, :, -1]
            last_pos_flat = last_pos_val.unsqueeze(2)  # [B, A, 1, 3]
            x_perm = x.permute(0, 1, 3, 2)
        else:
            last_pos_flat = x[:, :, -1, :].unsqueeze(2)
            x_perm = x

        # 归一化输入
        norm_past_traj = (x_perm - last_pos_flat).reshape(batch_size * agent_num, -1, 3)

        # 1. Map Encoding
        if route_priors is not None:
            rel_coords = route_priors[..., :3]
            widths = route_priors[..., 3:]
            norm_priors = torch.cat([rel_coords.to(x.device), widths.to(x.device)], dim=-1)
            map_features = self.route_encoder(norm_priors)
        else:
            map_features = torch.zeros(batch_size * agent_num, self.n_clusters, self.map_feature_dim).to(x.device)

        # 2. 网络前向
        traj_mask = torch.zeros(batch_size * agent_num, batch_size * agent_num).to(x.device) + 1.0
        social_context = self.context_encoder(norm_past_traj, traj_mask).squeeze(1)
        fused_context, _ = self.agent_map_attn(social_context, map_features)
        led_input = self.map_to_led(fused_context)
        _, guess_mean, _ = self.model_initializer(norm_past_traj, traj_mask, led_input)

        # 获取 6 通道输出，但 inference 通常只需要位置
        refined_traj_rel_all, scores = self.refinement_module(guess_mean, fused_context)

        # 只取前3个通道 (位置 x,y,z)，忽略后3个通道 (Scale)
        refined_traj_rel_pos = refined_traj_rel_all[..., :3]

        # 反归一化
        last_pos_xyz = last_pos_flat.view(batch_size * agent_num, 1, 1, 3).to(x.device)
        refined_traj_abs = refined_traj_rel_pos + last_pos_xyz

        return refined_traj_abs