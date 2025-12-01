import torch
import torch.nn as nn
import torch.nn.functional as F

'''基于粗糙的轨迹，结合上下文信息，输出一个更精准的带有不确定度的最终轨迹'''

# 输入route_priors（航线）提取路径的几何特征
class QCNetGeometryExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, route_priors):
        pos = route_priors[..., :3]
        width = route_priors[..., 3:]
        next_pos = torch.cat([pos[..., 1:, :], pos[..., -1:, :]], dim=-2)
        diff_vector = next_pos - pos   #位移向量
        azimuth = torch.atan2(diff_vector[..., 1], diff_vector[..., 0])  #航向角
        orient_vector = torch.stack([azimuth.cos(), azimuth.sin()], dim=-1)
        magnitude = torch.norm(diff_vector[..., :3], p=2, dim=-1, keepdim=True) #两个航路点之间的实际距离
        geometry_features = torch.cat([pos, width, orient_vector, magnitude], dim=-1)
        return geometry_features

# 航线/地图编码器--把一条由多个点组成的折线（一段航路），压缩成一个固定长度的特征向量
class QCNetStyleRouteEncoder(nn.Module):
    def __init__(self, in_channels=7, hidden_dim=64, out_dim=128):
        super().__init__()
        self.geometry_extractor = QCNetGeometryExtractor()
        #局部特征处理
        self.point_net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        # 全局特征聚合
        self.aggregator = nn.Sequential(
            nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

    def forward(self, route_priors):
        x = self.geometry_extractor(route_priors)
        batch, agents, clusters, seq_len, feats = x.shape
        x = x.view(-1, seq_len, feats)
        x = x.permute(0, 2, 1)
        x = self.point_net(x)
        x = self.aggregator(x)
        return x.view(batch * agents, clusters, -1)

# 交叉注意力模块 agent-map Attention
class QCNetAgentMapAttention(nn.Module):
    def __init__(self, agent_dim=256, map_dim=128, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(agent_dim, hidden_dim)  #agent当前状态或意图
        self.k_proj = nn.Linear(map_dim, hidden_dim) #地图上所有元素的索引
        self.v_proj = nn.Linear(map_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, agent_feat, map_feat):
        Q = self.q_proj(agent_feat).unsqueeze(1)
        K = self.k_proj(map_feat)
        V = self.v_proj(map_feat)
        attn_out, weights = self.attn(Q, K, V)
        x = self.q_proj(agent_feat).unsqueeze(1) + self.dropout(attn_out)
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x.squeeze(1), weights


class QCNetRefinementDecoder(nn.Module):
    def __init__(self, embed_dim=256, future_steps=12, num_modes=6, num_layers=2, output_dim=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.output_dim = output_dim  # 记录输出维度

        self.anchor_encoder = nn.GRU(
            input_size=3,
            hidden_size=embed_dim,
            batch_first=True
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.reg_head = nn.Linear(embed_dim, future_steps * output_dim)
        self.cls_head = nn.Linear(embed_dim, 1)

    def forward(self, anchors, scene_context):
        """
        anchors: [B*A, T, 3] (单模态) 或 [B*A, K, T, 3] (多模态)
        scene_context: [B*A, D]
        """
        # 自动处理维度
        if anchors.dim() == 3:
            # 初始 Anchors
            anchors = anchors.unsqueeze(1).repeat(1, self.num_modes, 1, 1)

        batch_size_agents, num_modes, steps, dim = anchors.shape

        # 1. Anchor Encoding
        # Flatten: [B*A*K, T, 3]
        flat_anchors = anchors.view(-1, steps, dim)[..., :3]

        # GRU -> Query [B*A*K, 1, D]，使用GRU读取整条anchor轨迹(x,y,z)
        _, hidden = self.anchor_encoder(flat_anchors)
        query = hidden.permute(1, 0, 2)

        # 2. Scene Context -> Memory [B*A*K, 1, D]
        memory = scene_context.unsqueeze(1).repeat(1, num_modes, 1).view(-1, 1, self.embed_dim)

        # 3. Decoding
        out = self.transformer_decoder(tgt=query, memory=memory)  # [B*A*K, 1, D]

        # 1.  线性层映射 [B, K, T, 6]
        raw_pred = self.reg_head(out).view(batch_size_agents, num_modes, steps, self.output_dim)

        # 2. 拆分数据：前3个是位置偏移(Offsets)，后3个是不确定度(Scale)
        loc_offsets = raw_pred[..., :3]  # (dx, dy, dz)
        raw_scales = raw_pred[..., 3:]  # (bx, by, bz)

        # 3.把偏移量加到 Anchor 上
        refined_loc = anchors[..., :3] + loc_offsets

        # 4. 保持输出维度为 6
        refined_traj = torch.cat([refined_loc, raw_scales], dim=-1)
        # 打分
        scores = self.cls_head(out).view(batch_size_agents, num_modes)

        return refined_traj, scores