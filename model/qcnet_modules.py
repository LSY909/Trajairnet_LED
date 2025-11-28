# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# #
# #
# # class QCNetGeometryExtractor(nn.Module):
# #     """
# #     对应 QCNet 论文 Section 3.2 "Scene Element Embedding"
# #     计算相对几何特征：位置、朝向向量、幅值。
# #
# #     [修正点]: 修复了差分向量计算时最后一个点变为 0 的问题。
# #     """
# #
# #     def __init__(self):
# #         super().__init__()
# #
# #     def forward(self, route_priors):
# #         # route_priors: [Batch, Agents, n_clusters, Length=12, Channels=4]
# #         # Channels: (x, y, z, width)
# #
# #         # 1. 基础特征分离
# #         pos = route_priors[..., :3]  # (x, y, z)
# #         width = route_priors[..., 3:]  # (width)
# #
# #         # 2. 计算差分向量 (P_t+1 - P_t)
# #         # [修正逻辑] 先计算前 L-1 个点的差分，最后一个点沿用倒数第二个点的趋势
# #         diff_raw = pos[..., 1:, :] - pos[..., :-1, :]  # [..., L-1, 3]
# #         last_diff = diff_raw[..., -1:, :]  # [..., 1, 3] 重复最后一个差分
# #         diff_vector = torch.cat([diff_raw, last_diff], dim=-2)  # [..., L, 3]
# #
# #         # 3. 计算朝向 (Orientation) -> 转换为向量 [cos, sin]
# #         # 使用 atan2 计算方位角 (主要在 xy 平面)
# #         # 注意：如果是 0 向量，atan2(0,0) = 0，但有了上面的修正，一般不会出现纯 0
# #         azimuth = torch.atan2(diff_vector[..., 1], diff_vector[..., 0])
# #         orient_vector = torch.stack([azimuth.cos(), azimuth.sin()], dim=-1)  # [..., 2]
# #
# #         # 4. 计算幅值 (Magnitude/Velocity)
# #         magnitude = torch.norm(diff_vector[..., :2], p=2, dim=-1, keepdim=True)  # [..., 1]
# #
# #         # 5. 组合特征
# #         # 最终特征维度: 3(pos) + 1(width) + 2(orient) + 1(mag) = 7维
# #         geometry_features = torch.cat([
# #             pos,  # 绝对位置
# #             width,  # 语义属性
# #             orient_vector,  # 几何属性：方向
# #             magnitude  # 几何属性：位移大小
# #         ], dim=-1)
# #
# #         return geometry_features
# #
# #
# # class QCNetStyleRouteEncoder(nn.Module):
# #     """
# #     对应 QCNet 论文 Section 3.2 "Scene Element Embedding"
# #     将一系列的点聚合为一个 Polygon Embedding。
# #     """
# #
# #     def __init__(self, in_channels=7, hidden_dim=64, out_dim=128):
# #         super().__init__()
# #         self.geometry_extractor = QCNetGeometryExtractor()
# #
# #         # Point-level 处理
# #         self.point_net = nn.Sequential(
# #             nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
# #             nn.BatchNorm1d(hidden_dim),
# #             nn.ReLU(),
# #             nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
# #             nn.BatchNorm1d(hidden_dim),
# #             nn.ReLU()
# #         )
# #
# #         # 聚合层
# #         self.aggregator = nn.Sequential(
# #             nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1),
# #             nn.BatchNorm1d(out_dim),
# #             nn.ReLU(),
# #             nn.AdaptiveMaxPool1d(1)  # Max Pooling 聚合
# #         )
# #
# #     def forward(self, route_priors):
# #         # 1. 提取几何特征
# #         # x shape: [B, A, Clusters, 12, 7]
# #         x = self.geometry_extractor(route_priors)
# #
# #         # 2. 维度重组：Flatten 适配 Conv1d
# #         batch, agents, clusters, seq_len, feats = x.shape
# #         x = x.view(-1, seq_len, feats)  # [B*A*Clusters, L, 7]
# #
# #         # 3. 编码
# #         x = x.permute(0, 2, 1)  # [N, C, L]
# #         x = self.point_net(x)
# #         x = self.aggregator(x)  # [N, Out_Dim, 1]
# #
# #         # 4. 恢复结构
# #         # 输出: [Batch*Agents, Clusters, Out_Dim]
# #         return x.view(batch * agents, clusters, -1)
# #
# #
# # class QCNetAgentMapAttention(nn.Module):
# #     """
# #     Agent-Map 融合模块
# #     """
# #
# #     def __init__(self, agent_dim=256, map_dim=128, hidden_dim=256, num_heads=4, dropout=0.1):
# #         super().__init__()
# #
# #         self.q_proj = nn.Linear(agent_dim, hidden_dim)
# #         self.k_proj = nn.Linear(map_dim, hidden_dim)
# #         self.v_proj = nn.Linear(map_dim, hidden_dim)
# #
# #         self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
# #
# #         self.norm1 = nn.LayerNorm(hidden_dim)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.ffn = nn.Sequential(
# #             nn.Linear(hidden_dim, hidden_dim * 2),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(hidden_dim * 2, hidden_dim)
# #         )
# #         self.norm2 = nn.LayerNorm(hidden_dim)
# #
# #     def forward(self, agent_feat, map_feat):
# #         """
# #         agent_feat: [B*A, Agent_Dim]
# #         map_feat:   [B*A, Clusters, Map_Dim]
# #         """
# #         # Query: [B*A, 1, Hidden]
# #         Q = self.q_proj(agent_feat).unsqueeze(1)
# #
# #         # Key/Value: [B*A, Clusters, Hidden]
# #         K = self.k_proj(map_feat)
# #         V = self.v_proj(map_feat)
# #
# #         # Attention
# #         attn_out, weights = self.attn(Q, K, V)
# #
# #         # Residual & Norm
# #         x = Q + self.dropout(attn_out)
# #         x = self.norm1(x)
# #
# #         # FFN
# #         x2 = self.ffn(x)
# #         x = x + x2
# #         x = self.norm2(x)
# #
# #         return x.squeeze(1), weights
# #
# #
# # class QCNetRefinementDecoder(nn.Module):
# #     """
# #     [QCNet Anchor-Based Refinement Module 适配版]
# #
# #     [修正点]:
# #     1. 修正了 Transformer 输入维度，确保 6 个 Mode 之间可以进行 Self-Attention 交互。
# #     2. 加入了 mode_emb 以区分不同的模态。
# #     """
# #
# #     def __init__(self, embed_dim=256, future_steps=12, num_modes=6, num_layers=2):
# #         super().__init__()
# #         self.embed_dim = embed_dim
# #         self.future_steps = future_steps
# #         self.num_modes = num_modes
# #
# #         # 1. Anchor Encoder
# #         self.anchor_encoder = nn.GRU(
# #             input_size=2,
# #             hidden_size=embed_dim,
# #             batch_first=True
# #         )
# #
# #         # 2. Transformer Decoder
# #         # batch_first=True -> 输入形状为 (Batch, Seq, Feature)
# #         decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512, batch_first=True)
# #         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
# #
# #         # [新增] Learnable Mode Embedding
# #         # 帮助模型区分第1条轨迹、第2条轨迹...
# #         self.mode_emb = nn.Parameter(torch.randn(1, num_modes, embed_dim))
# #
# #         # 3. Prediction Heads
# #         self.reg_head = nn.Linear(embed_dim, future_steps * 2)
# #         self.cls_head = nn.Linear(embed_dim, 1)
# #
# #     def forward(self, anchors, scene_context):
# #         """
# #         anchors: [B*A, Num_Modes, Future_Steps, 2] (LED输出的粗轨迹)
# #         scene_context: [B*A, Embed_Dim] (Encoder输出的特征)
# #         """
# #         batch_size_agents, num_modes, steps, dim = anchors.shape
# #
# #         # --- Step 1: Encode Anchors ---
# #         # 展平以通过 GRU: [B*A*K, T, 2]
# #         flat_anchors = anchors.view(-1, steps, dim)[..., :2]
# #         _, hidden = self.anchor_encoder(flat_anchors)
# #         # hidden: [1, B*A*K, Embed] -> permute -> [B*A*K, 1, Embed] (这里还是扁平的)
# #
# #         # --- Step 2: Prepare Query (维度重组) ---
# #         # [关键] 将 Batch 和 Modes 维度分开，以便 Transformer 在 Modes 之间做 Attention
# #         # Target Shape: [B*A, Num_Modes, Embed]
# #         query = hidden.permute(1, 0, 2).view(batch_size_agents, num_modes, self.embed_dim)
# #
# #         # 加上 Mode Embedding
# #         query = query + self.mode_emb
# #
# #         # --- Step 3: Prepare Memory ---
# #         # Scene Context 是每个 Agent 一个特征，需要扩展维度适配 Transformer
# #         # Memory Shape: [B*A, 1, Embed] (Seq_len=1, 因为是全局特征)
# #         memory = scene_context.unsqueeze(1)
# #
# #         # --- Step 4: Refinement (Transformer) ---
# #         # Self-Attention 发生在 dim=1 (Num_Modes) 之间 -> Mode-to-Mode Interaction
# #         # Cross-Attention 发生在 Query(Modes) 和 Memory(Scene) 之间 -> Mode-to-Scene Interaction
# #         out = self.transformer_decoder(tgt=query, memory=memory)  # [B*A, Num_Modes, Embed]
# #
# #         # --- Step 5: Prediction ---
# #         # 回归 Offset
# #         offsets = self.reg_head(out).view(batch_size_agents, num_modes, steps, 2)
# #         refined_traj = anchors[..., :2] + offsets
# #
# #         # 预测概率 (squeeze 掉最后一个维度 1)
# #         scores = self.cls_head(out).squeeze(-1)  # [B*A, Num_Modes]
# #
# #         return refined_traj, scores
#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class QCNetGeometryExtractor(nn.Module):
#     """
#     对应 QCNet 论文 Section 3.2 "Scene Element Embedding"
#     计算相对几何特征：位置、朝向向量、幅值。
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, route_priors):
#         # route_priors: [Batch, Agents, n_clusters, Length=12, Channels=4]
#         # Channels: (x, y, z, width)
#
#         pos = route_priors[..., :3]  # (x, y, z)
#         width = route_priors[..., 3:]  # (width)
#
#         # 计算差分向量 (P_t+1 - P_t)
#         next_pos = torch.cat([pos[..., 1:, :], pos[..., -1:, :]], dim=-2)
#         diff_vector = next_pos - pos  # (dx, dy, dz)
#
#         # 计算朝向 (Orientation) -> [cos, sin]
#         azimuth = torch.atan2(diff_vector[..., 1], diff_vector[..., 0])
#         orient_vector = torch.stack([azimuth.cos(), azimuth.sin()], dim=-1)  # [..., 2]
#
#         # 计算幅值
#         magnitude = torch.norm(diff_vector[..., :2], p=2, dim=-1, keepdim=True)  # [..., 1]
#
#         # 组合特征: 3+1+2+1 = 7维
#         geometry_features = torch.cat([
#             pos,
#             width,
#             orient_vector,
#             magnitude
#         ], dim=-1)
#
#         return geometry_features
#
#
# class QCNetStyleRouteEncoder(nn.Module):
#     """
#     对应 QCNet 论文 Section 3.2 "Scene Element Embedding"
#     将一系列的点聚合为一个 Polygon Embedding。
#     使用 1D 卷积替代 Fourier+MLP 处理定长序列。
#     """
#
#     def __init__(self, in_channels=7, hidden_dim=64, out_dim=128):
#         super().__init__()
#         self.geometry_extractor = QCNetGeometryExtractor()
#
#         self.point_net = nn.Sequential(
#             nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU()
#         )
#
#         self.aggregator = nn.Sequential(
#             nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1),
#             nn.BatchNorm1d(out_dim),
#             nn.ReLU(),
#             nn.AdaptiveMaxPool1d(1)
#         )
#
#     def forward(self, route_priors):
#         # 1. 提取几何特征
#         x = self.geometry_extractor(route_priors)  # [B, A, C, L, 7]
#
#         # 2. 维度重组
#         batch, agents, clusters, seq_len, feats = x.shape
#         x = x.view(-1, seq_len, feats)
#
#         # 3. 编码
#         x = x.permute(0, 2, 1)  # [N, 7, 12]
#         x = self.point_net(x)
#         x = self.aggregator(x)  # [N, Out, 1]
#
#         # 4. 恢复结构
#         return x.view(batch * agents, clusters, -1)
#
#
# class QCNetAgentMapAttention(nn.Module):
#     """
#     对应 QCNet 论文 Section 3.2 "Factorized Attention for Agent Encoding"
#     """
#
#     def __init__(self, agent_dim=256, map_dim=128, hidden_dim=256, num_heads=4, dropout=0.1):
#         super().__init__()
#
#         self.q_proj = nn.Linear(agent_dim, hidden_dim)
#         self.k_proj = nn.Linear(map_dim, hidden_dim)
#         self.v_proj = nn.Linear(map_dim, hidden_dim)
#
#         self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
#
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
#         self.norm2 = nn.LayerNorm(hidden_dim)
#
#     def forward(self, agent_feat, map_feat):
#         # Query: [B*A, 1, H]
#         Q = self.q_proj(agent_feat).unsqueeze(1)
#         # Key/Value: [B*A, Clusters, H]
#         K = self.k_proj(map_feat)
#         V = self.v_proj(map_feat)
#
#         attn_out, weights = self.attn(Q, K, V)
#
#         x = self.q_proj(agent_feat).unsqueeze(1) + self.dropout(attn_out)
#         x = self.norm1(x)
#
#         x = x + self.ffn(x)
#         x = self.norm2(x)
#
#         return x.squeeze(1), weights
#
#
# class QCNetRefinementDecoder(nn.Module):
#     """
#     [QCNet Refinement Module]
#     输入粗轨迹(Anchors)和上下文，输出偏移量和概率。
#     """
#
#     def __init__(self, embed_dim=256, future_steps=12, num_modes=6, num_layers=2):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.future_steps = future_steps
#         self.num_modes = num_modes
#
#         # Anchor Encoder (GRU)
#         self.anchor_encoder = nn.GRU(
#             input_size=2,  # (x, y)
#             hidden_size=embed_dim,
#             batch_first=True
#         )
#
#         # Refinement Transformer
#         decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512, batch_first=True)
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
#
#         # Prediction Heads
#         self.reg_head = nn.Linear(embed_dim, future_steps * 2)  # dx, dy
#         self.cls_head = nn.Linear(embed_dim, 1)  # Probability
#
#     def forward(self, anchors, scene_context):
#         """
#         anchors: [B*A, K, T, 3] (取前2维 x,y 使用)
#         scene_context: [B*A, D]
#         """
#         batch_size_agents, num_modes, steps, dim = anchors.shape
#
#         # 1. Anchor Encoding
#         # Flatten: [B*A*K, T, 2]
#         flat_anchors = anchors.view(-1, steps, dim)[..., :2]
#
#         # GRU -> Query [B*A*K, 1, D]
#         _, hidden = self.anchor_encoder(flat_anchors)
#         query = hidden.permute(1, 0, 2)
#
#         # 2. Scene Context -> Memory [B*A*K, 1, D]
#         memory = scene_context.unsqueeze(1).repeat(1, num_modes, 1).view(-1, 1, self.embed_dim)
#
#         # 3. Decoding
#         out = self.transformer_decoder(tgt=query, memory=memory)  # [B*A*K, 1, D]
#
#         # 4. Heads
#         offsets = self.reg_head(out).view(batch_size_agents, num_modes, steps, 2)
#         refined_traj = anchors[..., :2] + offsets
#
#         scores = self.cls_head(out).view(batch_size_agents, num_modes)
#
#         return refined_traj, scores


import torch
import torch.nn as nn
import torch.nn.functional as F


class QCNetGeometryExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, route_priors):
        pos = route_priors[..., :3]
        width = route_priors[..., 3:]
        next_pos = torch.cat([pos[..., 1:, :], pos[..., -1:, :]], dim=-2)
        diff_vector = next_pos - pos
        azimuth = torch.atan2(diff_vector[..., 1], diff_vector[..., 0])
        orient_vector = torch.stack([azimuth.cos(), azimuth.sin()], dim=-1)
        magnitude = torch.norm(diff_vector[..., :2], p=2, dim=-1, keepdim=True)
        geometry_features = torch.cat([pos, width, orient_vector, magnitude], dim=-1)
        return geometry_features


class QCNetStyleRouteEncoder(nn.Module):
    def __init__(self, in_channels=7, hidden_dim=64, out_dim=128):
        super().__init__()
        self.geometry_extractor = QCNetGeometryExtractor()
        self.point_net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
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


class QCNetAgentMapAttention(nn.Module):
    def __init__(self, agent_dim=256, map_dim=128, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(agent_dim, hidden_dim)
        self.k_proj = nn.Linear(map_dim, hidden_dim)
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
    def __init__(self, embed_dim=256, future_steps=12, num_modes=6, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.num_modes = num_modes

        self.anchor_encoder = nn.GRU(
            input_size=2,
            hidden_size=embed_dim,
            batch_first=True
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.reg_head = nn.Linear(embed_dim, future_steps * 2)
        self.cls_head = nn.Linear(embed_dim, 1)

    def forward(self, anchors, scene_context):
        """
        anchors: [B*A, T, 3] (单模态) 或 [B*A, K, T, 3] (多模态)
        scene_context: [B*A, D]
        """
        # [关键修复] 自动处理维度
        if anchors.dim() == 3:
            # Case 1: anchors 是 [N, T, 3]，说明 LED 只输出了一个均值
            # 我们需要把它复制扩展 K 次作为初始 Anchors
            anchors = anchors.unsqueeze(1).repeat(1, self.num_modes, 1, 1)

        batch_size_agents, num_modes, steps, dim = anchors.shape

        # 1. Anchor Encoding
        # Flatten: [B*A*K, T, 2]
        flat_anchors = anchors.view(-1, steps, dim)[..., :2]

        # GRU -> Query [B*A*K, 1, D]
        _, hidden = self.anchor_encoder(flat_anchors)
        query = hidden.permute(1, 0, 2)

        # 2. Scene Context -> Memory [B*A*K, 1, D]
        memory = scene_context.unsqueeze(1).repeat(1, num_modes, 1).view(-1, 1, self.embed_dim)

        # 3. Decoding
        out = self.transformer_decoder(tgt=query, memory=memory)  # [B*A*K, 1, D]

        # 4. Heads
        # 注意: reg_head 输出的是相对于 Anchor 的偏移量
        offsets = self.reg_head(out).view(batch_size_agents, num_modes, steps, 2)
        refined_traj = anchors[..., :2] + offsets

        scores = self.cls_head(out).view(batch_size_agents, num_modes)

        return refined_traj, scores