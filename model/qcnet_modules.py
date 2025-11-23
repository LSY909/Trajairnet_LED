import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. 傅里叶位置编码 (Fourier Embedding)
# 作用：将低维连续坐标 (x,y,z) 映射到高维频率空间，捕捉精细几何结构
# ============================================================
class FourierEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 生成频率带: 2^0, 2^1, ..., 2^(N-1)
        self.freq_bands = 2 ** torch.linspace(0, num_freq_bands - 1, num_freq_bands)

        # 输出维度: input_dim * (2 * num_freq + 1)
        # 解释: 每个坐标 x 变成 [x, sin(x), cos(x), sin(2x), cos(2x)...]
        self.out_dim = input_dim * (2 * num_freq_bands + 1)

        # 映射到隐藏层维度的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: [..., input_dim]
        输出: [..., hidden_dim]
        """
        # 在最后一个维度扩展，准备进行广播乘法
        x_expanded = x.unsqueeze(-1)  # [..., D, 1]

        # 将频率带移动到与输入相同的设备上
        freq_bands = self.freq_bands.to(x.device).reshape(1, 1, -1)  # [1, 1, K]

        # 计算频率项: x * freq * pi
        x_freq = x_expanded * freq_bands * math.pi  # [..., D, K]

        # 计算 Sin 和 Cos
        x_sin = torch.sin(x_freq)
        x_cos = torch.cos(x_freq)

        # 拼接: [原始x, sin特征, cos特征]
        z = torch.cat([x_expanded, x_sin, x_cos], dim=-1)  # [..., D, 2K+1]

        # 展平最后两个维度
        z = z.reshape(*x.shape[:-1], -1)  # [..., D * (2K+1)]

        return self.mlp(z)


# ============================================================
# 2. QCNet 风格航线编码器 (Route Encoder)
# 作用：先做傅里叶变换，再用双向 GRU 提取整条航线的形状特征
# ============================================================
class QCNetRouteEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_freq_bands=6):
        super(QCNetRouteEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        # 1. 几何特征增强 (Fourier)
        self.fourier_embed = FourierEmbedding(input_dim, hidden_dim, num_freq_bands)

        # 2. 序列建模 (Sequence Modeling)
        # 使用双向 GRU 捕捉航线的前后文形状关系
        self.sequence_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # 双向 GRU 输出是 2 * hidden_dim，需要投影回 hidden_dim
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, route_priors):
        """
        输入: [Batch, Agent, K_Clusters, Time, Dim] (例如: B, 7, 3, 12, 3)
        输出: [Batch*Agent, K_Clusters, Hidden_Dim] (例如: B*7, 3, 64)
        """
        b, a, k, t, d = route_priors.shape

        # 1. 展平 Batch 和 Agent 维度，统一处理
        # [B*A, K, T, D]
        x = route_priors.reshape(b * a, k, t, d)

        # 2. Fourier Embedding
        # [B*A, K, T, Hidden]
        x_emb = self.fourier_embed(x)

        # 3. 序列编码
        # 合并前两个维度以便 GRU 并行处理所有航线: [B*A*K, T, Hidden]
        x_flat = x_emb.view(-1, t, self.hidden_dim)

        # GRU 处理
        # output: [Batch, Seq, 2*Hidden]
        # h_n: [Num_Layers * Num_Directions, Batch, Hidden] -> [2, B*A*K, Hidden]
        _, h_n = self.sequence_encoder(x_flat)

        # 拼接双向隐状态 (Forward + Backward) -> [B*A*K, 2*Hidden]
        route_feat = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        # 投影回目标维度 -> [B*A*K, Hidden]
        route_feat = self.out_proj(route_feat)

        # 还原维度 -> [B*A, K, Hidden]
        # 这里保留了 K (Clusters) 维度，供后续 Attention 挑选
        return route_feat.view(b * a, k, -1)


# ============================================================
# 3. 智能体-地图 交叉注意力 (Agent-Map Cross Attention)
# 作用：让 Agent (Query) 从 K 条航线 (Key/Value) 中动态选择最重要的一条
# ============================================================
class AgentMapCrossAttention(nn.Module):
    def __init__(self, agent_dim=256, map_dim=64, hidden_dim=256, num_heads=4):
        super(AgentMapCrossAttention, self).__init__()

        # 投影层：将 Agent 和 Map 映射到同一 Attention 空间
        self.q_proj = nn.Linear(agent_dim, hidden_dim)
        self.k_proj = nn.Linear(map_dim, hidden_dim)
        self.v_proj = nn.Linear(map_dim, hidden_dim)

        # 多头注意力机制
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # 残差连接和归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(agent_dim)
        self.dropout = nn.Dropout(0.1)

        # 最终输出融合层: 将原始 Agent 特征与提取到的 Map 特征融合
        self.out_proj = nn.Linear(hidden_dim + agent_dim, agent_dim)

    def forward(self, agent_feat, map_feat):
        """
        输入:
            agent_feat: [B*A, Agent_Dim] (社交上下文)
            map_feat:   [B*A, K, Map_Dim] (K条航线的特征)
        输出:
            fused_context: [B*A, Agent_Dim]
        """
        # 1. 准备 Q, K, V
        # Query: Agent 自身 [B*A, 1, Hidden] (增加序列维度 1)
        query = self.q_proj(agent_feat).unsqueeze(1)

        # Key, Value: K 条航线 [B*A, K, Hidden]
        key = self.k_proj(map_feat)
        value = self.v_proj(map_feat)

        # 2. Attention 计算
        # attn_output: [B*A, 1, Hidden] - 加权后的航线特征
        # attn_weights: [B*A, 1, K] - 模型对每条航线的关注度
        attn_output, attn_weights = self.attn(query, key, value)

        # 3. Add & Norm (针对 Map Context)
        map_context = self.norm1(attn_output.squeeze(1))
        map_context = self.dropout(map_context)

        # 4. 融合: [原始 Agent 特征, 注意力提取的 Map 特征]
        combined = torch.cat([agent_feat, map_context], dim=-1)

        # 5. 最终投影与残差
        # [B*A, Agent_Dim]
        out = self.out_proj(combined)
        return self.norm2(out + agent_feat)  # 再次残差连接，保证不丢失原始信息