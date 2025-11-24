# import math
# import torch
# import torch.nn as nn
# from torch.nn import Module, Linear
# import pdb
# from models.layers import PositionalEncoding, ConcatSquashLinear
#
#
# class st_encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         channel_in = 3
#         channel_out = 32
#         dim_kernel = 3
#         self.dim_embedding_key = 256
#         self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
#         self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)
#         self.relu = nn.ReLU()
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.kaiming_normal_(self.spatial_conv.weight)
#         nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
#         nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
#         nn.init.zeros_(self.spatial_conv.bias)
#         nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
#         nn.init.zeros_(self.temporal_encoder.bias_hh_l0)
#
#     def forward(self, X):
#         X_t = torch.transpose(X, 1, 2)
#         X_after_spatial = self.relu(self.spatial_conv(X_t))
#         X_embed = torch.transpose(X_after_spatial, 1, 2)
#         output_x, state_x = self.temporal_encoder(X_embed)
#         state_x = state_x.squeeze(0)
#         return state_x
#
#
# class social_transformer(nn.Module):
#     def __init__(self):
#         super(social_transformer, self).__init__()
#         self.encode_past = nn.Linear(33, 256, bias=False)
#         self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
#         self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
#
#     def forward(self, h, mask):
#         h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
#         h_feat_ = self.transformer_encoder(h_feat, mask)
#         h_feat = h_feat + h_feat_
#         return h_feat
#
#
# class TransformerDenoisingModel(Module):
#     def __init__(self, context_dim=256, tf_layer=2):
#         super().__init__()
#         self.encoder_context = social_transformer()
#         self.pos_emb = PositionalEncoding(d_model=2 * context_dim, dropout=0.1, max_len=24)
#
#         self.concat1 = ConcatSquashLinear(3, 2 * context_dim, context_dim + 3)
#         self.layer = nn.TransformerEncoderLayer(d_model=2 * context_dim, nhead=2, dim_feedforward=2 * context_dim)
#         self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
#         self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
#         self.concat4 = ConcatSquashLinear(context_dim, context_dim // 2, context_dim + 3)
#         self.linear = ConcatSquashLinear(context_dim // 2, 3, context_dim + 3)
#
#     # forward 方法 (用于训练时的 Loss 计算)
#     def forward(self, x, beta, context, mask, map_features=None):
#         batch_size = x.size(0)
#         beta = beta.view(batch_size, 1, 1)
#
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#
#         # 1. 确定使用的 Context (Map Fusion 逻辑)
#         # 如果 map_features 存在，它就是 trajairnet 传来的融合后的 final_context [B*N, 256]
#         if map_features is not None:
#             final_ctx = map_features.unsqueeze(1)
#         else:
#             # 否则使用原始 Social Context
#             final_ctx = self.encoder_context(context, mask)
#
#         time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
#
#         # 拼接 final_ctx
#         ctx_emb = torch.cat([time_emb, final_ctx], dim=-1)  # (B, 1, 256+3)
#
#         x = self.concat1(ctx_emb, x)
#         final_emb = x.permute(1, 0, 2)
#         final_emb = self.pos_emb(final_emb)
#
#         trans = self.transformer_encoder(final_emb).permute(1, 0, 2)
#         trans = self.concat3(ctx_emb, trans)
#         trans = self.concat4(ctx_emb, trans)
#         return self.linear(ctx_emb, trans)
#
#     # [修正 2] generate_accelerate 方法 (用于预测时的采样循环)
#     def generate_accelerate(self, x, beta, context, mask, map_features=None):
#         batch_size = x.size(0)
#         sample_num = x.shape[1]
#         points_num = x.shape[2]
#
#         beta = beta.view(beta.size(0), 1, 1)
#
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#
#         # 1. 确定使用的 Context
#         if map_features is not None:
#             final_ctx = map_features.unsqueeze(1)
#         else:
#             final_ctx = self.encoder_context(context, mask)
#
#         time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
#
#         #  拼接 final_ctx
#         ctx_emb = torch.cat([time_emb, final_ctx], dim=-1).repeat(1, sample_num, 1).unsqueeze(2)
#
#         x = self.concat1.batch_generate(ctx_emb, x).contiguous().view(-1, points_num, 512)
#         final_emb = x.permute(1, 0, 2)
#         final_emb = self.pos_emb(final_emb)
#
#         trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, sample_num, points_num, 512)
#
#         trans = self.concat3.batch_generate(ctx_emb, trans)
#         trans = self.concat4.batch_generate(ctx_emb, trans)
#         return self.linear.batch_generate(ctx_emb, trans)


import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear
import pdb
from models.layers import PositionalEncoding, ConcatSquashLinear


class st_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        channel_in = 3
        channel_out = 32
        dim_kernel = 3
        self.dim_embedding_key = 256
        self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        nn.init.zeros_(self.spatial_conv.bias)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, X):
        X_t = torch.transpose(X, 1, 2)
        X_after_spatial = self.relu(self.spatial_conv(X_t))
        X_embed = torch.transpose(X_after_spatial, 1, 2)
        output_x, state_x = self.temporal_encoder(X_embed)
        state_x = state_x.squeeze(0)
        return state_x


class social_transformer(nn.Module):
    def __init__(self):
        super(social_transformer, self).__init__()
        self.encode_past = nn.Linear(33, 256, bias=False)
        self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

    def forward(self, h, mask):
        h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
        h_feat_ = self.transformer_encoder(h_feat, mask)
        h_feat = h_feat + h_feat_
        return h_feat


class TransformerDenoisingModel(Module):
    def __init__(self, context_dim=256, tf_layer=2):
        super().__init__()
        self.encoder_context = social_transformer()
        self.pos_emb = PositionalEncoding(d_model=2 * context_dim, dropout=0.1, max_len=24)

        self.concat1 = ConcatSquashLinear(3, 2 * context_dim, context_dim + 3)
        self.layer = nn.TransformerEncoderLayer(d_model=2 * context_dim, nhead=2, dim_feedforward=2 * context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2 * context_dim, context_dim, context_dim + 3)
        self.concat4 = ConcatSquashLinear(context_dim, context_dim // 2, context_dim + 3)
        self.linear = ConcatSquashLinear(context_dim // 2, 3, context_dim + 3)

    # [核心修改] forward 接收 map_features
    def forward(self, x, beta, context, mask, map_features=None):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)

        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        # 1. 确定使用的 Context (Map Fusion 逻辑)
        # 如果 map_features 存在，它就是 trajairnet 传来的融合后的 final_context [B*N, 256]
        if map_features is not None:
            final_ctx = map_features.unsqueeze(1)  # [B, 1, 256]
        else:
            # 否则使用原始 Social Context
            final_ctx = self.encoder_context(context, mask)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)

        # 2. 拼接 final_ctx
        ctx_emb = torch.cat([time_emb, final_ctx], dim=-1)  # (B, 1, 256+3)

        x = self.concat1(ctx_emb, x)
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb).permute(1, 0, 2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)

    # [核心修改] generate_accelerate 接收 map_features
    def generate_accelerate(self, x, beta, context, mask, map_features=None):
        batch_size = x.size(0)
        sample_num = x.shape[1]
        points_num = x.shape[2]

        beta = beta.view(beta.size(0), 1, 1)

        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        # 1. 确定使用的 Context
        if map_features is not None:
            final_ctx = map_features.unsqueeze(1)  # [B, 1, 256]
        else:
            final_ctx = self.encoder_context(context, mask)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)

        # 2. 拼接 final_ctx 并处理维度
        ctx_emb = torch.cat([time_emb, final_ctx], dim=-1).repeat(1, sample_num, 1).unsqueeze(2)

        x = self.concat1.batch_generate(ctx_emb, x).contiguous().view(-1, points_num, 512)
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, sample_num, points_num, 512)

        trans = self.concat3.batch_generate(ctx_emb, trans)
        trans = self.concat4.batch_generate(ctx_emb, trans)
        return self.linear.batch_generate(ctx_emb, trans)