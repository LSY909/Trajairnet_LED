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

## 社会交互编码
class social_transformer(nn.Module):
    def __init__(self):
        super(social_transformer, self).__init__()
        self.encode_past = nn.Linear(33, 256, bias=False)
        # self.encode_past = nn.Linear(27, 256, bias=False)
        self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
    def forward(self, h, mask):
        '''
        h: batch_size, t, 2
        '''
        # print(h.shape)
        h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
        # print(h_feat.shape)
        # n_samples, 1, 64
        h_feat_ = self.transformer_encoder(h_feat, mask)
        h_feat = h_feat + h_feat_

        return h_feat

## 扩散模型
class TransformerDenoisingModel(Module):

    def __init__(self, context_dim=256, tf_layer=2, route_priors_dim=0, dense_future_dim=0):
        super().__init__()
        self.encoder_context = social_transformer()
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        # 维度可能不一致
        self.concat1 = ConcatSquashLinear(3, 2*context_dim, context_dim+3)
        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=2, dim_feedforward=2*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, 3, context_dim+3)
        # optional projections to fold external priors/dense features into encoder context
        self.context_dim = context_dim
        if route_priors_dim and route_priors_dim > 0:
            self.route_priors_proj = nn.Linear(route_priors_dim, context_dim)
        else:
            self.route_priors_proj = None
        if dense_future_dim and dense_future_dim > 0:
            self.dense_future_proj = nn.Linear(dense_future_dim, context_dim)
        else:
            self.dense_future_proj = None
        


    def forward(self, x, beta, context, mask, route_priors=None, dense_future_feat=None):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        context = self.encoder_context(context, mask)
        # expected shapes:
        #  - route_priors: (B, 1, route_priors_dim) or (B, route_priors_dim)
        #  - dense_future_feat: (B, 1, dense_future_dim) or (B, dense_future_dim)
        if route_priors is not None and self.route_priors_proj is not None:
            rp = route_priors
            if rp.dim() == 2:
                rp = rp.unsqueeze(1)
            rp_proj = self.route_priors_proj(rp)  # -> (B,1,context_dim)
            context = context + rp_proj
        if dense_future_feat is not None and self.dense_future_proj is not None:
            df = dense_future_feat
            if df.dim() == 2:
                df = df.unsqueeze(1)
            df_proj = self.dense_future_proj(df)  # -> (B,1,context_dim)
            context = context + df_proj
        # context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
        
        x = self.concat1(ctx_emb, x)
        final_emb = x.permute(1,0,2)
        final_emb = self.pos_emb(final_emb)
        
        trans = self.transformer_encoder(final_emb).permute(1,0,2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)
    
    def generate_accelerate(self, x, beta, context, mask, route_priors=None, dense_future_feat=None):
        #pdb.set_trace()

        batch_size = x.size(0)
        sample_num = x.shape[1]
        points_num = x.shape[2]

        beta = beta.view(beta.size(0), 1, 1)          # (B, 1, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        context = self.encoder_context(context, mask)
        # integrate priors/dense features into context (mirror forward)
        if route_priors is not None and self.route_priors_proj is not None:
            rp = route_priors
            if rp.dim() == 2:
                rp = rp.unsqueeze(1)
            if rp.dim() == 3 and rp.size(1) != 1:
                rp = rp.mean(dim=1, keepdim=True)
            rp_proj = self.route_priors_proj(rp)  # -> (B,1,context_dim)
            context = context + rp_proj
        if dense_future_feat is not None and self.dense_future_proj is not None:
            df = dense_future_feat
            if df.dim() == 2:
                df = df.unsqueeze(1)
            if df.dim() == 3 and df.size(1) != 1:
                df = df.mean(dim=1, keepdim=True)
            df_proj = self.dense_future_proj(df)  # -> (B,1,context_dim)
            context = context + df_proj
        #pdb.set_trace()
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        # time_emb: [11, 1, 3]
        # context: [11, 1, 256]
        ctx_emb = torch.cat([time_emb, context], dim=-1).repeat(1, sample_num, 1).unsqueeze(2)
        # x: 11, 10, 20, 2
        # ctx_emb: 11, 10, 1, 259
        x = self.concat1.batch_generate(ctx_emb, x).contiguous().view(-1, points_num, 512)
        # x: 110, 20, 512
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)
        
        trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, sample_num, points_num, 512)
        # trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, 10, 15, 512)
        # trans: 11【智能体数量】, 10【batch_size】, 20, 512
        trans = self.concat3.batch_generate(ctx_emb, trans)
        trans = self.concat4.batch_generate(ctx_emb, trans)
        return self.linear.batch_generate(ctx_emb, trans)
