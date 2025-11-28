import torch
import torch.nn as nn
from models.layers import MLP, social_transformer, st_encoder


class LEDInitializer(nn.Module):
    def __init__(self, t_h: int = 11, d_h: int = 3, t_f: int = 12, d_f: int = 3, k_pred: int = 20):
        super(LEDInitializer, self).__init__()
        self.n = k_pred
        self.input_dim = t_h * d_h
        self.output_dim = t_f * d_f * k_pred
        self.fut_len = t_f

        # ============================================================
        # [修改] 适配 TrajAirNet 的 Map Encoder 输出
        # ============================================================
        # 输入维度：64 (来自 TrajAirNet 的 map_feature_dim)
        self.prior_input_dim = 64
        self.prior_feature_dim = 32
        self.prior_encoder = MLP(self.prior_input_dim, self.prior_feature_dim, hid_feat=(64, 64), activation=nn.ReLU())

        # 基础编码器
        self.social_encoder = social_transformer(t_h)
        self.ego_var_encoder = st_encoder()
        self.ego_mean_encoder = st_encoder()
        self.ego_scale_encoder = st_encoder()
        self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

        # [修改] Mean Decoder 输入维度: Ego(256) + Social(256) + Prior(32) = 544
        self.mean_decoder = MLP(256 * 2 + self.prior_feature_dim, t_f * d_f, hid_feat=(256, 128), activation=nn.ReLU())

        self.var_decoder = MLP(256 * 2 + 32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
        self.scale_decoder = MLP(256 * 2, 1, hid_feat=(256, 128), activation=nn.ReLU())

    def forward(self, x, mask=None, map_features=None):
        '''
        map_features: [Batch*Agent, 64]
        '''
        var_num = 3
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        social_embed = self.social_encoder(x, mask).squeeze(1)
        ego_var_embed = self.ego_var_encoder(x)
        ego_mean_embed = self.ego_mean_encoder(x)
        ego_scale_embed = self.ego_scale_encoder(x)

        # [处理] 航线特征
        if map_features is not None:
            # map_features 已经是 64 维特征
            priors_embed = self.prior_encoder(map_features)  # -> 32维
        else:
            priors_embed = torch.zeros(x.size(0), self.prior_feature_dim).to(x.device)

        # [融合]
        mean_total = torch.cat((ego_mean_embed, social_embed, priors_embed), dim=-1)
        guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, var_num)

        scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
        guess_scale = self.scale_decoder(scale_total)
        guess_scale_feat = self.scale_encoder(guess_scale)
        var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
        guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, var_num)

        return guess_var, guess_mean, guess_scale