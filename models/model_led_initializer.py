import torch
import torch.nn as nn
from models.layers import MLP, social_transformer, st_encoder


class LEDInitializer(nn.Module):
    def __init__(self, t_h: int = 11, d_h: int = 3, t_f: int = 12, d_f: int = 3, k_pred: int = 20):
        '''
        Parameters
        ----
        t_h: history timestamps,
        d_h: dimension of each historical timestamp,
        t_f: future timestamps,
        d_f: dimension of each future timestamp,
        k_pred: number of predictions.
        '''
        super(LEDInitializer, self).__init__()
        self.n = k_pred
        self.input_dim = t_h * d_h
        self.output_dim = t_f * d_f * k_pred
        self.fut_len = t_f

        # ============================================================
        # [新增/修改] 航线先验编码器 (Prior Encoder)
        # 接收 TrajAirNet 编码后的 Map Feature (64维)
        # ============================================================
        self.prior_input_dim = 64
        self.prior_feature_dim = 32
        self.prior_encoder = MLP(self.prior_input_dim, self.prior_feature_dim, hid_feat=(64, 64), activation=nn.ReLU())

        # 建模智能体之间的交互信息
        self.social_encoder = social_transformer(t_h)
        self.ego_var_encoder = st_encoder()
        self.ego_mean_encoder = st_encoder()
        self.ego_scale_encoder = st_encoder()

        self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

        # [修改] 解码器输入维度: Ego(256) + Social(256) + Prior(32) = 544
        self.var_decoder = MLP(256 * 2 + 32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
        self.mean_decoder = MLP(256 * 2 + self.prior_feature_dim, t_f * d_f, hid_feat=(256, 128), activation=nn.ReLU())
        self.scale_decoder = MLP(256 * 2, 1, hid_feat=(256, 128), activation=nn.ReLU())

    # [关键] 必须接收 map_features
    def forward(self, x, mask=None, map_features=None):
        '''
        x: 历史轨迹, 形状 [Batch*Agent, T, 3]
        map_features: 航线特征, 形状 [Batch*Agent, 64] (来自 TrajAirNet 的 Map Encoder 输出)
        '''
        var_num = 3
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        # 1. 提取基础特征
        social_embed = self.social_encoder(x, mask).squeeze(1)  # [B*A, 256]
        ego_var_embed = self.ego_var_encoder(x)
        ego_mean_embed = self.ego_mean_encoder(x)  # [B*A, 256]
        ego_scale_embed = self.ego_scale_encoder(x)

        # 2. [处理] 航线先验特征 (编码 Map Feature)
        if map_features is not None:
            # 编码特征 (64维 -> 32维)
            priors_embed = self.prior_encoder(map_features)  # [B*A, 32]
        else:
            priors_embed = torch.zeros(x.size(0), self.prior_feature_dim).to(x.device)

        # 3. [融合] 均值预测 (Ego + Social + Prior)
        mean_total = torch.cat((ego_mean_embed, social_embed, priors_embed), dim=-1)
        guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, var_num)

        # 4. 方差分支 (保持原逻辑)
        scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
        guess_scale = self.scale_decoder(scale_total)

        guess_scale_feat = self.scale_encoder(guess_scale)
        var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
        guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, var_num)

        return guess_var, guess_mean, guess_scale
