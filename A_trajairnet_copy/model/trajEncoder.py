import torch
import torch.nn as nn


class TrajEncoder(nn.Module):
    def __init__(self, in_dim=3, d_model=128, n_layers=2):
        super().__init__()
        self.input_fc = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, 4, dim_feedforward=4 * d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, traj):
        # traj: [T, C] -> [1, T, C] 临时增加 batch
        traj = traj.unsqueeze(0)
        x = self.input_fc(traj)  # [1, T, D]
        x = self.encoder(x)  # [1, T, D]
        return x.mean(dim=1).squeeze(0)  # [D]


class PriorFusion(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.hist_encoder = TrajEncoder(d_model=d_model)
        self.cand_encoder = TrajEncoder(d_model=d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

    def forward(self, hist_traj, cand_trajs):
        """
        hist_traj: [T_obs, C]
        cand_trajs: [K, T_pred, C]
        """
        K, T_pred, C = cand_trajs.shape
        hist_feat = self.hist_encoder(hist_traj)  # [D]
        cand_feats = self.cand_encoder(cand_trajs.view(K, T_pred, C))  # [K, D]

        # Cross-Attention: query=hist_feat, key/value=cand_feats
        hist_feat_batch = hist_feat.unsqueeze(0).unsqueeze(0)  # [1,1,D]
        cand_feats_batch = cand_feats.unsqueeze(0)  # [1,K,D]
        fused_feat, _ = self.cross_attn(hist_feat_batch, cand_feats_batch, cand_feats_batch)
        return fused_feat.squeeze(0).squeeze(0)  # [D]
