import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_mlps(c_in, mlp_channels, ret_before_act=False, without_norm=False):
    """
    输入一般是 (N, C) 的二维张量。
    这里的C是3
    """
    layers = []
    for k, c_out in enumerate(mlp_channels):
        is_last = (k + 1 == len(mlp_channels))
        if is_last and ret_before_act:
            layers.append(nn.Linear(c_in, c_out, bias=True))
        else:
            if without_norm:
                layers += [nn.Linear(c_in, c_out, bias=True), nn.ReLU()]
            else:
                layers += [nn.Linear(c_in, c_out, bias=False), nn.BatchNorm1d(c_out), nn.ReLU()]
            c_in = c_out
    return nn.Sequential(*layers)


class DenseFuturePredictor(nn.Module):
    """
    Dense Future Prediction 模块：
    - forward: 预测所有对象的稠密未来轨迹，并把“未来特征”融合回 obj_feature
    - loss: 计算 dense future prediction 的监督项（速度 L1 + GMM NLL）
    """
    def __init__(self, hidden_dim: int, num_future_frames: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_future_frames = num_future_frames

        # == build_dense_future_prediction_layers ==
        self.obj_pos_encoding_layer = build_mlps(
            c_in=3,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True,
            without_norm=True
        )
        self.dense_future_head = build_mlps(
            c_in=hidden_dim * 2,
            # output per-frame layout (diagonal 3D Gaussian + vel):
            # [x, y, z, logvar_x, logvar_y, logvar_z, vx, vy, vz] -> 3 + 3 + 3 = 9
            mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 9],
            ret_before_act=True
        )
        self.future_traj_mlps = build_mlps(
            # future input now: (x,y,z,vx,vy,vz) per frame -> 6 * T
            c_in=6 * num_future_frames,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True,
            without_norm=True
        )
        self.traj_fusion_mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True,
            without_norm=True
        )

    @torch.no_grad()
    def _assert_shapes(self, obj_feature, obj_mask, obj_pos):
        assert obj_feature.ndim == 3, "obj_feature should be (B, N, C)"
        assert obj_mask.ndim == 2, "obj_mask should be (B, N)"
        assert obj_pos.ndim == 3 and obj_pos.size(-1) >= 3, "obj_pos should be (B, N, >=3)"
        assert obj_feature.shape[0] == obj_mask.shape[0] == obj_pos.shape[0]
        assert obj_feature.shape[1] == obj_mask.shape[1] == obj_pos.shape[1]

    def forward(self, obj_feature, obj_mask, obj_pos):
        """
        Args:
            obj_feature: (B, N, C)
            obj_mask: (B, N) bool
            obj_pos: (B, N, 3) 
        Returns:
            obj_feature_enhanced: (B, N, C)  (融合 future feature 后)
            pred_dense_trajs: (B, N, T, 7)
        """
        self._assert_shapes(obj_feature, obj_mask, obj_pos)
        assert obj_mask.dtype == torch.bool, "obj_mask must be bool"

        B, N, C = obj_feature.shape
        T = self.num_future_frames
        device = obj_feature.device
        dtype = obj_feature.dtype

        # == apply_dense_future_prediction ==
        # 支持三维位置：(x,y,z)
        obj_pos_valid = obj_pos[obj_mask][..., 0:3]                # (Nv, 3)
        obj_feature_valid = obj_feature[obj_mask]                  # (Nv, C)

        obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)  # (Nv, hidden_dim)
        obj_fused_feature_valid = torch.cat([obj_pos_feature_valid, obj_feature_valid], dim=-1)  # (Nv, hidden_dim+C)

        pred_dense_trajs_valid = self.dense_future_head(obj_fused_feature_valid)  # (Nv, T*9)
        pred_dense_trajs_valid = pred_dense_trajs_valid.view(-1, T, 9)           # (Nv, T, 9)

        # 把相对位移转成绝对坐标（与 MTR 保持一致）
        # layout now: [dx,dy,dz, logvar_x,logvar_y,logvar_z, vx,vy,vz]
        temp_center_xyz = pred_dense_trajs_valid[:, :, 0:3] + obj_pos_valid[:, None, 0:3]
        pred_dense_trajs_valid = torch.cat([temp_center_xyz, pred_dense_trajs_valid[:, :, 3:]], dim=-1)

        # 用预测的 future (x,y,z,vx,vy,vz) 编码 future feature，并融合回 past obj_feature
        # pick (x,y,z, vx,vy,vz)
        obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, 2, -3, -2, -1]].flatten(1, 2)  # (Nv, T*6)
        obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)             # (Nv, hidden_dim)

        obj_full_trajs_feature = torch.cat([obj_feature_valid, obj_future_feature_valid], dim=-1)  # (Nv, C+hidden_dim)
        obj_feature_valid_new = self.traj_fusion_mlps(obj_full_trajs_feature)                       # (Nv, hidden_dim==C 通常)

        obj_feature_enhanced = torch.zeros((B, N, obj_feature_valid_new.shape[-1]), device=device, dtype=dtype)
        obj_feature_enhanced[obj_mask] = obj_feature_valid_new

        pred_dense_trajs = torch.zeros((B, N, T, 9), device=device, dtype=dtype)
        pred_dense_trajs[obj_mask] = pred_dense_trajs_valid
        return obj_feature_enhanced, pred_dense_trajs

    def loss(self, pred_dense_trajs, obj_trajs_future_state, obj_trajs_future_mask, nll_loss_gmm_direct=None):
        """
        get_dense_future_prediction_loss 的核心计算。

        Args:
            pred_dense_trajs: (B, N, T, 9) 来自 forward，布局为 [x,y,z, logvar_x,logvar_y,logvar_z, vx,vy,vz]
            obj_trajs_future_state: (B, N, T, 3) GT: [x,y,z]
            obj_trajs_future_mask: (B, N, T) 0/1 或 bool
            nll_loss_gmm_direct: legacy arg ignored (kept for signature compatibility)
        Returns:
            scalar loss
        """
        # pred now layout: [x,y,z, logvar_x,logvar_y,logvar_z, vx,vy,vz] -> 9
        assert pred_dense_trajs.shape[-1] == 9
        # GT expected as [x,y,z, vx,vy,vz] -> 6
        assert obj_trajs_future_state.shape[-1] == 6

        if obj_trajs_future_mask.dtype != torch.bool:
            obj_trajs_future_mask = obj_trajs_future_mask.bool()

        # 3D 对角高斯 NLL (使用 pred 的位置与 logvar)
        pred_mean_xyz = pred_dense_trajs[:, :, :, 0:3]     # (B,N,T,3)
        pred_logvar_xyz = pred_dense_trajs[:, :, :, 3:6]   # (B,N,T,3)
        gt_xyz = obj_trajs_future_state[:, :, :, 0:3]      # (B,N,T,3)

        var_exp = torch.exp(pred_logvar_xyz)
        nll_per_dim = 0.5 * (pred_logvar_xyz + ((gt_xyz - pred_mean_xyz) ** 2) / torch.clamp(var_exp, min=1e-12)) + 0.5 * np.log(2 * np.pi)
        nll = nll_per_dim.sum(dim=-1)  # sum over xyz -> (B,N,T)
        nll = (nll * obj_trajs_future_mask).sum(dim=-1)  # (B,N)

        loss_reg = nll
        obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0  # (B, N)
        loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / torch.clamp_min(obj_valid_mask.sum(dim=-1), 1.0)
        return loss_reg.mean()