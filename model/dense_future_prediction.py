import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlps(c_in, mlp_channels, ret_before_act=False, without_norm=False):
    """
    复刻 MTR 的 common_layers.build_mlps（含 BatchNorm1d）。
    输入一般是 (N, C) 的二维张量。
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
    从 MTRDecoder 抽出的 Dense Future Prediction 模块：
    - forward: 预测所有对象的稠密未来轨迹，并把“未来特征”融合回 obj_feature
    - loss: 计算 dense future prediction 的监督项（速度 L1 + GMM NLL）
    """
    def __init__(self, hidden_dim: int, num_future_frames: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_future_frames = num_future_frames

        # == build_dense_future_prediction_layers ==
        self.obj_pos_encoding_layer = build_mlps(
            c_in=2,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True,
            without_norm=True
        )
        self.dense_future_head = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 7],
            ret_before_act=True
        )
        self.future_traj_mlps = build_mlps(
            c_in=4 * num_future_frames,
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
        assert obj_pos.ndim == 3 and obj_pos.size(-1) >= 2, "obj_pos should be (B, N, >=2)"
        assert obj_feature.shape[0] == obj_mask.shape[0] == obj_pos.shape[0]
        assert obj_feature.shape[1] == obj_mask.shape[1] == obj_pos.shape[1]

    def forward(self, obj_feature, obj_mask, obj_pos):
        """
        Args:
            obj_feature: (B, N, C)
            obj_mask: (B, N) bool
            obj_pos: (B, N, 3) or (B, N, 2) 只用前两维
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
        obj_pos_valid = obj_pos[obj_mask][..., 0:2]                # (Nv, 2)
        obj_feature_valid = obj_feature[obj_mask]                  # (Nv, C)

        obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)  # (Nv, hidden_dim)
        obj_fused_feature_valid = torch.cat([obj_pos_feature_valid, obj_feature_valid], dim=-1)  # (Nv, hidden_dim+C)

        pred_dense_trajs_valid = self.dense_future_head(obj_fused_feature_valid)  # (Nv, T*7)
        pred_dense_trajs_valid = pred_dense_trajs_valid.view(-1, T, 7)           # (Nv, T, 7)

        # 把相对位移转成绝对坐标（与 MTR 保持一致）
        temp_center = pred_dense_trajs_valid[:, :, 0:2] + obj_pos_valid[:, None, 0:2]
        pred_dense_trajs_valid = torch.cat([temp_center, pred_dense_trajs_valid[:, :, 2:]], dim=-1)

        # 用预测的 future (x,y,vx,vy) 编码 future feature，并融合回 past obj_feature
        obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, -2, -1]].flatten(1, 2)  # (Nv, T*4)
        obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)             # (Nv, hidden_dim)

        obj_full_trajs_feature = torch.cat([obj_feature_valid, obj_future_feature_valid], dim=-1)  # (Nv, C+hidden_dim)
        obj_feature_valid_new = self.traj_fusion_mlps(obj_full_trajs_feature)                       # (Nv, hidden_dim==C 通常)

        obj_feature_enhanced = torch.zeros((B, N, obj_feature_valid_new.shape[-1]), device=device, dtype=dtype)
        obj_feature_enhanced[obj_mask] = obj_feature_valid_new

        pred_dense_trajs = torch.zeros((B, N, T, 7), device=device, dtype=dtype)
        pred_dense_trajs[obj_mask] = pred_dense_trajs_valid
        return obj_feature_enhanced, pred_dense_trajs

    def loss(self, pred_dense_trajs, obj_trajs_future_state, obj_trajs_future_mask, nll_loss_gmm_direct):
        """
        复刻 get_dense_future_prediction_loss 的核心计算。

        Args:
            pred_dense_trajs: (B, N, T, 7) 来自 forward
            obj_trajs_future_state: (B, N, T, 4) GT: [x,y,vx,vy]
            obj_trajs_future_mask: (B, N, T) 0/1 或 bool
            nll_loss_gmm_direct: 一个函数，签名需兼容：
                nll_loss_gmm_direct(pred_scores, pred_trajs, gt_trajs, gt_valid_mask,
                                    pre_nearest_mode_idxs, timestamp_loss_weight=None, use_square_gmm=False)
        Returns:
            scalar loss
        """
        assert pred_dense_trajs.shape[-1] == 7
        assert obj_trajs_future_state.shape[-1] == 4

        if obj_trajs_future_mask.dtype != torch.bool:
            obj_trajs_future_mask = obj_trajs_future_mask.bool()

        pred_dense_trajs_gmm = pred_dense_trajs[:, :, :, 0:5]
        pred_dense_trajs_vel = pred_dense_trajs[:, :, :, 5:7]

        # vel L1
        loss_reg_vel = F.l1_loss(pred_dense_trajs_vel, obj_trajs_future_state[:, :, :, 2:4], reduction="none")
        loss_reg_vel = (loss_reg_vel * obj_trajs_future_mask[:, :, :, None]).sum(dim=-1).sum(dim=-1)  # (B, N)

        B, N, T, _ = pred_dense_trajs.shape
        fake_scores = pred_dense_trajs.new_zeros((B * N, 1))  # 单模态占位
        temp_pred_trajs = pred_dense_trajs_gmm.contiguous().view(B * N, 1, T, 5)
        temp_gt_idx = torch.zeros(B * N, device=pred_dense_trajs.device, dtype=torch.long)
        temp_gt_trajs = obj_trajs_future_state[:, :, :, 0:2].contiguous().view(B * N, T, 2)
        temp_gt_mask = obj_trajs_future_mask.contiguous().view(B * N, T)

        loss_reg_gmm, _ = nll_loss_gmm_direct(
            pred_scores=fake_scores,
            pred_trajs=temp_pred_trajs,
            gt_trajs=temp_gt_trajs,
            gt_valid_mask=temp_gt_mask,
            pre_nearest_mode_idxs=temp_gt_idx,
            timestamp_loss_weight=None,
            use_square_gmm=False,
        )
        loss_reg_gmm = loss_reg_gmm.view(B, N)

        loss_reg = loss_reg_vel + loss_reg_gmm
        obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0  # (B, N)
        loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / torch.clamp_min(obj_valid_mask.sum(dim=-1), 1.0)
        return loss_reg.mean()