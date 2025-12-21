import torch
import torch.nn as nn
from models.layers import MLP, social_transformer, st_encoder

class LEDInitializer(nn.Module):
	def __init__(self, t_h: int=11, d_h: int=3, t_f: int=12, d_f: int=3, k_pred: int=20):
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


		## 建模智能体之间的交互信息
		self.social_encoder = social_transformer(t_h)
		## 建模目标自身历史特征 方差、均值、尺度预测结果
		self.ego_var_encoder = st_encoder()
		self.ego_mean_encoder = st_encoder()
		self.ego_scale_encoder = st_encoder()

		# ============================================================
		# Route priors (RAG/GMM) encoder: (B*A, C, T, 3) -> (B*A, prior_dim)
		# 使用“点级 MLP + pooling”方式，避免依赖 C 的固定值
		# ============================================================
		self.prior_dim = 32
		self.route_point_mlp = nn.Sequential(
			nn.Linear(d_f, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
		)
		self.route_agg = nn.Sequential(
			nn.Linear(64, self.prior_dim),
			nn.ReLU(),
		)

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		# 融入 route prior 后：
		# mean/scale 输入: Ego(256) + Social(256) + Prior(32) = 544
		# var 输入: EgoVar(256) + Social(256) + ScaleFeat(32) + Prior(32) = 576
		self.var_decoder = MLP(256*2+32+self.prior_dim, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(256*2+self.prior_dim, t_f * d_f, hid_feat=(256, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(256*2+self.prior_dim, 1, hid_feat=(256, 128), activation=nn.ReLU())


	def forward(self, x, mask=None,route_priors=None):
		'''
		x: batch size, t_p, 6
		route_priors: [Batch, Agent, 3, 12, 3]
		'''
		var_num  = 3
		## mask用来屏蔽无效邻居
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		social_embed = self.social_encoder(x, mask)
		social_embed = social_embed.squeeze(1)
		# B, 256

		ego_var_embed = self.ego_var_encoder(x)
		ego_mean_embed = self.ego_mean_encoder(x)
		ego_scale_embed = self.ego_scale_encoder(x)
		# B, 256

		# ====== 航线先验编码 ======
		# route_priors 期望形状: (B*A, C, T, 3) 或 (B, A, C, T, 3)
		if route_priors is not None:
			if route_priors.dim() == 5:
				# (B, A, C, T, 3) -> (B*A, C, T, 3)
				route_priors = route_priors.reshape(-1, route_priors.shape[-3], route_priors.shape[-2], route_priors.shape[-1])

			# 使用相对坐标（减去当前最后观测点），提高泛化
			last_pos = x[:, -1, :].unsqueeze(1).unsqueeze(1)  # (B*A, 1, 1, 3)
			route_rel = route_priors - last_pos
			points = route_rel.reshape(route_rel.shape[0], -1, route_rel.shape[-1])  # (B*A, C*T, 3)
			pt_feat = self.route_point_mlp(points)  # (B*A, C*T, 64)
			pooled = pt_feat.mean(dim=1)            # (B*A, 64)
			priors_embed = self.route_agg(pooled)   # (B*A, 32)
		else:
			priors_embed = torch.zeros(x.size(0), self.prior_dim, device=x.device, dtype=x.dtype)

		# ====== 均值分支 ======
		mean_total = torch.cat((ego_mean_embed, social_embed, priors_embed), dim=-1)
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, var_num)
		# ====== 方差分支 ======
		scale_total = torch.cat((ego_scale_embed, social_embed, priors_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total)

		guess_scale_feat = self.scale_encoder(guess_scale)
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat, priors_embed), dim=-1)
		guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, var_num)

		return guess_var, guess_mean, guess_scale

