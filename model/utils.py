import math
import os
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from torch import nn
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial import distance_matrix
import random
from model.Rag import TimeSeriesRAG, TimeSeriesDocument
import uuid
from model.Rag_embedder import TimeSeriesEmbedder


# ==========================================
# 1. 核心 Dataset 类
# ==========================================
class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, data_dir, obs_len=11, pred_len=120, skip=8, step=10,
                 min_agent=0, delim=' ', priors_path=None):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.step = step
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.seq_final_len = self.obs_len + int(math.ceil(self.pred_len / self.step))

        # [关键] 文件排序，确保索引对齐
        all_files = sorted(os.listdir(self.data_dir))
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        num_agents_in_seq = []
        seq_list = []
        seq_list_rel = []
        context_list = []

        for path in tqdm(all_files, desc="Loading Raw Data"):
            data = read_file(path, delim)
            if (len(data) == 0 or len(data[:, 0]) == 0): continue
            frames = np.unique(data[:, 0]).tolist()
            frame_data = [data[frame == data[:, 0], :] for frame in frames]
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])

                self.max_agents_in_frame = max(getattr(self, 'max_agents_in_frame', 0), len(agents_in_curr_seq))

                curr_seq = np.zeros((len(agents_in_curr_seq), 3, self.seq_final_len))
                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 3, self.seq_final_len))
                curr_context = np.zeros((len(agents_in_curr_seq), 2, self.seq_final_len))

                num_agents_considered = 0
                for _, agent_id in enumerate(agents_in_curr_seq):
                    curr_agent_seq = curr_seq_data[curr_seq_data[:, 1] == agent_id, :]
                    pad_front = frames.index(curr_agent_seq[0, 0]) - idx
                    pad_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len: continue

                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:])
                    obs = curr_agent_seq[:, :obs_len]
                    pred = curr_agent_seq[:, obs_len + step - 1::step]
                    curr_agent_seq = np.hstack((obs, pred))
                    context = curr_agent_seq[-2:, :]

                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    rel_curr_agent_seq[:, 1:] = curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]

                    if (curr_agent_seq.shape[1] != self.seq_final_len): continue

                    curr_seq[num_agents_considered, :, pad_front:pad_end] = curr_agent_seq[:3, :]
                    curr_seq_rel[num_agents_considered, :, pad_front:pad_end] = rel_curr_agent_seq[:3, :]
                    curr_context[num_agents_considered, :, pad_front:pad_end] = context
                    num_agents_considered += 1

                if num_agents_considered > min_agent:
                    num_agents_in_seq.append(num_agents_considered)
                    seq_list.append(curr_seq[:num_agents_considered])
                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])
                    context_list.append(curr_context[:num_agents_considered])

        self.num_seq = len(seq_list)
        if self.num_seq > 0:
            seq_list = np.concatenate(seq_list, axis=0)
            seq_list_rel = np.concatenate(seq_list_rel, axis=0)
            context_list = np.concatenate(context_list, axis=0)

            self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).float()
            self.obs_context = torch.from_numpy(context_list[:, :, :self.obs_len]).float()
            self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).float()
            self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).float()
            self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).float()

            cum_start_idx = [0] + np.cumsum(num_agents_in_seq).tolist()
            self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
            self.max_agents = max(num_agents_in_seq) if num_agents_in_seq else 0
        else:
            self.max_agents = 0
            self.seq_start_end = []
            # 防止空数据报错，初始化空 tensor
            self.obs_traj = torch.empty(0)
            self.pred_traj = torch.empty(0)
            self.obs_traj_rel = torch.empty(0)
            self.pred_traj_rel = torch.empty(0)
            self.obs_context = torch.empty(0)

        # [修复] 正确的加载逻辑：检查传入的 priors_path
        self.priors_data = None
        if priors_path is not None and os.path.exists(priors_path):
            print(f"Loading precomputed priors from {priors_path}...")
            self.priors_data = torch.load(priors_path)

            # 简单校验长度
            if len(self.priors_data) != len(self.obs_traj):
                print(f"Warning: Priors length {len(self.priors_data)} != Obs length {len(self.obs_traj)}")
                # 强制对齐 (取较小值防止越界)
                min_len = min(len(self.priors_data), len(self.obs_traj))
                self.priors_data = self.priors_data[:min_len]

    def __len__(self):
        return self.num_seq

    def __max_agents__(self):
        return self.max_agents

    # [关键修复] 唯一的 __getitem__，包含 priors 逻辑
    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.obs_context[start:end, :]
        ]

        # 如果有 priors，追加到返回列表
        if self.priors_data is not None:
            out.append(self.priors_data[start:end])

        return out


# ==========================================
# 2. 辅助类 (DotDict, RAG Dataset)
# ==========================================
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __getstate__ = dict
    __setstate__ = dict.update


class TrajectoryDataset_RAG(Dataset):
    """用于构建 RAG 索引的简化 Dataset"""

    def __init__(self, data_dir, obs_len=11, pred_len=120, skip=8, step=10, min_agent=0, delim=' '):
        super(TrajectoryDataset_RAG, self).__init__()
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.rag_system = TimeSeriesRAG()

        all_files = sorted(os.listdir(self.data_dir))
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        seq_list = []
        for path in tqdm(all_files, desc="Building RAG"):
            data = read_file(path, delim)
            if (len(data) == 0): continue
            frames = np.unique(data[:, 0]).tolist()
            frame_data = [data[frame == data[:, 0], :] for frame in frames]
            num_sequences = int(math.ceil((len(frames) - (obs_len + pred_len) + 1) / skip))

            for idx in range(0, num_sequences * skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + obs_len + pred_len], axis=0)
                agents = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(agents), 3, obs_len + int(math.ceil(pred_len / step))))
                num_valid = 0
                for _, agent_id in enumerate(agents):
                    agent_seq = curr_seq_data[curr_seq_data[:, 1] == agent_id, :]
                    if len(agent_seq) != obs_len + pred_len: continue

                    traj = np.transpose(agent_seq[:, 2:])
                    obs = traj[:, :obs_len]
                    pred = traj[:, obs_len + step - 1::step]
                    full = np.hstack((obs, pred))
                    if full.shape[1] != curr_seq.shape[2]: continue

                    curr_seq[num_valid, :, :] = full[:3, :]
                    num_valid += 1

                if num_valid > min_agent:
                    seq_list.append(curr_seq[:num_valid])

        if len(seq_list) > 0:
            seq_list = np.concatenate(seq_list, axis=0)
            obs_traj = seq_list[:, :, :self.obs_len]
            pred_traj = seq_list[:, :, self.obs_len:]

            embedder = TimeSeriesEmbedder()
            # print("Adding documents to RAG...")
            for i in range(seq_list.shape[0]):
                obs = np.transpose(obs_traj[i], (1, 0))
                pred = np.transpose(pred_traj[i], (1, 0))
                embedding = embedder.embed(obs)
                doc = TimeSeriesDocument(
                    id=str(uuid.uuid4()),
                    data=obs,
                    pred_data=pred,
                    metadata={},
                    embedding=embedding
                )
                self.rag_system.add_document(doc)


# ==========================================
# 3. Collate 函数 (含 Padding 逻辑)
# ==========================================
def seq_collate_with_padding(data):
    """
    Collate function with padding.
    [Fix] Permute dimensions to (Batch, Agents, Time, Features)
    """
    # 检查是否包含 priors
    has_priors = (len(data[0]) > 5)

    if has_priors:
        obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, context_list, priors_list = zip(*data)
    else:
        obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, context_list = zip(*data)
        priors_list = None

    padding_num = 7

    new_obs, new_pred, new_obs_rel, new_pred_rel, new_ctx = [], [], [], [], []
    new_priors = []

    for i in range(len(data)):
        agent_num = obs_seq_list[i].shape[0]

        # 决定如何截断或重复
        if agent_num >= padding_num:
            idxs = list(range(padding_num))
        else:
            # 简单重复填充
            idxs = list(range(agent_num)) * padding_num
            idxs = idxs[:padding_num]

        # Padding Helper Function
        def pad_tensor(tensor, indices):
            # tensor: [N, 3, Time]
            if tensor.shape[0] >= padding_num:
                return tensor[:padding_num]
            # Gather by indices
            return torch.stack([tensor[idx] for idx in indices], dim=0)

        new_obs.append(pad_tensor(obs_seq_list[i], idxs))
        new_pred.append(pad_tensor(pred_seq_list[i], idxs))
        new_obs_rel.append(pad_tensor(obs_seq_rel_list[i], idxs))
        new_pred_rel.append(pad_tensor(pred_seq_rel_list[i], idxs))
        new_ctx.append(pad_tensor(context_list[i], idxs))

        if has_priors:
            new_priors.append(pad_tensor(priors_list[i], idxs))

    # Stack into Batch -> (Batch, Agents, 3, Time)
    obs_traj = torch.stack(new_obs, dim=0)
    pred_traj = torch.stack(new_pred, dim=0)
    obs_traj_rel = torch.stack(new_obs_rel, dim=0)
    pred_traj_rel = torch.stack(new_pred_rel, dim=0)
    # Context usually (Batch, Agents, 2, Time)
    context = torch.stack(new_ctx, dim=0)

    # ============================================================
    # [CRITICAL FIX] Permute dimensions
    # 当前: (Batch, Agents, Feat, Time)
    # 目标: (Batch, Agents, Time, Feat)
    # ============================================================
    obs_traj = obs_traj.permute(0, 1, 3, 2)
    pred_traj = pred_traj.permute(0, 1, 3, 2)
    obs_traj_rel = obs_traj_rel.permute(0, 1, 3, 2)
    pred_traj_rel = pred_traj_rel.permute(0, 1, 3, 2)
    context = context.permute(0, 1, 3, 2)

    # Dummy seq_start_end
    seq_start_end = torch.tensor([1] * len(data))

    results = [obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start_end]

    if has_priors:
        priors_traj = torch.stack(new_priors, dim=0)
        # Priors 通常是 (Batch, Agents, Clusters, Pred_Len, 4)
        # 这里需要检查 Priors 的原始维度，如果也是 Features 在前，也需要 permute
        # 假设 Priors 在 get_batch_route_priors 生成时已经是 (..., Pred_Len, 4) 则不需要动
        results.append(priors_traj)

    return tuple(results)


def seq_collate(data):
    """Old collate without padding"""
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, context_list) = zip(*data)
    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    context = torch.cat(context_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    out = [obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start_end]
    return tuple(out)


# ==========================================
# 4. 辅助函数与 Loss
# ==========================================
def ade(y1, y2):
    # y1, y2 shape: (Seq_Len, 2) or (Seq_Len, 2, K)
    loss = np.sqrt(np.sum((np.transpose(y1, (1, 0)) - np.transpose(y2, (1, 0))) ** 2, 1))
    return np.mean(loss)


def fde(y1, y2):
    return np.sqrt(np.sum((y1[:, -1] - y2[:, -1]) ** 2))


def rmse(y1, y2):
    return torch.sqrt(nn.MSELoss()(y1, y2))


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            try:
                data.append([float(i) for i in line])
            except:
                pass
    return np.asarray(data)


def acc_to_abs(acc, obs, delta=1):
    acc = acc.permute(2, 1, 0)
    pred = torch.empty_like(acc)
    pred[0] = 2 * obs[-1] - obs[0] + acc[0]
    pred[1] = 2 * pred[0] - obs[-1] + acc[1]
    for i in range(2, acc.shape[0]):
        pred[i] = 2 * pred[i - 1] - pred[i - 2] + acc[i]
    return pred.permute(2, 1, 0)


def loss_func(recon_y, y, mean, log_var):
    traj_loss = rmse(recon_y, y)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return traj_loss + KLD


def loss_func_MSE(recon_y, y):
    min_loss = 0
    for i in range(recon_y.shape[0]):
        traj_loss = rmse(recon_y[i], y.squeeze())
        min_loss += traj_loss
    return min_loss


# ==========================================
# 5. 在线计算航道先验 (用于 preprocess.py)
# ==========================================
def get_batch_route_priors(obs_traj, rag_system, embedder, k_retrieve=20, n_clusters=3, pred_len=12, device='cuda'):
    """
        利用 RAG 检索历史相似轨迹，并使用 GMM 聚类生成未来的先验路径（Priors）。

        输入:
            obs_traj:  (Batch, Agents, Obs_Len, 3)

        输出:
            priors: (Batch, Agents, n_clusters, pred_len, 4)
                      输出包含 x, y, z 和 width (不确定性范围)。
        """
    if obs_traj.shape[2] == 3 and obs_traj.shape[3] != 3:
        obs_traj = obs_traj.permute(0, 1, 3, 2)

    batch_size, num_agents, obs_len, dim = obs_traj.shape

    # 检索：(Batch * Agents, Obs_Len, 3)
    flat_obs = obs_traj.reshape(-1, obs_len, dim).cpu().numpy()
    query_emb = embedder.embed_batch(flat_obs).astype(np.float32)
    #Batch * Agents
    results_list = rag_system.search_batch(query_emb, k=k_retrieve)

    #逐个 Agent 处理
    batch_priors = []
    for i, res in enumerate(results_list):
        raw_trajs = []
        #提取检索到的“未来真值”
        for r in res:
            p_data = np.array(r['pred_data'])
            if p_data.shape[0] == 3: p_data = p_data.T
            raw_trajs.append(p_data)

        raw = np.array(raw_trajs)
        if raw.shape[0] == 0:
            batch_priors.append(np.zeros((n_clusters, pred_len, 4)))
            continue
        # 坐标归一化
        start_points = raw[:, 0:1, :]
        relative_raw = raw - start_points
        flat_relative = relative_raw.reshape(k_retrieve, -1)

        try:
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=42, max_iter=10,
                                  n_init=1)
            gmm.fit(flat_relative)

            # 聚类中心
            means = gmm.means_.reshape(n_clusters, pred_len, 3)
            # 计算标准差作为宽度
            covariances = gmm.covariances_.reshape(n_clusters, pred_len, 3)
            stds = np.sqrt(covariances)
            # 不确定性 widths,三个方向的标准差取平均
            widths = np.mean(stds, axis=2, keepdims=True)
            #拼接
            agent_prior = np.concatenate([means, widths], axis=2)
            batch_priors.append(agent_prior)
        except:
            batch_priors.append(np.zeros((n_clusters, pred_len, 4)))
    #(Batch * Agents, Clusters, Time, Feat) -> (Batch, Agents, Clusters, Time, Feat)
    priors_tensor = torch.tensor(np.array(batch_priors), dtype=torch.float32)
    priors_tensor = priors_tensor.view(batch_size, num_agents, n_clusters, pred_len, 4)

    return priors_tensor.to(device)