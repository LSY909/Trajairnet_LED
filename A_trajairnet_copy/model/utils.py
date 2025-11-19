import math
import os
from tqdm import tqdm

from torch import nn
import torch
from torch.utils.data import Dataset

import numpy as np
from scipy.spatial import distance_matrix
import random

from model.Rag import TimeSeriesRAG, TimeSeriesDocument
import uuid
from model.Rag_embedder import TimeSeriesEmbedder

##Dataloader class
## 飞机轨迹数据集类
class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets
    Modified from https://github.com/alexmonti19/dagnet"""
    
    def __init__(
        self, data_dir, obs_len=11, pred_len=120, skip=8,step=10,
        min_agent=0, delim=' '):
        """
        Args:
        数据集文件所在目录
        - data_dir: Directory containing dataset files in the format
        帧  智能体编号  x  y
        <frame_id> <agent_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        一个序列中至少包含的智能体数量
        - min_agent: Minimum number of agents that should be in a seqeunce
        - step: Subsampling for pred
        数据集文件中的分隔符
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_agents_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.step = step
        ## 完整轨迹序列的总长度
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        ## 下采样后完整轨迹序列的总长度 math.ceil向上取整
        self.seq_final_len = self.obs_len + int(math.ceil(self.pred_len/self.step))
        ## 获取数据目录下的所有文件名
        all_files = os.listdir(self.data_dir)
        ## 拼接完整文件路径
        ## os.path.join(a, b)：拼接路径的函数
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_agents_in_seq = []
        ## 绝对坐标轨迹序列  相对坐标序列
        seq_list = []
        seq_list_rel = []
        context_list = []
        num = 0
        ## tqdm:进度条工具
        for path in tqdm(all_files):
            data = read_file(path, delim)
            ## data[:, 0]: 提取所有行的第一列（帧号）
            if (len(data)==0 or len(data[:,0])==0):
                print("File is empty")
                continue
            ## 提取轨迹序列中的帧号本身，frames是一个列表，包含所有帧号
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                ## 按帧分组：把同一帧的所有飞机数据放一起
                frame_data.append(data[frame == data[:, 0], :])
            ## 预测的轨迹序列的个数，（总帧数-一段轨迹长度+1）/往后滑几帧
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
            ## 按skip间隔取轨迹片段
            for idx in range(0, num_sequences * self.skip + 1, skip):
                ## 拼接数组
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                ## 提取当前序列中的智能体编号
                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])

                self.max_agents_in_frame = max(self.max_agents_in_frame,len(agents_in_curr_seq))
                
                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 3,
                                         self.seq_final_len))
                curr_seq = np.zeros((len(agents_in_curr_seq), 3,self.seq_final_len ))
                curr_context =  np.zeros((len(agents_in_curr_seq), 2,self.seq_final_len ))
                ## 有效的智能体数量
                num_agents_considered = 0
                for _, agent_id in enumerate(agents_in_curr_seq):
                    curr_agent_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 agent_id, :]
                    ## 计算当前智能体轨迹在序列中的起始索引
                    pad_front = frames.index(curr_agent_seq[0, 0]) - idx
                    pad_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1
                    ## 过滤不完整的轨迹序列
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:])
                    obs = curr_agent_seq[:,:obs_len]
                    pred = curr_agent_seq[:,obs_len+step-1::step]
                    curr_agent_seq = np.hstack((obs,pred))
                    context = curr_agent_seq[-2:,:]
                    assert(~np.isnan(context).any())
                    
                    # Make coordinates relative
                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    rel_curr_agent_seq[:, 1:] = \
                        curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]

                    _idx = num_agents_considered

                    if (curr_agent_seq.shape[1]!=self.seq_final_len):
                        continue

                   
                    curr_seq[_idx, :, pad_front:pad_end] = curr_agent_seq[:3,:]
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_agent_seq[:3,:]
                    curr_context[_idx,:,pad_front:pad_end] = context
                    num_agents_considered += 1

                if num_agents_considered > min_agent:
                    num_agents_in_seq.append(num_agents_considered)
                    seq_list.append(curr_seq[:num_agents_considered])
                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])
                    context_list.append(curr_context[:num_agents_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        context_list = np.concatenate(context_list, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_context = torch.from_numpy(
            context_list[:,:,:self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_agents_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.max_agents = -float('Inf')
        for (start, end) in self.seq_start_end:
            n_agents = end - start
            self.max_agents = n_agents if n_agents > self.max_agents else self.max_agents

    def __len__(self):
        return self.num_seq
    
    def __max_agents__(self):
        return self.max_agents

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :], self.obs_context[start:end, :]
        ]
        return out
### 辅助类（字典的扩展类，支持点号访问属性）
class DotDict(dict):
    r"""dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __getstate__ = dict
    __setstate__ = dict.update


### 带检索增强的轨迹数据集类
class TrajectoryDataset_RAG(Dataset):
    """Dataloder for the Trajectory datasets
    Modified from https://github.com/alexmonti19/dagnet"""

    def __init__(
            self, data_dir, obs_len=11, pred_len=120, skip=8, step=10,
            min_agent=0, delim=' '):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <agent_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - min_agent: Minimum number of agents that should be in a seqeunce
        - step: Subsampling for pred
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset_RAG, self).__init__()

        self.max_agents_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.step = step
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.seq_final_len = self.obs_len + int(math.ceil(self.pred_len / self.step))

        self.rag_system = TimeSeriesRAG()

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_agents_in_seq = []
        seq_list = []
        seq_list_rel = []
        context_list = []
        for path in tqdm(all_files):
            # print(path)
            data = read_file(path, delim)
            if (len(data[:, 0]) == 0):
                print("File is empty")
                continue
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)

                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_agents_in_frame = max(self.max_agents_in_frame, len(agents_in_curr_seq))

                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 3,
                                         self.seq_final_len))
                curr_seq = np.zeros((len(agents_in_curr_seq), 3, self.seq_final_len))
                curr_context = np.zeros((len(agents_in_curr_seq), 2, self.seq_final_len))
                num_agents_considered = 0
                for _, agent_id in enumerate(agents_in_curr_seq):
                    curr_agent_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                   agent_id, :]
                    pad_front = frames.index(curr_agent_seq[0, 0]) - idx
                    pad_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:])
                    obs = curr_agent_seq[:, :obs_len]
                    pred = curr_agent_seq[:, obs_len + step - 1::step]
                    curr_agent_seq = np.hstack((obs, pred))
                    context = curr_agent_seq[-2:, :]
                    assert (~np.isnan(context).any())

                    # Make coordinates relative
                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    rel_curr_agent_seq[:, 1:] = \
                        curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]

                    _idx = num_agents_considered

                    if (curr_agent_seq.shape[1] != self.seq_final_len):
                        continue

                    curr_seq[_idx, :, pad_front:pad_end] = curr_agent_seq[:3, :]
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_agent_seq[:3, :]
                    curr_context[_idx, :, pad_front:pad_end] = context
                    num_agents_considered += 1

                if num_agents_considered > min_agent:
                    num_agents_in_seq.append(num_agents_considered)
                    seq_list.append(curr_seq[:num_agents_considered])
                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])
                    context_list.append(curr_context[:num_agents_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        context_list = np.concatenate(context_list, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_context = torch.from_numpy(
            context_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_agents_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.max_agents = -float('Inf')
        for (start, end) in self.seq_start_end:
            n_agents = end - start
            self.max_agents = n_agents if n_agents > self.max_agents else self.max_agents

        obs_traj = seq_list[:, :, :self.obs_len]
        pred_traj = seq_list[:, :, self.obs_len:]
        embedder = TimeSeriesEmbedder()
        for i in range(seq_list.shape[0]):
            obs_sub_traj = obs_traj[i]
            pred_sub_traj = pred_traj[i]
            metadata_dict = {}
            doc_id = str(uuid.uuid4())
            # 有一个问题，如果效果不好的话，就是检索的过程你是拿输入的和完整的进行对比，是否不太合适？
            obs_sub_traj = np.transpose(obs_sub_traj, (1,0))
            embedding = embedder.embed(obs_sub_traj)
            doc = TimeSeriesDocument(
                id=doc_id,
                data=obs_sub_traj,
                pred_data = np.transpose(pred_sub_traj, (1,0)),
                metadata=metadata_dict,
                embedding=embedding
            )
            self.rag_system.add_document(doc)



### Metrics  评估指标
## ADE：平均距离误差
def ade(y1,y2):
    """
    y: (seq_len,2)
    """
    y1 = np.transpose(y1,(1,0))
    y2 = np.transpose(y2,(1,0))
    loss = y1 -y2
    loss = loss**2
    loss = np.sqrt(np.sum(loss,1))

    return np.mean(loss)
## FDE：最后一个时间步的误差
def fde(y1,y2):

    loss = (y1[:,-1] - y2[:,-1])**2
    return np.sqrt(np.sum(loss))

def rel_to_abs(obs,rel_pred):
    ## obs:已知轨迹的最后一个位置
    ## rel_pred:预测的相对移动
    pred = rel_pred.copy()
    pred[0] += obs[-1]
    for i in range(1,len(pred)):
        pred[i] += pred[i-1]
    ## 得到预测的最终位置
    return pred 
## RMSE：均方根误差
def rmse(y1,y2):
    criterion = nn.MSELoss()

    # return loss
    return torch.sqrt(criterion(y1, y2))

## General utils
## 读文件
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    ## 打开文件
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
        ## 将每一行数据添加到列表中
            data.append(line)
    ## 将列表转换为NumPy数组
    return np.asarray(data)

## permute:调整维度顺序
## acc:飞机的加速度数据
## 已知飞机的位置
def acc_to_abs(acc,obs,delta=1):
    ## 调整acc的维度顺序
    acc = acc.permute(2,1,0)
    ## 准备空容器pred用来存放像acc一样的预测位置
    pred = torch.empty_like(acc)
    ## 两个预测帧
    pred[0] = 2*obs[-1] - obs[0] + acc[0]
    pred[1] = 2*pred[0] - obs[-1] + acc[1]
    ## 预测帧的位置
    for i in range(2,acc.shape[0]):
        pred[i] = 2*pred[i-1] - pred[i-2] + acc[i]
    ## 把维度顺序调回和输入的acc一样的维度
    return pred.permute(2,1,0)
    

# batch为什么是6个，在这里做的
## 数据批次化函数(把不同场景的轨迹数据打包成一个batch)
def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,context_list) = zip(*data)

    # 各个场景的目标长度
    _len = [len(seq) for seq in obs_seq_list]
    # 计算纵轴的累加和
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    context = torch.cat(context_list, dim=0 ).permute(2,0,1)
    seq_start_end = torch.LongTensor(seq_start_end)


    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start_end
    ]
    return tuple(out)

# # batch为什么是6个，在这里做的
# def seq_collate_with_padding(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,context_list) = zip(*data)
    padding_num = 7
    # 便利每个场景，按照便利重复的方式padding



    # 检索
    '''
    1、遍历每个观测数据和预测数据
    2、记录每个场景的数据长度
    3、如果场景的数据长度小于padding_num就开始重复padding
    '''

    new_obs_seq_list = []
    new_pred_seq_list = []
    new_obs_seq_rel_list = []
    new_pred_seq_rel_list = []
    new_context_list = []


    # 在这里就要检索，然后检索的结果需要合理的拼在一起
    for i in range(len(data)):
        agent_num = obs_seq_list[i].shape[0]

        if agent_num >= 7:
            print(f"最大的智能体数量为{agent_num}")
            # obs_seq_list[i] = obs_seq_list[i][:padding_num,:,:]
            # obs_seq_list[i] = torch.stack(obs_seq_list[i], dim=0)[:padding_num, :, :]
            obs_seq_list[i] = obs_seq_list[i][:padding_num, :, :]
            continue
        index_list = list(range(agent_num))
        index_list = index_list * 7
        need_padding = padding_num - agent_num
        index_list = index_list[:need_padding]
        obs_traj_padding_aircraft_list = []
        pred_traj_padding_aircraft_list = []
        obs_traj_rel_padding_aircraft_list = []
        pred_traj_rel_padding_aircraft_list = []
        context_padding_aircraft_list = []
        for index in index_list:
            obs_traj_padding_aircraft = obs_seq_list[i][index].detach().clone()
            pred_traj_padding_aircraft = pred_seq_list[i][index].detach().clone()
            obs_traj_rel_padding_aircraft = obs_seq_rel_list[i][index].detach().clone()
            pred_traj_rel_padding_aircraft = pred_seq_rel_list[i][index].detach().clone()
            context_padding_aircraft = context_list[i][index].detach().clone()

            obs_traj_padding_aircraft_list.append(obs_traj_padding_aircraft.unsqueeze(dim=0))
            pred_traj_padding_aircraft_list.append(pred_traj_padding_aircraft.unsqueeze(dim=0))
            obs_traj_rel_padding_aircraft_list.append(obs_traj_rel_padding_aircraft.unsqueeze(dim=0))
            pred_traj_rel_padding_aircraft_list.append(pred_traj_rel_padding_aircraft.unsqueeze(dim=0))
            context_padding_aircraft_list.append(context_padding_aircraft.unsqueeze(dim=0))

        _obs_traj_padding_aircraft = torch.cat(obs_traj_padding_aircraft_list, dim=0)
        _pred_traj_padding_aircraft = torch.cat(pred_traj_padding_aircraft_list, dim=0)
        _obs_traj_rel_padding_aircraft = torch.cat(obs_traj_rel_padding_aircraft_list, dim=0)
        _pred_traj_rel_padding_aircraft = torch.cat(pred_traj_rel_padding_aircraft_list, dim=0)
        _context_padding_aircraft = torch.cat(context_padding_aircraft_list, dim=0)


        new_obs_seq_list.append(torch.cat([obs_seq_list[i],_obs_traj_padding_aircraft], dim=0))
        new_pred_seq_list.append(torch.cat([pred_seq_list[i],_pred_traj_padding_aircraft], dim=0))
        new_obs_seq_rel_list.append(torch.cat([obs_seq_rel_list[i],_obs_traj_rel_padding_aircraft], dim=0))
        new_pred_seq_rel_list.append(torch.cat([pred_seq_rel_list[i],_pred_traj_rel_padding_aircraft], dim=0))
        new_context_list.append(torch.cat([context_list[i],_context_padding_aircraft], dim=0))


    # 这个参数目前没用
    seq_start_end = torch.tensor([1, 2, 3, 4, 5])

    obs_traj = torch.stack(new_obs_seq_list, dim=0)
    pred_traj = torch.stack(new_pred_seq_list, dim=0)
    obs_traj_rel = torch.stack(new_obs_seq_rel_list, dim=0)
    pred_traj_rel = torch.stack(new_pred_seq_rel_list, dim=0)
    context = torch.stack(new_context_list, dim=0)

    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start_end
    ]
    return tuple(out)

## 打包数据批次（带有扩展）
def seq_collate_with_padding(data):
    """
    Collate function for DataLoader with sequence padding for multiple agents.
    Ensures each scene has exactly `padding_num` agents by truncating or repeating agents.
    Avoids modifying the original tuples.

    Args:
        data: list of dataset items, each item is a tuple:
              (obs_seq, pred_seq, obs_seq_rel, pred_seq_rel, context)

    Returns:
        Tuple of stacked tensors:
        (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start_end)
    """
    padding_num = 7  # 最大智能体数量

    # 解包 batch 数据
    obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, context_list = zip(*data)

    # 新列表保存处理后的 batch
    new_obs_seq_list = []
    new_pred_seq_list = []
    new_obs_seq_rel_list = []
    new_pred_seq_rel_list = []
    new_context_list = []

    for i in range(len(data)):
        agent_num = obs_seq_list[i].shape[0]

        if agent_num >= padding_num:
            # 超过 padding_num，直接截断
            print(f"最大的智能体数量为{agent_num}")
            new_obs_seq_list.append(obs_seq_list[i][:padding_num, :, :])
            new_pred_seq_list.append(pred_seq_list[i][:padding_num, :, :])
            new_obs_seq_rel_list.append(obs_seq_rel_list[i][:padding_num, :, :])
            new_pred_seq_rel_list.append(pred_seq_rel_list[i][:padding_num, :, :])
            new_context_list.append(context_list[i][:padding_num, :, :])
            continue

        # agent_num < padding_num，需要重复 padding
        index_list = list(range(agent_num)) * padding_num
        need_padding = padding_num - agent_num
        index_list = index_list[:need_padding]

        # 生成重复 padding 的 Tensor
        obs_padding_list = [obs_seq_list[i][idx].detach().clone().unsqueeze(0) for idx in index_list]
        pred_padding_list = [pred_seq_list[i][idx].detach().clone().unsqueeze(0) for idx in index_list]
        obs_rel_padding_list = [obs_seq_rel_list[i][idx].detach().clone().unsqueeze(0) for idx in index_list]
        pred_rel_padding_list = [pred_seq_rel_list[i][idx].detach().clone().unsqueeze(0) for idx in index_list]
        context_padding_list = [context_list[i][idx].detach().clone().unsqueeze(0) for idx in index_list]

        # 拼接原始数据和 padding 数据
        new_obs_seq_list.append(torch.cat([obs_seq_list[i], torch.cat(obs_padding_list, dim=0)], dim=0))
        new_pred_seq_list.append(torch.cat([pred_seq_list[i], torch.cat(pred_padding_list, dim=0)], dim=0))
        new_obs_seq_rel_list.append(torch.cat([obs_seq_rel_list[i], torch.cat(obs_rel_padding_list, dim=0)], dim=0))
        new_pred_seq_rel_list.append(torch.cat([pred_seq_rel_list[i], torch.cat(pred_rel_padding_list, dim=0)], dim=0))
        new_context_list.append(torch.cat([context_list[i], torch.cat(context_padding_list, dim=0)], dim=0))

    # 生成 seq_start_end（可根据需要修改）
    seq_start_end = torch.tensor([1, 2, 3, 4, 5])

    # stack 所有 batch(叠数据）
    ##在第0个维度前面添加一个维度，用来表示batch_size
    obs_traj = torch.stack(new_obs_seq_list, dim=0)
    pred_traj = torch.stack(new_pred_seq_list, dim=0)
    obs_traj_rel = torch.stack(new_obs_seq_rel_list, dim=0)
    pred_traj_rel = torch.stack(new_pred_seq_rel_list, dim=0)
    context = torch.stack(new_context_list, dim=0)
    ## 观察轨迹、预测轨迹、观察轨迹的相对坐标（前十一个点的相对移动距离）、预测轨迹的相对坐标、上下文、序列开始结束索引（属于哪个场景）
    return obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start_end

## 损失函数（包括轨迹重建损失和KLD损失）
### recon_y: 模型预测的轨迹 y: 真实轨迹 mean: 均值 log_var: 对数方差
def loss_func(recon_y,y,mean,log_var):
    traj_loss = rmse(recon_y,y)
    ## 计算KLD损失（KL散度）瞎猜
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return traj_loss + KLD

# 必须得是整体的预测结果，而如果只用最准确目标，损失好像不会降低
## 计算所有场景的总误差（min_loss取名不好）
def loss_func_MSE(recon_y,y):
    min_loss = 0
    for i in range(recon_y.shape[0]):
        traj_loss = rmse(recon_y[i],y.squeeze())
        min_loss += traj_loss
    return min_loss

# 仿照Sigulartrajectory的写法
# def loss_func_MSE(recon_y,y):
#     recon_y = recon_y.permute(0,2,1)
#     y = y.permute(0,2,1)
#     error_displacement = (recon_y - y.unsqueeze(dim=0)).norm(p=2, dim=-1)
#     min_loss = error_displacement.mean(dim=-1).min(dim=0)[0].mean()
#     # (pred_traj_recon - pred_traj.unsqueeze(dim=0)).norm(p=2, dim=-1)
#     # for i in range(recon_y.shape[0]):
#     #     traj_loss = rmse(recon_y[i],y.squeeze())
#     #     min_loss += traj_loss
#     return min_loss

# def loss_func_MSE(recon_y,y):
#     min_loss = float('inf')
#     for i in range(recon_y.shape[0]):
#         traj_loss = rmse(recon_y[i],y.squeeze())
#         if min_loss >= traj_loss:
#             min_loss = traj_loss
#     return min_loss

