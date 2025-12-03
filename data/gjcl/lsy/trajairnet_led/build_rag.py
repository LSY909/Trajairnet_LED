import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# 确保文件名和类名正确
from model.Rag_embedder import TimeSeriesEmbedder
from model.Rag import TimeSeriesRAG, TimeSeriesDocument
from model.utils import TrajectoryDataset


def build_rag_database():
    print("=== Running Fixed Script v3 (Auto-Transpose & Dynamic Dim) ===")

    parser = argparse.ArgumentParser(description="构建 RAG 向量数据库")
    parser.add_argument('--dataset_name', type=str, default='7days1', help='数据集名称')
    parser.add_argument('--set_type', type=str, default='train', help='构建库通常只用 train 集')
    parser.add_argument('--obs_len', type=int, default=11, help='历史观测长度')
    parser.add_argument('--pred_len', type=int, default=120, help='未来预测长度')
    parser.add_argument('--save_dir_name', type=str, default='7days1_rag', help='保存 RAG 数据库的文件夹名')

    args = parser.parse_args()

    # 1. 路径配置
    root = os.getcwd()
    data_path = os.path.join(root, 'dataset', args.dataset_name, 'processed_data', args.set_type)
    save_path = os.path.join(root, 'dataset', args.save_dir_name)

    print(f"数据源: {data_path}")
    print(f"保存目标: {save_path}")

    # 2. 初始化 Embedder
    embedder = TimeSeriesEmbedder(target_length=256)

    # 注意：这里我们先不初始化 RAG，等读到第一条数据确认维度后再初始化
    rag = None

    # 3. 加载数据集
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据路径 {data_path}")
        return

    dataset = TrajectoryDataset(data_path, obs_len=args.obs_len, pred_len=args.pred_len, step=10)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"正在处理 {len(dataset)} 个场景...")

    # 4. 遍历数据并存入 RAG
    total_docs = 0
    skipped_docs = 0

    for batch_idx, batch in enumerate(tqdm(loader)):
        try:
            # batch[0] shape: (1, Agents, Ch, Time) OR (1, Agents, Time, Ch)
            # 我们先取 [0] 去掉 Batch 维
            scene_obs = batch[0][0].numpy()  # e.g., (1, 3, 11)
            scene_future = batch[1][0].numpy()

            num_agents = scene_obs.shape[0]

            for agent_i in range(num_agents):
                # 提取单条轨迹
                agent_obs_traj = scene_obs[agent_i]  # e.g., (3, 11)
                agent_fut_traj = scene_future[agent_i]

                # === 关键修正：自动转置 ===
                # 如果形状是 (3, 11) 且 obs_len=11，说明时间在最后，需要转置为 (11, 3)
                if agent_obs_traj.shape[-1] == args.obs_len and agent_obs_traj.shape[0] < args.obs_len:
                    agent_obs_traj = agent_obs_traj.T
                    # 如果 future 也是同样的格式，也顺手转一下，方便存储
                    if agent_fut_traj.shape[-1] == args.pred_len:
                        agent_fut_traj = agent_fut_traj.T

                # 现在的 agent_obs_traj 应该是 (11, 3) 或者 (11, 2)

                # === 延迟初始化 RAG (只执行一次) ===
                if rag is None:
                    # 用第一条真实数据计算维度
                    first_emb = embedder.embed(agent_obs_traj)
                    real_dim = first_emb.shape[1]
                    print(f"✅ 成功检测真实数据维度。")
                    print(f"   输入数据形状: {agent_obs_traj.shape}")
                    print(f"   Embedding 维度: {real_dim}")
                    rag = TimeSeriesRAG(embedding_dim=real_dim)

                # 生成 Embedding
                embedding = embedder.embed(agent_obs_traj)

                # 创建文档
                doc = TimeSeriesDocument(
                    id=f"scene_{batch_idx}_agent_{agent_i}",
                    data=agent_obs_traj,
                    pred_data=agent_fut_traj,
                    metadata={
                        "dataset": args.dataset_name,
                        "scene_index": batch_idx,
                        "agent_index": agent_i
                    },
                    embedding=embedding
                )

                rag.add_document(doc)
                total_docs += 1

        except Exception as e:
            if skipped_docs == 0:
                print(f"\nError details (first occurrence): {e}")
                print(f"Data shape that failed: {scene_obs.shape}")
            skipped_docs += 1
            continue

    # 5. 保存数据库
    if rag is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        rag.save(os.path.join(save_path, "rag_index"))
        print(f"=== 构建完成 ===")
        print(f"成功存储: {total_docs} 条轨迹")
    else:
        print("错误：未能成功处理任何数据，未生成数据库。")

    print(f"文件已保存至: {save_path}")


if __name__ == "__main__":
    build_rag_database()