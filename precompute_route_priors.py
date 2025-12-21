import argparse
import os
from typing import Optional

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from model.Rag_embedder import TimeSeriesEmbedder
from model.utils import TrajectoryDataset, TrajectoryDataset_RAG


def pad_indices(agent_num: int, padding_num: int) -> np.ndarray:
    """与 seq_collate_with_padding 的 padding 策略保持一致：不足就按原顺序重复补齐，超出就截断。"""
    if agent_num >= padding_num:
        return np.arange(padding_num, dtype=np.int64)
    index_list = list(range(agent_num)) * padding_num
    need_padding = padding_num - agent_num
    index_list = index_list[:need_padding]
    # 这里的 index_list 表示要补齐的 agent 索引
    return np.concatenate([np.arange(agent_num, dtype=np.int64), np.array(index_list, dtype=np.int64)], axis=0)


def normalize_pred_data(pred_data) -> np.ndarray:
    """确保 pred_data 形状为 (T, 3)"""
    arr = np.array(pred_data)
    if arr.ndim != 2:
        raise ValueError(f"pred_data ndim != 2: shape={arr.shape}")
    if arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T
    if arr.shape[1] != 3:
        raise ValueError(f"pred_data last dim != 3: shape={arr.shape}")
    return arr


def build_route_priors_for_scene(
    obs_traj_scene: torch.Tensor,  # (A, 3, Obs)
    rag_system,
    embedder: TimeSeriesEmbedder,
    k_retrieve: int,
    n_clusters: int,
    route_prior_mode: str,
    random_state: int = 0,
) -> np.ndarray:
    """
    返回: (A, C, T, 3) 的绝对航线先验
    """
    assert obs_traj_scene.ndim == 3 and obs_traj_scene.shape[1] == 3
    agent_num, _, obs_len = obs_traj_scene.shape

    # (A, Obs, 3)
    obs_np = obs_traj_scene.detach().cpu().numpy().transpose(0, 2, 1).astype(np.float32)
    embeddings = embedder.embed_batch(obs_np)
    search_res = rag_system.search_batch(embeddings, k=k_retrieve)

    # raw: (A, K, T, 3)
    raw_list = []
    for a in range(agent_num):
        arrs = [normalize_pred_data(item["pred_data"]) for item in search_res[a]]
        raw_list.append(np.stack(arrs, axis=0))
    raw = np.stack(raw_list, axis=0)

    t_f = raw.shape[2]
    rel = raw - raw[:, :, 0:1, :]  # (A, K, T, 3)

    if route_prior_mode == "topk":
        ctrs = rel[:, :n_clusters]  # (A, C, T, 3)
    elif route_prior_mode == "per_batch":
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            random_state=random_state,
        ).fit(rel.reshape(-1, t_f * 3))
        ctrs = np.broadcast_to(gmm.means_.reshape(1, n_clusters, t_f, 3), (agent_num, n_clusters, t_f, 3))
    else:
        # per_agent（默认/最精细）
        ctrs_list = []
        for a in range(agent_num):
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type="diag",
                random_state=random_state,
            ).fit(rel[a].reshape(k_retrieve, -1))
            ctrs_list.append(gmm.means_.reshape(n_clusters, t_f, 3))
        ctrs = np.stack(ctrs_list, axis=0)

    # 还原到当前观测最后一点 (A, 3)
    last_pos = obs_traj_scene.detach().cpu().numpy()[:, :, -1]  # (A, 3)
    abs_ctrs = ctrs + last_pos[:, None, None, :]
    return abs_ctrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/dataset/")
    parser.add_argument("--dataset_name", type=str, default="7days1")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])

    parser.add_argument("--obs", type=int, default=11)
    parser.add_argument("--preds", type=int, default=120)
    parser.add_argument("--preds_step", type=int, default=10)
    parser.add_argument("--delim", type=str, default=" ")

    parser.add_argument("--rag_dir", type=str, default="./dataset/rag_file_7days2")
    parser.add_argument("--k_retrieve", type=int, default=50)
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--route_prior_mode", type=str, default="per_agent",
                        choices=["per_agent", "per_batch", "topk"])
    parser.add_argument("--padding_num", type=int, default=7)

    parser.add_argument("--output", type=str, default="./dataset/route_priors_train.pt")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])

    args = parser.parse_args()

    root = os.getcwd()
    data_dir = os.path.join(root, args.dataset_folder.strip("/"), args.dataset_name, "processed_data", args.split)
    print(f"Loading dataset: {data_dir}")
    ds = TrajectoryDataset(data_dir, obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim)

    print(f"Building RAG from: {args.rag_dir}")
    rag = TrajectoryDataset_RAG(args.rag_dir, obs_len=args.obs, pred_len=args.preds, step=args.preds_step, delim=args.delim).rag_system
    embedder = TimeSeriesEmbedder()

    priors_all: Optional[torch.Tensor] = None
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    for idx in tqdm(range(len(ds)), desc=f"Precompute route priors ({args.split})"):
        obs_traj_scene = ds[idx][0]  # (A, 3, Obs)
        agent_num = obs_traj_scene.shape[0]
        keep_idx = pad_indices(agent_num, args.padding_num)
        obs_traj_padded = obs_traj_scene[keep_idx]

        priors_np = build_route_priors_for_scene(
            obs_traj_padded,
            rag_system=rag,
            embedder=embedder,
            k_retrieve=args.k_retrieve,
            n_clusters=args.n_clusters,
            route_prior_mode=args.route_prior_mode,
        )  # (padding_num, C, T, 3)

        priors_t = torch.from_numpy(priors_np).to(dtype=dtype)
        if priors_all is None:
            priors_all = torch.empty((len(ds),) + priors_t.shape, dtype=dtype)
        priors_all[idx] = priors_t

    assert priors_all is not None
    out_path = os.path.join(root, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "route_priors": priors_all,
            "meta": {
                "dataset_name": args.dataset_name,
                "split": args.split,
                "obs": args.obs,
                "preds": args.preds,
                "preds_step": args.preds_step,
                "k_retrieve": args.k_retrieve,
                "n_clusters": args.n_clusters,
                "route_prior_mode": args.route_prior_mode,
                "padding_num": args.padding_num,
                "dtype": args.dtype,
            },
        },
        out_path,
    )
    print(f"Saved route priors to: {out_path}")


if __name__ == "__main__":
    main()


