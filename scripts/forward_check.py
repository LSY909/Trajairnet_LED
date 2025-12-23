#!/usr/bin/env python3
"""
Run a single-batch forward pass to print tensor shapes and basic stats.
Usage:
  python scripts/forward_check.py --dataset_name 7days1_small --batch_size 1 --device cuda

This script requires a working PyTorch environment. It is safe to run on CPU by passing --device cpu.
"""
import os
import argparse
import traceback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='7days1_small')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dense_prior_weight', type=float, default=0.0)
    parser.add_argument('--route_priors_train', type=str, default='')
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from model.utils import TrajectoryDataset, seq_collate_with_padding
        from model.trajairnet import TrajAirNet
    except Exception:
        traceback.print_exc()
        print("Please run this script in an environment with PyTorch and the project installed.")
        return

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if device.type == 'cpu' else os.environ.get('CUDA_VISIBLE_DEVICES', '')

    datapath = os.getcwd() + args.dataset_folder + args.dataset_name + "/processed_data/"
    print("Loading dataset from", datapath)
    dataset = TrajectoryDataset(datapath + "train", route_priors_path=(args.route_priors_train if args.route_priors_train else None))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=seq_collate_with_padding)

    # Build args-like object to pass into TrajAirNet
    class SimpleArgs:
        pass

    s_args = SimpleArgs()
    # minimal required args used by TrajAirNet
    s_args.input_channels = 3
    s_args.preds = 120
    s_args.preds_step = 10
    s_args.tcn_channel_size = 256
    s_args.tcn_layers = 2
    s_args.tcn_kernels = 4
    s_args.gat_heads = 16
    s_args.graph_hidden = 256
    s_args.alpha = 0.2
    s_args.cvae_hidden = 128
    s_args.cvae_channel_size = 128
    s_args.cvae_layers = 2
    s_args.mlp_layer = 32
    s_args.obs = 11
    s_args.preds_step = 10
    s_args.k = 4
    s_args.num_samples = 15
    s_args.traj_dim = 3
    s_args.agent_num = 3
    s_args.cnn_kernels = 2
    s_args.num_context_output_c = 7
    s_args.dense_prior_weight = args.dense_prior_weight
    s_args.route_prior_mode = 'none' if args.route_priors_train == '' else 'per_batch'

    model = TrajAirNet(s_args)
    model.to(device)
    model.eval()

    with torch.no_grad():
        try:
            batch = next(iter(loader))
        except StopIteration:
            print("Dataset is empty or no samples found.")
            return

        # Print batch structure
        print("Batch length:", len(batch))
        names = ['obs_traj','pred_traj','obs_traj_rel','pred_traj_rel','context','seq_start_end']
        if len(batch) == 7:
            names.append('route_priors')
        for n, v in zip(names, batch):
            if isinstance(v, torch.Tensor):
                print(f"{n}: shape={v.shape}, dtype={v.dtype}, device={v.device}, mean={v.float().mean().item():.6f}, std={v.float().std().item():.6f}")
            else:
                print(f"{n}: type={type(v)} value={v}")

        # move to device
        batch = [ (b.to(device) if isinstance(b, torch.Tensor) else b) for b in batch ]

        # call model forward (will compute losses)
        try:
            if len(batch) == 7:
                obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start, route_priors = batch
            else:
                obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start = batch
                route_priors = None

            # model expects context transposed in train.py: torch.transpose(context,1,2)
            loss_dist, loss_uncertainty = model(
                obs_traj,
                pred_traj,
                torch.ones((obs_traj.shape[1], obs_traj.shape[1]), device=device)[0],
                torch.transpose(context, 1, 2),
                route_priors=route_priors,
                rag_system=None,
                embedder=None,
            )
            print("Forward succeeded. loss_dist:", float(loss_dist), "loss_uncertainty:", float(loss_uncertainty))
        except Exception:
            traceback.print_exc()
            print("Model forward raised an exception.")

if __name__ == "__main__":
    main()


