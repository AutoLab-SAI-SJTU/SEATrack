import argparse
import random
import subprocess
import sys
import torch


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str,  default='seatrack', help='training script name')
    parser.add_argument('--config', type=str, help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, default='./output', help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, default=torch.cuda.device_count(), help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, help='training script name')
    parser.add_argument('--config_prv', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # whether to use wandb
    parser.add_argument('--seed', type=int, default=0, help='seed for model and data randomness')
    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')

    # for multiple machines
    parser.add_argument('--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--world-size', type=int, help='Number of processes participating in the job.')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
    parser.add_argument('--port', type=int, default='20000', help='Port of the current rank 0.')

    args = parser.parse_args()

    return args

'''" python -m torch.distributed.launch --nproc_per_node %d --master_port %d'''


def _append_optional(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def _append_common_train_args(cmd, args):
    cmd.extend([
        "--script", args.script,
        "--config", args.config,
        "--save_dir", args.save_dir,
        "--seed", str(args.seed),
        "--use_lmdb", str(args.use_lmdb),
        "--use_wandb", str(args.use_wandb),
        "--distill", str(args.distill),
    ])
    _append_optional(cmd, "--script_prv", args.script_prv)
    _append_optional(cmd, "--config_prv", args.config_prv)
    _append_optional(cmd, "--script_teacher", args.script_teacher)
    _append_optional(cmd, "--config_teacher", args.config_teacher)
    return cmd


def main():
    args = parse_args()
    if args.mode == "single":
        train_cmd = _append_common_train_args([sys.executable, "lib/train/run_training.py"], args)
    elif args.mode == "multiple":
        train_cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes", "1",
            "--nproc_per_node", str(args.nproc_per_node),
            "--master_port", str(random.randint(10000, 50000)),
            "lib/train/run_training.py",
        ]
        _append_common_train_args(train_cmd, args)
    elif args.mode == "multi_node":
        train_cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(args.nproc_per_node),
            "--master_addr", args.ip,
            "--master_port", str(args.port),
            "--nnodes", str(args.world_size),
            "--node_rank", str(args.rank),
            "lib/train/run_training.py",
        ]
        _append_common_train_args(train_cmd, args)
    else:
        raise ValueError("mode should be 'single' or 'multiple' or 'multi_node'.")
    subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()
