import argparse
from trainer import trainer_PPT as trainer_ppt
import logging
import random
import os
import torch
import numpy as np


def seed_torch(seed=1666):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_config():
    parser = argparse.ArgumentParser(description="MemoNet with SDD dataset")
    # 实验结果根路径
    parser.add_argument("--root_path", type=str, help="实验结果根路径")

    # 训练 or 测试
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--model_path', type=str, default='')

    # 是否使用图像
    parser.add_argument("--use_image", default=False)
    # 缩放系数
    parser.add_argument("--divide_coefficient", type=float, default=1)

    # 是否使用时间编码
    parser.add_argument("--use_temporal_enc", default=False)

    parser.add_argument("--cuda", default=True)
    # verify the CUDA_VISIBLE_DEVICES
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=717)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--learning_rate_min", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=600)

    parser.add_argument("--goal_num", type=int, default=20)

    # wandb
    parser.add_argument("--use_wandb", action="store_false")
    parser.add_argument("--wandb_project", type=str, default="1005idea")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--notes", type=str, default="", help="wandb实验详细介绍")

    parser.add_argument(
        "--past_len", type=int, default=8, help="length of past (in timesteps)"
    )
    parser.add_argument(
        "--future_len", type=int, default=12, help="length of future (in timesteps)"
    )
    parser.add_argument("--dim_embedding_key", type=int, default=128)

    # Configuration for SDD dataset.
    # 数据集根路径
    parser.add_argument("--dataset_path", type=str, default="/home/tyl/code/1005idea/dataset/")
    # 具体数据名
    parser.add_argument("--dataset_name", type=str, default="sdd")
    parser.add_argument("--dist_threshold", type=int, default=2)

    # 数据增强相关
    parser.add_argument("--use_augmentation", default=False)
    parser.add_argument("--smooth", default=False)
    parser.add_argument("--rotation", default=False)

    # 数据缩放
    parser.add_argument("--data_scaling", type=list, default=[1.9, 0.4])

    parser.add_argument('--num_workers', type=int, default=4)

    # Transformer Decoder Param
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--int_num_layers_list", type=int, default=[3, 1], nargs='+')
    parser.add_argument("--forward_expansion", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--embd_pdrop", type=float, default=0.1)
    parser.add_argument("--resid_pdrop", type=float, default=0.1)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--traj_embed_dropout", type=float, default=0, help="对轨迹编码后的特征进行dropout")
    # Loss Function
    parser.add_argument("--loss_function", type=str, default="mse", choices=["mse", "smooth_l1", "huber"])
    parser.add_argument("--delta", type=float, default=0.5, help="huber loss参数")

    parser.add_argument("--lambda_des", type=float, default=30, help="目的地损失权重")

    parser.add_argument("--lambda_j", type=float, default=100)
    parser.add_argument("--lambda_recon", type=float, default=1)
    parser.add_argument("--d_scale", type=float, default=1)

    parser.add_argument(
        "--info",
        type=str,
        default="",
        help="Name of training. " "It will be used in test folder",
    )
    return parser.parse_args()


def main(config):
    seed_torch(config.seed)
    t = trainer_ppt.Trainer(config)
    t.logger.info(f"[M] start training modules for {config.dataset_name.upper()} dataset.")
    if config.no_train:
        t.test_model()  # 测试模型
    else:
        t.fit()  # 训练模型


if __name__ == "__main__":
    config = parse_config()
    print(config)
    main(config)
