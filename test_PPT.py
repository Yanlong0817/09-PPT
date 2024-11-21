import argparse
from trainer import test_final_trajectory as trainer_ppt

import numpy as np
import random
import torch


def prepare_seed(rand_seed):
	np.random.seed(rand_seed)
	random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed_all(rand_seed)


def parse_config():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("--cuda", default=True)

    # 数据集根路径
    parser.add_argument("--dataset_path", type=str, default="/home/tyl/code/1005idea/dataset/")
    # 具体数据名
    parser.add_argument("--dataset_name", type=str, default="sdd")
    parser.add_argument("--dist_threshold", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=512)

    # 数据增强相关
    parser.add_argument("--use_augmentation", type=bool, default=False)
    parser.add_argument("--smooth", type=bool, default=False)
    parser.add_argument("--rotation", type=bool, default=False)

    # 数据缩放
    parser.add_argument("--data_scaling", type=list, default=[1.9, 0.4])

    parser.add_argument("--past_len", type=int, default=8, help="length of past (in timesteps)")
    parser.add_argument("--future_len", type=int, default=12, help="length of future (in timesteps)")
    parser.add_argument("--dim_embedding_key", type=int, default=24)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--model_Pretrain", default='/home/tyl/code/1005idea/09-PPT2/training/sdd/2024.11.17_baseline/model.ckpt')

    parser.add_argument("--reproduce", default=False)

    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    if config.reproduce:
        config.model_Pretrain = './training/Pretrained_Models/SDD/model_ALL.ckpt'
        print(config.model_Pretrain)
        t = trainer_ppt.Trainer(config)
        t.fit()
    else:
        print(config.model_Pretrain)
        t = trainer_ppt.Trainer(config)
        t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
