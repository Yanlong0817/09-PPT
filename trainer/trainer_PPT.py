import os
import datetime
import torch
import torch.nn as nn

# from models.model_AIO import model_encdec
from models.model import Final_Model

import logging
from dataset_loader import *
from models.preprocessing import *

# for visualization
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import yaml
from torch.utils.data import DataLoader
from einops import repeat
from openpyxl import Workbook, load_workbook

import wandb

torch.set_num_threads(5)


class Trainer:
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the autoencoder.
        :param config: configuration parameters (see train_ae.py)
        """

        # 实验结果根路径  training/sdd/2024-10-01_try_ST/training_Short_term
        self.root_path = f"training/{config.dataset_name}/{config.root_path}_{config.info}"

        # test folder creating
        self.folder_test = self.root_path

        # 创建目录
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + "/"

        obs_len = config.past_len  # 8
        fut_len = config.future_len  # 12
        self.total_len = obs_len + fut_len  # 20

        print("Preprocess data")

        train_dataset = TrajectoryDataset(
            dataset_path=config.dataset_path,
            dataset_name=config.dataset_name,
            dataset_type="train",
            rotation=config.rotation,
            obs_len=obs_len,
            dist_threshold=config.dist_threshold,
            smooth=config.smooth,
            use_augmentation=config.use_augmentation,  # 使用数据增强
        )

        val_dataset = TrajectoryDataset(
            dataset_path=config.dataset_path,
            dataset_name=config.dataset_name,
            dataset_type="test",
            rotation=False,
            obs_len=obs_len,
            dist_threshold=config.dist_threshold,
            use_augmentation=False,
            smooth=False
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, collate_fn=train_dataset.coll_fn, shuffle=True, num_workers=config.num_workers
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, collate_fn=val_dataset.coll_fn, shuffle=False, num_workers=config.num_workers
        )
        print("Loaded data!")

        if torch.cuda.is_available():  # 设置GPU编号
            torch.cuda.set_device(config.gpu)

        self.max_epochs = config.max_epochs

        # Initialize model
        self.model = Final_Model(
            config=config
        )

        # 优化器
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )

        # 余弦退火
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=config.max_epochs, eta_min=config.learning_rate_min
        )
        self.criterion = nn.SmoothL1Loss()
        self.criterion_des = nn.MSELoss()

        if config.cuda:
            self.model = self.model.cuda()
        self.start_epoch = 0
        self.config = config
        self.device = torch.device("cuda") if config.cuda else torch.device("cpu")

        self.logger_init()
        if config.use_wandb:
            experiment_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')}_{config.info}"
            wandb.init(
                project=config.wandb_project,
                group=config.wandb_group,
                config=config,
                name=experiment_name,
                tags=[config.dataset_name],
                notes=config.notes,
            )

    def logger_init(self):
        self.logger = logging.getLogger("test")
        self.logger.setLevel(level=logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        )

        file_handler = logging.FileHandler(os.path.join(self.folder_test, "train.log"))  # 保存训练日志的路径
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        # stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        # tensorboard writer
        # folder_log = "training/" + config.dataset_name + "/training_" + config.mode
        folder_log = os.path.join(self.root_path, 'logs')
        folder_log = os.path.join(self.root_path, "logs")
        self.tb_writer = SummaryWriter(folder_log)

    def print_model_param(self, model):
        total_num = sum(p.numel() for p in model.parameters())  # 模型的总参数
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练的参数
        self.logger.info(
            "\033[1;31;40mTrainable/Total: {}/{}\033[0m".format(
                trainable_num, total_num
            )
        )
        return 0

    def fit(self):
        self.print_model_param(self.model)  # 打印模型参数
        minValue = 200
        minADE = 2000
        minFDE = 2000
        for epoch in range(self.start_epoch, self.config.max_epochs):
            self.logger.info(" ----- Epoch: {}".format(epoch))
            loss = self._train_single_epoch(epoch)
            self.logger.info("Loss: {}".format(loss))
            self.tb_writer.add_scalar("train/loss", loss, epoch)

            self.scheduler.step()

            # validation
            # if (epoch + 1) % 1 == 0 and c_pred_len == self.config.future_len-1:
            if (epoch + 1) % 1 == 0:
                self.model.eval()
                fde_, currentValue = self.evaluate_trajectory()
                self.tb_writer.add_scalar("train/val", currentValue, epoch)

                if 3 * currentValue + fde_ < minValue:
                    minValue = 3 * currentValue + fde_
                    minFDE = fde_.item()
                    minADE = currentValue.item()
                    self.logger.info("min ADE value: {}".format(minADE))
                    self.logger.info("min FDE value: {}".format(minFDE))
                    torch.save(self.model, self.folder_test + f'model.ckpt')

                if self.config.use_wandb:
                    wandb.log({
                        f"ade": currentValue,
                        f"fde": fde_,
                        f"loss": loss,
                    })

        # 训练结束,输出最终ADE和FDE
        self.logger.info("min ADE value: {}".format(minADE))
        self.logger.info("min FDE value: {}".format(minFDE))

    def joint_loss(self, pred):  # pred:[B, 20, 2]  多样性损失
        loss = 0.0
        for Y in pred:  # 取出来每个行人预测的20个目的地
            dist = F.pdist(Y, 2) ** 2  # 计算每两个目的地之间的距离  20个目的地两两组合,共有(19 * 20) / 2 = 190个组合
            loss += (-dist / self.config.d_scale).exp().mean()  # 计算每个组合的损失,并取平均
        loss /= pred.shape[0]  # 取所有行人预测的损失的平均
        return loss

    def recon_loss(self, pred, gt):
        distances = torch.norm(pred - gt, dim=2)
        index_min = torch.argmin(distances, dim=1)  # 找到每个行人预测的20个目的地中,与真实目的地距离最小的那个
        min_distances = distances[torch.arange(0, len(index_min)), index_min]  # 计算每个行人预测的20个目的地中,与真实目的地距离最小的那个距离
        loss_recon = torch.sum(min_distances) / distances.shape[0]  # 计算所有行人预测的20个目的地中,与真实目的地距离最小的那个距离的平均
        return loss_recon

    def loss_function(self, pred, gt):  # pred:[B, 20, 2]
        # joint loss
        JL = self.joint_loss(pred) if self.config.lambda_j > 0 else 0.0  # 多样性损失
        RECON = self.recon_loss(pred, gt) if self.config.lambda_recon > 0 else 0.0  # 目的地损失
        # print('JL', JL * self.config.lambda_j, 'RECON', RECON * self.config.lambda_recon)
        loss = JL * self.config.lambda_j + RECON * self.config.lambda_recon  # 总损失
        return loss

    def L2_Loss(self, pred, gt):
        distances = torch.norm(pred - gt, dim=2)
        loss = torch.sum(torch.mean(distances, dim=1))
        return loss / distances.shape[0]

    def _train_single_epoch(self, epoch, c_pred_len=12):
        self.model.train()
        count = 0
        train_loss = 0.0

        for _, (ped, neis, mask, initial_pos, scene) in enumerate(self.train_loader):
            ped, neis, mask, initial_pos = (
                ped.to(self.device),
                neis.to(self.device),
                mask.to(self.device),
                initial_pos.to(self.device),
            )  # (512, 20, 2)  (512, 1, 20, 2)  (512, 1, 1)  (512, 1, 2)

            if self.config.dataset_name == "eth":
                ped[:, :, 0] = ped[:, :, 0] * self.config.data_scaling[0]
                ped[:, :, 1] = ped[:, :, 1] * self.config.data_scaling[1]

            scale = torch.randn(ped.shape[0]) * 0.05 + 1
            scale = scale.to(self.device)
            scale = scale.reshape(ped.shape[0], 1, 1)  # (512, 1, 1)
            ped = ped * scale
            scale = scale.reshape(ped.shape[0], 1, 1, 1)
            neis = neis * scale

            nei_obs = neis[:, :, :self.config.past_len]  # (512, 2, 8, 2)  邻居的观察帧

            traj_norm = ped  # 减去第八帧做归一化  (513, 20, 2)
            x = traj_norm[:, : self.config.past_len, :]  # 前8帧数据 (513, 8, 2)  观察帧
            destination = traj_norm[:, -1:, :]  # 最后一帧数据 (513, 1, 2)  目的地
            y = traj_norm[:, self.config.past_len :, :]  # 后12帧数据 (513, 12, 2)  预测帧

            trajectory = traj_norm + initial_pos  # 加上初始位置  (512, 20, 2)
            abs_past = trajectory[:, : self.config.past_len, :]  # 前8帧数据 (512, 8, 2)  未归一化版本
            initial_pose = trajectory[:, self.config.past_len - 1, :]  # 第八帧数据 (512, 2)  未归一化

            self.opt.zero_grad()
            loss = torch.tensor(0.0, device=self.device)
            loss = self.model(traj_norm, neis, mask, destination, scene)
            train_loss += loss.item()
            count += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0, norm_type=2
            )
            self.opt.step()

        return train_loss / count

    def evaluate_trajectory(self):
        # for the fulfillment stage or trajectory stage, we should have a fixed past/intention memory bank.
        samples = 0
        dict_metrics = {}

        ADE = []
        FDE = []

        with torch.no_grad():
            for _, (ped, neis, mask, initial_pos, scene) in enumerate(self.val_loader):
                ped, neis, mask, initial_pos = (
                        ped.to(self.device),
                        neis.to(self.device),
                        mask.to(self.device),
                        initial_pos.to(self.device),
                    )
                if self.config.dataset_name == "eth":
                    ped[:, :, 0] = ped[:, :, 0] * self.config.data_scaling[0]
                    ped[:, :, 1] = ped[:, :, 1] * self.config.data_scaling[1]

                traj_norm = ped
                output = self.model.get_trajectory(traj_norm, neis, mask, scene)
                output = output.data
                # print(output.shape)

                samples += output.shape[0]

                future_rep = traj_norm[:, 8:-1, :].unsqueeze(1).repeat(1, self.config.goal_num, 1, 1)
                future_goal = traj_norm[:, -1:, :].unsqueeze(1).repeat(1, self.config.goal_num, 1, 1)
                future = torch.cat((future_rep, future_goal), dim=2)
                distances = torch.norm(output - future, dim=3)

                fde_mean_distances = torch.mean(distances[:, :, -1:], dim=2) # find the tarjectory according to the last frame's distance
                fde_index_min = torch.argmin(fde_mean_distances, dim=1)
                fde_min_distances = distances[torch.arange(0, len(fde_index_min)), fde_index_min]
                FDE.append(fde_min_distances[:, -1])

                ade_mean_distances = torch.mean(distances[:, :, :], dim=2) # find the tarjectory according to the last frame's distance
                ade_index_min = torch.argmin(ade_mean_distances, dim=1)
                ade_min_distances = distances[torch.arange(0, len(ade_index_min)), ade_index_min]
                ADE.append(torch.mean(ade_min_distances, dim=1))

        if self.config.dataset_name == "sdd_world":  # 将sdd的世界坐标转化成像素坐标
            convert = torch.load("convert.ckpt")
            ADE = torch.cat(ADE) * torch.tensor(convert).to(self.device)
            FDE = torch.cat(FDE) * torch.tensor(convert).to(self.device)
        else:
            ADE = torch.cat(ADE)
            FDE = torch.cat(FDE)

        dict_metrics['fde_48s'] = FDE.mean() * self.config.divide_coefficient
        dict_metrics['ade_48s'] = ADE.mean() * self.config.divide_coefficient

        return dict_metrics['fde_48s'], dict_metrics['ade_48s']