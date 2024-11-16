import os
import datetime
import torch
import torch.nn as nn

# from models.model_AIO import model_encdec
from trainer.evaluations import *
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
                fde_, currentValue = evaluate_trajectory(
                    self.val_loader, self.model, self.config, self.device
                )
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

            # 冻结token
            # self.model.rand_token[0, c_pred_len-1].detach().requires_grad_(False)

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

        for _, (ped, neis, mask, initial_pos) in enumerate(self.train_loader):
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

            # 对输入轨迹进行编码
            past_state = self.model.traj_encoder(traj_norm)  # (513, 20, 128)

            # 提取社会交互信息
            int_feat = self.model.spatial_interaction(past_state[:, :self.config.past_len], neis[:, :, :self.config.past_len], mask)  # (512, 8, 128)
            # past_state[:, :self.config.past_len] = int_feat  # (512, 20, 128)

            # 本次使用的真实值
            # past_feat = past_state[:, :self.total_len - c_pred_len - 1]
            past_feat = int_feat

            # 先预测目的地
            des_token = repeat(
                self.model.rand_token[:, -1:], "() n d -> b n d", b=past_feat.size(0)
            )  # (513, 1, 128)  可学习编码
            des_state = self.model.token_encoder(des_token)  # (513, 1, 128)  对可学习编码进行编码

            # 位置编码
            des_input = torch.cat((past_feat, des_state), dim=1)
            des_feat = self.model.get_pe(des_input)
            des_feat = self.model.AR_Model(
                des_feat, mask_type="causal"
            )  # (514, 9, 128)
            pred_des = self.model.predictor_Des(
                des_feat[:, -1]
            )  # generate 20 destinations for each trajectory  (512, 1, 40)  每条轨迹生成20个目的地
            pred_des = pred_des.view(pred_des.size(0), self.config.goal_num, -1)  # (512, 20, 2)

            # 目的地损失
            true_des_feat = self.model.traj_encoder(destination.squeeze())  # (514, 128) 对预测的目的地进行编码
            # pred_des_feat = self.model.des_loss_encoder(des_feat[:, -1])  # (514, 128) 对预测的目的地进行编码

            loss += self.criterion_des(des_feat[:, -1], true_des_feat) * self.config.lambda_des
            loss += self.loss_function(pred_des, destination) * self.config.lambda_des

            # 从20个预测目的地中找到和真实目的地最接近的目的地
            distances = torch.norm(destination - pred_des, dim=2)  # (514, 20)  计算每个预测目的地与真实目的地的距离
            index_min = torch.argmin(distances, dim=1)  # (514)  找到每个轨迹的最小距离的索引
            min_des_traj = pred_des[torch.arange(0, len(index_min)), index_min]  # (514, 2)  找到每个轨迹的最小距离的目的地
            destination_prediction = min_des_traj  # (514, 2)  预测的目的地

            # 本次预测的观察帧
            traj_input = past_feat[:, :self.config.past_len]

            for c_pred_len in range(1, self.config.future_len):
                # 本次预测的帧id
                pred_frame_id = [v for v in range((self.config.past_len),(self.config.past_len+c_pred_len))]

                # 预测帧token
                fut_token = self.model.rand_token[0, [v-8 for v in pred_frame_id]].unsqueeze(0)
                fut_token = repeat(
                fut_token, "() n d -> b n d", b=traj_input.size(0)
                )  # (514, 11, 128)  可学习编码
                fut_feat = self.model.token_encoder(fut_token)

                # 目的地  训练用真实目的地
                des = self.model.traj_encoder(destination.squeeze())  # (514, 128) 对预测的目的地进行编码

                # 拼接 观察帧轨迹 + 可学习编码 + 预测的目的地编码
                concat_traj_feat = torch.cat((traj_input, fut_feat, des.unsqueeze(1)), 1)  # (514, 10, 128)
                concat_traj_feat = self.model.get_pe(concat_traj_feat)
                prediction_feat = self.model.AR_Model(concat_traj_feat, mask_type="all")  # (514, 10, 128)  Transformer  没有用mask

                pred_traj = self.model.traj_decoder(prediction_feat[:, self.config.past_len:-1])  # (514, 2)  预测的中间轨迹

                # 对第19帧进行编码  得到第二十帧的预测轨迹
                des_prediction = self.model.traj_decoder_20(
                    prediction_feat[:, -1]
                ) + destination_prediction  # (514, 1, 2)  预测终点的残差

                # 拼接预测轨迹
                pred_results = torch.cat(
                    (pred_traj, des_prediction.unsqueeze(1)), 1
                )

                # 计算轨迹损失
                traj_gt = traj_norm[:, pred_frame_id[0]:pred_frame_id[-1]+1]  # 中间轨迹
                traj_gt = torch.cat((traj_gt, traj_norm[:, -1].unsqueeze(1)), 1)  # 加上终点

                loss += self.criterion(pred_results, traj_gt)



                train_loss += loss.item()
                count += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0, norm_type=2
            )
            self.opt.step()

        return train_loss / count
