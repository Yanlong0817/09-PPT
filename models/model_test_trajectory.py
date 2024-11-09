import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat
from copy import deepcopy

class Final_Model(nn.Module):

    def __init__(self, config, pretrained_model):
        super(Final_Model, self).__init__()

        self.name_model = 'PPT_Model_Test'
        self.use_cuda = config.cuda
        self.dim_embedding_key = 128
        self.past_len = config.past_len
        self.future_len = config.future_len

        # 对输入轨迹进行编码
        self.traj_encoder = deepcopy(pretrained_model.traj_encoder)
        self.AR_Model = deepcopy(pretrained_model.AR_Model)
        self.predictor_1 = deepcopy(pretrained_model.predictor_1)

        # 用于预测目的地
        self.predictor_Des = deepcopy(pretrained_model.predictor_Des)
        self.rand_token = deepcopy(pretrained_model.rand_token)
        self.token_encoder = deepcopy(pretrained_model.token_encoder)

        # 社交关系
        self.nei_embedding = deepcopy(pretrained_model.nei_embedding)
        self.social_decoder = deepcopy(pretrained_model.social_decoder)

        # 阶段三新加的层
        self.traj_decoder = deepcopy(pretrained_model.traj_decoder)
        self.traj_decoder_9 = deepcopy(pretrained_model.traj_decoder_9)
        self.traj_decoder_20 = deepcopy(pretrained_model.traj_decoder_20)

        for p in self.parameters():
            p.requires_grad = False

    def spatial_interaction(self, ped, neis, mask):
        # ped (512, 1, 64)
        # neis (512, 2, 8, 2)  N is the max number of agents of current scene
        # mask (512, 2, 2) is used to stop the attention from invalid agents

        neis = neis.reshape(neis.shape[0], neis.shape[1], -1)  # (512, N, 16)
        nei_embeddings = self.nei_embedding(neis)  # (512, N, 128)

        mask = mask[:, 0:1].repeat(1, ped.shape[1], 1)  # [B K N]  (512, 8, 1)
        int_feat = self.social_decoder(
            ped, nei_embeddings, mask
        )  # [B K embed_size]  (512, 8, 128)

        return int_feat


    def forward(self, ped, neis, mask):
        predictions = torch.Tensor().cuda()

        traj_norm = ped  # 减去第八帧做归一化  (513, 20, 2)
        x = traj_norm[:, : self.past_len, :]  # 前8帧数据 (513, 8, 2)  观察帧
        destination = traj_norm[:, -1:, :]  # 最后一帧数据 (513, 1, 2)  目的地
        y = traj_norm[:, self.past_len :, :]  # 后12帧数据 (513, 12, 2)  预测帧

        # 对输入轨迹进行编码
        past_state = self.traj_encoder(traj_norm)  # (513, 8, 128)
        # 提取社会交互信息
        int_feat = self.spatial_interaction(past_state[:, :self.past_len], neis, mask)  # (512, 8, 128)
        past_state[:, :self.past_len] = int_feat  # (512, 8, 128)

        # 先预测目的地
        des_token = repeat(
            self.rand_token[:, -1:], "() n d -> b n d", b=past_state.size(0)
        )  # (513, 1, 128)  可学习编码
        des_state = self.token_encoder(des_token)  # (513, 1, 128)  对可学习编码进行编码
        des_feat = self.AR_Model(
            torch.cat((past_state, des_state), dim=1), mask_type="causal"
        )  # (514, 9, 128)
        pred_des = self.predictor_Des(
            des_feat[:, -1]
        )  # generate 20 destinations for each trajectory  (512, 1, 40)  每条轨迹生成20个目的地
        destination_prediction = pred_des.view(pred_des.size(0), 20, -1)  # (512, 20, 2)

        for i in range(20):
            fut_token = repeat(self.rand_token[:, :-1], '() n d -> b n d', b=ped.size(0))
            fut_feat = self.token_encoder(fut_token)
            traj_input = past_state

            # 目的地
            des = self.traj_encoder(destination_prediction[:, i])  # (514, 128) 对预测的目的地进行编码

            # 拼接 观察帧轨迹 + 可学习编码 + 预测的目的地编码
            concat_traj_feat = torch.cat((traj_input, fut_feat, des.unsqueeze(1)), 1)  # (514, 10, 128)
            prediction_feat = self.AR_Model(concat_traj_feat, mask_type="all")  # (514, 20, 128)  Transformer  没有用mask

            pred_traj = self.traj_decoder(prediction_feat[:, self.past_len:-1])  # (514, 2)  预测的中间轨迹

            # 对第19帧进行编码  得到第二十帧的预测轨迹
            des_prediction = self.traj_decoder_20(
                prediction_feat[:, -1]
            ) + destination_prediction[:, i]  # (514, 1, 2)  预测终点的残差

            # 拼接预测轨迹
            pred_results = torch.cat(
                (pred_traj, des_prediction.unsqueeze(1)), 1
            )

            predictions = torch.cat((predictions, pred_results.unsqueeze(1)), dim=1)

        return predictions