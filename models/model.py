"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import repeat
import numpy as np

from models.layer_utils import *
from scipy import interpolate
from copy import deepcopy

from models.transformer_encoder import Encoder

# -----------------------------------------------------------------------------


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) *
                    (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # qkv
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )  # 下三角矩阵,主对角线及以下全为1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, mask_type="causal", mask_input=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # (513, 19, 128)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        k = k.view(B, k.size(1), self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)  (513, 4 ,19, 32)
        q = q.view(B, q.size(1), self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, v.size(1), self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * \
            (1.0 / math.sqrt(k.size(-1)))  # (513, 4, 19, 19)
        if mask_input != None:
            mask = mask_input == 0
            # print(mask_input[0])
        elif mask_type == "causal":  # 前三个训练阶段用这个
            mask = self.bias[:, :, :T, :T] == 0
        elif mask_type == "all":  # 最后一个阶段用这个
            self.bias[:, :, :T, :T] = 1
            mask = self.bias[:, :, :T, :T] == 0
        else:
            self.bias[:, :, :T, :T] = 1
            mask = self.bias[:, :, :T, :T] == 0

        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side  (513, 19, 128)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config, pretrain=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)  # 自注意力机制
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 2 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(2 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x, mask_type="causal", mask_input=None):
        self.mlp[1].approximate = 'none'  # 自己加,版本不一样,防止报错
        # TODO: check that training still works
        x = x + self.dropout(self.attn(self.ln1(x), mask_type,
                          mask_input))  # 每次进入attn之前都做归一化
        # x = self.dropout(x)
        x = x + self.dropout(self.mlp(self.ln2(x)))  # 每次进入mlp之前都做归一化
        return x


class GPT(nn.Module):
    """GPT Language Model"""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size  # 128
        params_given = all(
            [
                config.int_num_layers_list is not None,
                config.n_head is not None,
                config.n_embd is not None,
            ]
        )  # 只有这三个值都给定才为True

        assert params_given
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),

                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block(config)
                                for _ in range(config.int_num_layers_list[0])]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = MLP(config.n_embd, config.vocab_size, (64,))

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.int_num_layers_list[0])
                )

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, input, social=None, targets=None, mask_type="causal", mask_input=None
    ):
        x = self.transformer.drop(input)  # (513, 19, 128)
        for block in self.transformer.h:
            x = block(x, mask_type, mask_input)
        output_feat = self.transformer.ln_f(x)  # (513, 19, 128)  对输出做归一化
        return output_feat


EPSILON = np.finfo(np.float32).tiny


class Final_Model(nn.Module):

    def __init__(self, config):
        super(Final_Model, self).__init__()

        # 保存配置文件
        self.config = config

        self.name_model = "PPT_Model"
        self.use_cuda = config.cuda
        self.past_len = config.past_len  # 8
        self.future_len = config.future_len  # 12
        self.total_len = self.past_len + self.future_len  # 20

        self.dropout = config.traj_embed_dropout

        # 位置编码
        self.wpe=nn.Embedding(config.block_size, config.n_embd)  # 位置编码

        # 对输入轨迹进行编码
        # self.traj_encoder = nn.Linear(config.vocab_size, config.n_embd)  # FC(2, 128)  *********************
        self.traj_encoder = nn.Sequential(
            nn.Linear(config.vocab_size, config.n_embd),
            # nn.ReLU(),
            # nn.Dropout(self.dropout)
        )  # FC(2, 128)
        self.AR_Model = GPT(config)  # 搭建模型
        self.predictor_1 = nn.Linear(
            config.n_embd, config.vocab_size)  # FC(128, 2)

        # 用于预测目的地
        self.predictor_Des = MLP(config.n_embd, config.goal_num * 2, (512, 512, 512))
        self.rand_token = nn.Parameter(torch.rand(
            1, 12, config.n_embd))  # (1, 12, 128)  可学习编码
        # self.token_encoder = nn.Linear(config.n_embd, config.n_embd)  # 对可学习编码进行编码
        self.token_encoder = MLP(config.n_embd, config.n_embd, (512, 512, 512))

        # 社交关系
        # self.nei_embedding = nn.Linear(config.vocab_size * self.past_len, config.n_embd)
        self.nei_embedding = nn.Linear(config.vocab_size, config.n_embd)
        self.social_decoder = Encoder(
            config.n_embd, config.int_num_layers_list[1], config.n_head, config.forward_expansion, islinear=False
        )

        # 阶段三新加的层
        # self.traj_decoder = nn.Linear(
        #     config.n_embd, config.vocab_size)  # 得到第十帧到第十九帧的预测轨迹
        self.traj_decoder = MLP(
            config.n_embd, config.vocab_size, (512, 512, 512)
        )  # 得到第十帧到第十九帧的预测轨迹
        self.traj_decoder_20 = MLP(
            config.n_embd, config.vocab_size, (512, 512, 512)
        )  # trajectory decoder for the 20th trajectory point  第二十帧的预测轨迹

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # He initialization for weights
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    # 偏置初始化为0
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif isinstance(module, nn.LayerNorm):
                # LayerNorm层保持标准初始化
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_pe(self, input, if_social=False):
        device = input.device
        if if_social:
            b, n, t, d = input.size()
        else:
            b, t, d = input.size()
        # assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(1, t + 1, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)   (1, 19)
        # if t == 9:
        if not if_social:
            pos[:, -1] = 20
        tok_emb = input  # (513, 19, 128)
        pos_emb = self.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)  (1, 19, 128)
        if if_social:
            pos_emb = pos_emb.repeat(tok_emb.shape[1], 1, 1).unsqueeze(0)

        return tok_emb + pos_emb

    def spatial_interaction(self, ped, neis, mask):
        # ped (512, 1, 64)
        # neis (512, 18, 8, 2)  N is the max number of agents of current scene
        # mask (512, 2, 2) is used to stop the attention from invalid agents

        # 对邻居进行编码
        nei_embedding = self.nei_embedding(neis)  # (512, 8, 18, 128)
        nei_embedding[:, 0, :] = ped  # (512, 18, 128)
        # nei_embedding = self.get_pe(nei_embedding, if_social=True)
        _, n, t, _ = neis.shape
        nei_embeddings = nei_embedding.reshape(neis.shape[0], -1,
                            self.config.n_embd)  # (512, 162, 128)

        mask = mask[:, 0:1].reshape(neis.shape[0], -1, 1).repeat(1, 1, neis.shape[2]).view(neis.shape[0], -1).unsqueeze(1).repeat(1, nei_embeddings.shape[1], 1)  # (512, 1, 9) -> (512, 9, 1) -> (512, 9, 18) -> (512, 162) -> (512, 162, 1) -> (512, 162, 162)
        mask_traj = torch.tril(torch.ones((1, ped.shape[1], ped.shape[1]))).repeat(neis.shape[0], neis.shape[1], neis.shape[1]).to(device=neis.device)  # (512, 162, 162)
        mask = mask * mask_traj  # (512, 162, 162)
        int_feat = self.social_decoder(
            nei_embeddings, nei_embeddings, mask
        )  # [B K embed_size]  (512, 112, 128)

        # return int_feat.reshape(neis.shape[0], n, t, -1)[:, 0]
        return int_feat.contiguous().view(neis.shape[0], n, t, -1)[:, 0]

    def loss_function(self, pred, gt):  # pred:[B, 20, 2]
        # joint loss
        JL = self.joint_loss(
            pred) if self.config.lambda_j > 0 else 0.0  # 多样性损失
        RECON = self.recon_loss(
            pred, gt) if self.config.lambda_recon > 0 else 0.0  # 目的地损失
        # print('JL', JL * self.config.lambda_j, 'RECON', RECON * self.config.lambda_recon)
        loss = JL * self.config.lambda_j + RECON * self.config.lambda_recon  # 总损失
        return loss

    def joint_loss(self, pred):  # pred:[B, 20, 2]  多样性损失
        loss = 0.0
        for Y in pred:  # 取出来每个行人预测的20个目的地
            # 计算每两个目的地之间的距离  20个目的地两两组合,共有(19 * 20) / 2 = 190个组合
            dist = F.pdist(Y, 2) ** 2
            # 计算每个组合的损失,并取平均
            loss += (-dist / self.config.d_scale).exp().mean()
        loss /= pred.shape[0]  # 取所有行人预测的损失的平均
        return loss

    def recon_loss(self, pred, gt):
        distances = torch.norm(pred - gt, dim=2)
        # 找到每个行人预测的20个目的地中,与真实目的地距离最小的那个
        index_min = torch.argmin(distances, dim=1)
        # 计算每个行人预测的20个目的地中,与真实目的地距离最小的那个距离
        min_distances = distances[torch.arange(0, len(index_min)), index_min]
        # 计算所有行人预测的20个目的地中,与真实目的地距离最小的那个距离的平均
        loss_recon = torch.sum(min_distances) / distances.shape[0]
        return loss_recon

    def get_trajectory(self, traj_norm, neis, mask, c_pred_len):
        predictions = torch.Tensor().cuda()

        # 对输入轨迹进行编码
        past_state = self.traj_encoder(traj_norm)

        # 提取社会交互信息
        int_feat = self.spatial_interaction(
            past_state[:, :self.past_len], neis[:, :, :self.past_len], mask)  # (512, 8, 128)
        past_state[:, :self.past_len] = int_feat  # (512, 20, 128)

        # 本次使用的真实值
        past_feat = int_feat

        # 先预测目的地
        des_token = repeat(
            self.rand_token[:, -1:], "() n d -> b n d", b=past_feat.size(0)
        )  # (513, 1, 128)  可学习编码
        des_state = self.token_encoder(des_token)  # (513, 1, 128)  对可学习编码进行编码

        des_input = torch.cat((past_feat, des_state), dim=1)
        des_input = self.get_pe(des_input)
        des_feat = self.AR_Model(
            des_input
        )  # (514, 9, 128)
        pred_des = self.predictor_Des(
            des_feat[:, -1]
        )  # generate 20 destinations for each trajectory  (512, 1, 40)  每条轨迹生成20个目的地
        pred_des = pred_des.view(pred_des.size(0), self.config.goal_num, -1)

        # generate N=20 future trajectories
        for i in range(self.config.goal_num):
            fut_token = repeat(
                self.rand_token[:, :c_pred_len], "() n d -> b n d", b=traj_norm.size(0)
            )

            fut_feat = self.token_encoder(fut_token)
            des_feat = self.traj_encoder(pred_des[:, i])
            traj_feat = torch.cat(
                (past_feat, fut_feat, des_feat.unsqueeze(1)), 1)  # (512, 20, 128)

            traj_feat = self.get_pe(traj_feat)
            prediction_feat = self.AR_Model(traj_feat, mask_type="all")

            mid_prediction = self.traj_decoder(
                prediction_feat[:, self.past_len:self.past_len+c_pred_len])  # (512, 11, 2)
            des_prediction = self.traj_decoder_20(
                prediction_feat[:, -1]
            ) + pred_des[:, i]
            total_prediction = torch.cat(
                (mid_prediction, des_prediction.unsqueeze(1)), 1
            )

            prediction_single = total_prediction
            predictions = torch.cat(
                (predictions, prediction_single.unsqueeze(1)), dim=1
            )
        return predictions
