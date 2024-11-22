import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


'''MLP model'''


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3)
                                   if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


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
