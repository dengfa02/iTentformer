import math
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.nn.init import kaiming_normal_
from scipy.stats import multivariate_normal


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class SENet(nn.Module):
    def __init__(self, num_classes=1000, reduction=16):
        super(SENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(64, reduction=reduction)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.se(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()  # 在前向传播方法中，
        # 从输入张量x的末尾移除self.chomp_size个元素，并使用.contiguous()方法确保返回的张量是连续的。
        # 这个操作通常用于调整卷积操作的输出大小，以便与下游网络或任务的要求相匹配。


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.se = SELayer(n_outputs, reduction=16)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2, self.se)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    # def init_weights(self):
    #     self.conv1.weight.data.normal_(0, 0.01)
    #     self.conv2.weight.data.normal_(0, 0.01)
    #     if self.downsample is not None:
    #         self.downsample.weight.data.normal_(0, 0.01)  # 正态分布随机初始化神经网络中的权重矩阵
    def init_weights(self):
        kaiming_normal_(self.conv1.weight.data, mode='fan_in', nonlinearity='relu')
        kaiming_normal_(self.conv2.weight.data, mode='fan_in', nonlinearity='relu')
        if self.downsample is not None:
            kaiming_normal_(self.downsample.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)  # 将layers 列表中的每个元素解包并传递给nn.Sequential构造函数，
        # 以便构建一个包含所有卷积块的序列模块。

    def forward(self, x):
        return self.network(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0], [1], [2], [3], ...]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TCN(nn.Module):
    def __init__(self, input_size_tcn, local_intent_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size_tcn, num_channels, kernel_size, dropout=dropout)
        self.linear_intent = nn.Linear(num_channels[-1], local_intent_size)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (B, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        # output = output[:, -1, :]
        return output


class FusionBlock(nn.Module):  # concat+点积注意力
    def __init__(self, d_model, input_length, dropout):
        super(FusionBlock, self).__init__()
        self.embedding = nn.Linear(input_length, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.input_length = input_length
        self.d_model = d_model

        self.norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)

        self.num_heads = 4
        self.head_dim = d_model // 4


    def forward(self, x):
        x = x.transpose(1, 2)  # 16*40*10
        x = self.embedding(x)  # 16*40*128

        batch_size = x.size(0)
        seq_len = x.size(1)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float))
        attn_weights = F.softmax(attn_weights, dim=-1)  # 查看归一化权重
        att_values = torch.matmul(attn_weights, v)

        att_values = att_values.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        at = att_values + x
        out = self.norm(at)

        return out, attn_weights


class TrajModel(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, concat_dim, input_length, dropout):
        super(TrajModel, self).__init__()
        self.Fusion = FusionBlock(d_model, input_length, dropout)
        self.input_embedding = nn.Linear(input_dim, d_model)
        # self.output_embedding = nn.Linear(output_dim, d_model)
        self.concat_linear = nn.Linear(concat_dim * 10, output_dim * 10)
        self.fc = nn.Linear(d_model, 10)
        self.fc2 = nn.Linear(d_model, 10)

        # self.fc3 = nn.Linear(40, output_dim)
        # self.relu = nn.ReLU()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=512,
                                                        batch_first=True, dropout=dropout, activation='relu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0.1)
        self.predictor = nn.Linear(d_model, output_dim)

    def forward(self, src, intent):
        # 对src和tgt进行编码
        src = self.input_embedding(src).cuda()
        # tgt = self.output_embedding(tgt).cuda()
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src).cuda()
        # tgt = self.positional_encoding(tgt).cuda()

        encode = self.encoder(src, src_key_padding_mask=None)  # 16*8*128
        encode = self.fc(encode)  # 16*8*10
        memory_cat = torch.cat((encode.transpose(1, 2), intent), dim=-1)  # 16*10*40
        memory_cat, attn_weights = self.Fusion(memory_cat)  # 16*40*128
        memory_cat = self.fc2(memory_cat)  # 16*40*10

        memory_cat = memory_cat.reshape(memory_cat.size(0), -1)
        memory_cat = self.concat_linear(memory_cat)  # 16*400->16*40
        out = memory_cat.reshape(memory_cat.size(0), 10, -1)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out, attn_weights

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask


class IntentFlowNet(nn.Module):

    def __init__(self, input_size_tcn, input_size, local_intent_size, output_size, concat_dim, input_length,
                 num_channels, kernel_size, d_model, dropout):
        super(IntentFlowNet, self).__init__()
        self.TCN = TCN(input_size_tcn, local_intent_size, num_channels, kernel_size, dropout=dropout)
        self.TrajModel = TrajModel(input_size, d_model, output_size, concat_dim, input_length, dropout)
        # self.predictor = nn.Linear(d_model, output_size)
        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """

    def forward(self, delta, src):
        intent = self.TCN(delta)
        out, attn_weights = self.TrajModel(src.transpose(1, 2), intent)
        # intent = intent.reshape(intent.size(0), -1)
        out_intent = self.TCN.linear_intent(intent)
        # out_intent = out_intent.reshape(out_intent.size(0), 10, -1)

        return out_intent, out, attn_weights
