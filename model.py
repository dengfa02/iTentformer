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
        return x[:, :, :-self.chomp_size].contiguous()


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
    #         self.downsample.weight.data.normal_(0, 0.01)
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

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x为embedding后的inputs
        """
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TCN(nn.Module):
    def __init__(self, input_size_tcn, local_intent_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size_tcn, num_channels, kernel_size, dropout=dropout)
        self.linear_intent = nn.Linear(num_channels[-1], local_intent_size)

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return output


class FusionBlock(nn.Module):
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
        x = x.transpose(1, 2)
        x = self.embedding(x)

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
        attn_weights = F.softmax(attn_weights, dim=-1)
        att_values = torch.matmul(attn_weights, v)

        att_values = att_values.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        at = att_values + x
        out = self.norm(at)

        return out, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class TrajModel(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, concat_dim, input_length, dropout):
        super(TrajModel, self).__init__()
        self.Fusion = FusionBlock(d_model, input_length, dropout)
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.concat_linear = nn.Linear(concat_dim * 10, output_dim * 10)
        self.fc = nn.Linear(d_model, 10)
        self.fc2 = nn.Linear(d_model, 10)

        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=512, dropout=dropout)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=1)

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0.1)
        self.predictor = nn.Linear(d_model, output_dim)

    def forward(self, src, intent):
        src = self.input_embedding(src).cuda()
        src = self.positional_encoding(src).cuda()

        encode = self.encoder(src, src_key_padding_mask=None)  # 16*8*128
        encode = self.fc(encode)  # 16*8*10
        memory_cat = torch.cat((encode.transpose(1, 2), intent), dim=-1)  # 16*10*40
        memory_cat, attn_weights = self.Fusion(memory_cat)  # 16*40*128
        memory_cat = self.fc2(memory_cat)  # 16*40*10

        memory_cat = memory_cat.reshape(memory_cat.size(0), -1)
        memory_cat = self.concat_linear(memory_cat)  # 16*400->16*40
        out = memory_cat.reshape(memory_cat.size(0), 10, -1)

        return out, attn_weights


class iTentformer(nn.Module):

    def __init__(self, input_size_tcn, input_size, local_intent_size, output_size, concat_dim, input_length,
                 num_channels, kernel_size, d_model, dropout):
        super().__init__()
        self.TCN = TCN(input_size_tcn, local_intent_size, num_channels, kernel_size, dropout=dropout)
        self.TrajModel = TrajModel(input_size, d_model, output_size, concat_dim, input_length, dropout)

    def forward(self, delta, src):
        intent = self.TCN(delta)
        out, _ = self.TrajModel(src.transpose(1, 2), intent)
        out_intent = self.TCN.linear_intent(intent)

        return out_intent, out
