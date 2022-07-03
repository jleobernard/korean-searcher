import math

import torch
import torch.nn as nn
from torchvision import models
from torch import Tensor


class MyAttention(nn.Module):

    def __init__(self, embedding_size_in: int = 512, embedding_size_out: int = 512, dropout: float = 0.1):
        super(MyAttention, self).__init__()
        self.Q = torch.randn(embedding_size_in, embedding_size_out)
        self.K = torch.randn(embedding_size_in, embedding_size_out)
        self.V = torch.randn(embedding_size_in, embedding_size_out)
        self.with_residual = embedding_size_in == embedding_size_out
        self.divisor = math.sqrt(embedding_size_in)
        self.softmax = nn.Softmax2d()
        self.norm = nn.BatchNorm2d(embedding_size_in, affine=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor):
        x = self.norm(x)
        q = torch.matmul(x, self.Q)
        k = torch.matmul(x, self.K)
        v = torch.matmul(x, self.V)
        prod = torch.matmul(q, torch.swapaxes(k, -1, -2)) / self.divisor
        softmax = self.softmax(prod)
        attention = torch.matmul(softmax, v)
        out = self.dropout(attention)
        if self.with_residual:
            out = out + x
        else:
            out = out
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))
        pe = torch.zeros(max_len, 1, embedding_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        pass


class AttentionEntryPoint(nn.Module):

    def __init__(self, nb_channels_in: int = 512, nb_channels_out: int = 512, dropout: float = 0.1):
        super(AttentionEntryPoint, self).__init__()
        self.attention = MyAttention(embedding_size_in=nb_channels_in, embedding_size_out=nb_channels_out)
        self.position_encoder = PositionalEncoding(nb_channels_in, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape = _, <resnet_embedding_size>, 13, 19
        x = torch.flatten(x, start_dim=2)
        # x.shape = _, <resnet_embedding_size>, 13 x 19
        x = torch.transpose(x, 1, 2)
        x = self.position_encoder(x)
        x = self.attention(x)
        return x


class AttetionKoreanSubParser(nn.Module):

    def __init__(self, nb_attention_layers: int = 4, embedding_size: int = 512):
        super(AttetionKoreanSubParser, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.downgraded_height = 13
        self.downgraded_width = 19
        self.resnet_out_channels = 512
        for param in resnet.parameters():
            param.requires_grad = True
        layers = list(resnet.children())[:8]
        self.features = nn.Sequential(*layers)
        self.attention_entry_point = \
            AttentionEntryPoint(nb_channels_in=self.resnet_out_channels, nb_channels_out=embedding_size)
        self.attentions = [MyAttention(embedding_size_in=embedding_size, embedding_size_out=embedding_size) for _ in range(nb_attention_layers)]
        self.norm = nn.BatchNorm2d(embedding_size, affine=False)
        # Concatenate along the width axis to get one cell by width unit length
        self.height_concatenator = \
            nn.Conv2d(in_channels=embedding_size, out_channels=embedding_size, kernel_size=(self.downgraded_height, 1))
        # Concatenate along the height axis to get one cell by height unit length
        self.width_concatenator = \
            nn.Conv2d(in_channels=embedding_size, out_channels=embedding_size, kernel_size=(1, self.downgraded_width))
        # Big Conv2d (i.e. DenseFF) to determine if there is a subtitle box or not
        self.presence_regression = \
            nn.Conv2d(kernel_size=(self.downgraded_height, self.downgraded_width), in_channels=embedding_size, out_channels=1)
        # Big Conv2D (i.e. DenseFF) along the height axis to get the upper and lower bound of the bounding box
        self.height_regression =\
            nn.Conv2d(kernel_size=(self.downgraded_height, 1), in_channels=embedding_size, out_channels=2)
        # Big Conv2D (i.e. DenseFF) along the width axis to get the left-most and right-most bound of the bounding box
        self.width_regression = \
            nn.Conv2d(kernel_size=(1, self.downgraded_width), in_channels=embedding_size, out_channels=2)

    def forward(self, x):
        # x.shape = _, 3, 500, 500
        x = self.features(x)
        # x.shape = _, 13 x 19, <resnet_embedding_size>
        x = self.attention_entry_point(x)
        # x.shape = _, 13 x 19, <embedding_size>
        for attention in self.attentions:
            x = attention(x)
        ####################
        # Is it there
        presence = self.presence_regression(x)
        ####################
        # Height branch
        x_along_height = self.height_concatenator(x)
        x_along_height = self.height_regression(x)
        ####################
        # Width branch
        x_along_width = self.width_concatenator(x)
        x_along_width = self.width_regression(x)
        x = self.final_regression(x)
        return presence, x_along_height, x_along_width

    def initialize_weights(self):
        pass
