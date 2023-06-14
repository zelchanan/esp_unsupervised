import logging

import numpy as np
import torch
import torch.nn as nn

from data import examples
from utils import set_log
from torchinfo import summary


# class AE(nn.Module):
#     def __init__(self, blocks_num: int, block_size: int, layers_num: int, bottle_neck:int):
#         super().__init__()
#         self.bottle_nect:int  = bottle_neck
#         self.blocks_num = blocks_num
#         self.block_size = block_size
#         self.final_dim: int = blocks_num * block_size
#         self.layers_num = layers_num
#         self.final_hidden_layer_dim = (int(self.final_dim) // (2 ** self.layers_num)) ** 2
#         self.model = self.__create__()
#
#     def __create__(self):
#         ae_layers = []

class Encoder(nn.Module):

    def __init__(self, encoded_space_dim=-1, fc2_input_dim=-1):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=0),
            # nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 4, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        #
        # ### Flatten layer
        # self.flatten = nn.Flatten(start_dim=1)
        # ### Linear section
        # self.encoder_lin = nn.Sequential(
        #     nn.Linear(3 * 3 * 32, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, encoded_space_dim)
        # )

    def forward(self, x: torch.Tensor):
        x = self.encoder_cnn(x)
        # x = self.flatten(x)
        # x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim=-1, fc2_input_dim=-1):
        super().__init__()

        ### Convolutional section
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(4, 4, 3, stride=2, output_padding=0),
            nn.ConvTranspose2d(4, 4, 3, stride=2, output_padding=0),
            # nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, 3, stride=2, output_padding=0)
        )

        #
        # ### Flatten layer
        # self.flatten = nn.Flatten(start_dim=1)
        # ### Linear section
        # self.encoder_lin = nn.Sequential(
        #     nn.Linear(3 * 3 * 32, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, encoded_space_dim)
        # )

    def forward(self, x: torch.Tensor):
        x = self.decoder_cnn(x)
        # x = self.flatten(x)
        # x = self.encoder_lin(x)
        return x


class AE(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    # ae = AE()
    e = Encoder()
    summary(model=e, input_size=(1, 2000, 2000))
    d = Decoder()
    summary(model=d, input_size=(4, 2000, 2000))
    ae = AE()
    summary(model=ae, input_size=(1, 2000, 2000))
