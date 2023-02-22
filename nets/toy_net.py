# https://discuss.pytorch.org/t/how-to-print-the-computed-gradient-values-for-a-network/34179/6
import logging

import numpy as np
import torch
import torch.nn as nn

from data.examples import create_block_matrix
from utils import set_log
from torchinfo import summary


class ToyModel(nn.Module):
    def __init__(self, blocks_num: int, block_size: int, layers_num: int):
        super().__init__()
        self.blocks_num = blocks_num
        self.block_size = block_size
        self.final_dim: int = blocks_num * block_size
        self.layers_num = layers_num
        self.final_hidden_layer_dim = (int(self.final_dim) // (2 ** self.layers_num)) ** 2
        self.model = self.__create__()

        # self.summer = torch.ones(block_size)
        # self.selector = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, self.final_dim), padding="same"), nn.ReLU(),

    def __create__(self) -> nn.Sequential:
        layers = []
        layers.append(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, 1), padding="same"))
        layers.append(nn.ReLU())
        for i in range(self.layers_num):
            current_size = int(self.final_dim / 2 ** i)
            layers.append(nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, current_size), padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(current_size, 1), padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
            layers.append(nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
        layers.append(nn.Flatten(start_dim=1))
        layers.append(nn.Linear(in_features=10 * self.final_hidden_layer_dim, out_features=self.final_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.model(x)
        # logging.info(f"max: {weights.max().item()}. min: {weights.min().item()}")
        reshaped = torch.reshape(weights, (-1, self.blocks_num, self.block_size))
        return torch.nn.Softmax(dim=2)(reshaped)


def loss_fn(block: torch.Tensor, weights: torch.Tensor, final_dim: int) -> torch.Tensor:
    weights = torch.reshape(weights, (-1, 1, final_dim))
    # logging.info(f"nan: {torch.any(torch.isnan(weights))}")
    reg_loss = 0.01 * torch.mean(torch.sqrt(weights))  # anyway it cannot be smooshed
    #collision_loss = torch.mul((torch.matmul(weights, block.squeeze()), weights))
    collision_loss = torch.mul(torch.matmul(weights, block.squeeze()), weights)
    #print(collision_loss.shape)
    collision_loss = torch.sum(torch.mul(torch.matmul(weights, block.squeeze()), weights))

    # logging.info(f"reg_loss: {reg_loss.item()}, collision_loss: {collision_loss.item()}")
    return collision_loss


def train(model: ToyModel, batch_size: int, blocks_num: int, block_size: int, epsilon: float,
          seed: int = -1):
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i in range(500):
        data_batch = create_block_matrix(batch_size=batch_size, blocks_num=blocks_num, block_size=block_size,
                                         epsilon=epsilon, seed=seed)
        data_batch = torch.from_numpy(data_batch).float().cuda()
        optimizer.zero_grad()
        output = model(data_batch)
        loss = loss_fn(block=data_batch, weights=output, final_dim=model.final_dim)
        # plot_triple(data, labels, pred)

        # Backpropagation
        loss.backward()
        # print(loss.item())
        optimizer.step()
        logging.info(f"ind: {i}, {loss.item()}")


if __name__ == "__main__":
    set_log()
    blocks_num = 30
    block_size = 10
    batch_size = 1
    epsilon = 0.1
    seed = 1

    model = ToyModel(blocks_num=blocks_num, block_size=block_size, layers_num=3).float().cuda()
    summary(model, input_size=(4, 1, block_size * blocks_num, block_size * blocks_num))
    train(model=model, batch_size=batch_size, blocks_num=blocks_num, block_size=block_size, epsilon=epsilon, seed=seed)
