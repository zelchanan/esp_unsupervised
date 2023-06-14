# https://discuss.pytorch.org/t/how-to-print-the-computed-gradient-values-for-a-network/34179/6
import logging
import pickle

import numpy as np
import torch
import torch.nn as nn

from data import examples
from utils import set_log
from torchinfo import summary


class BlockModel(nn.Module):
    def __init__(self, blocks_num: int, block_size: int, layers_num: int):
        super().__init__()
        self.blocks_num = blocks_num
        self.block_size = block_size
        self.vec_len = self.blocks_num * self.block_size
        self.model = self.__create__()

        # self.summer = torch.ones(block_size)
        # self.selector = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, self.final_dim), padding="same"), nn.ReLU(),

    def __create__(self) -> nn.Sequential:
        layers = []
        layers.append(
            torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=self.block_size, stride=self.block_size))
        layers.append(nn.ReLU())
        layers.append(torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=1))
        layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=self.blocks_num ** 2 * 20, out_features=self.vec_len))
        layers.append(nn.ReLU())
        layers.append(
            nn.Linear(in_features=self.vec_len, out_features=self.vec_len))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.model(x)
        # logging.info(f"max: {weights.max().item()}. min: {weights.min().item()}")
        reshaped = torch.reshape(weights, (-1, self.blocks_num, self.block_size))
        return torch.nn.Softmax(dim=2)(reshaped)

    def loss_fn(self, collision_matrix: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(weights.flatten(), collision_matrix.squeeze()), weights.flatten())

    def opt(self, collision_matirx: np.ndarray, iterations: int) -> np.ndarray:
        block = np.expand_dims(np.expand_dims(collision_matirx, 0), 0)
        block = torch.Tensor(block).float().cpu()

        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for i in range(iterations):
            # data_batch = create_block_matrix(batch_size=batch_size, blocks_num=blocks_num, block_size=block_size,
            #                                  epsilon=epsilon, seed=seed)
            optimizer.zero_grad()
            output = self(block)
            loss = self.loss_fn(collision_matrix=block, weights=output)
            # plot_triple(data, labels, pred)

            # Backpropagation
            loss.backward()
            # print(loss.item())
            optimizer.step()
            logging.info(f"ind: {i}, {loss.item()}")
        return output.squeeze().detach().cpu().numpy()

    # def loss_fn(block: torch.Tensor, weights: torch.Tensor, final_dim: int) -> torch.Tensor:
    #     weights = torch.reshape(weights, (-1, 1, final_dim))
    #     # logging.info(f"nan: {torch.any(torch.isnan(weights))}")
    #     reg_loss = 0.01 * torch.mean(torch.sqrt(weights))  # anyway it cannot be smooshed
    #     # collision_loss = torch.mul((torch.matmul(weights, block.squeeze()), weights))
    #     collision_loss = torch.mul(torch.matmul(weights, block.squeeze()), weights)
    #     # print(collision_loss.shape)
    #     collision_loss = torch.sum(torch.mul(torch.matmul(weights, block.squeeze()), weights))
    #
    #     # logging.info(f"reg_loss: {reg_loss.item()}, collision_loss: {collision_loss.item()}")
    #     return collision_loss

    # def loss_fn():
    def loss_fn(self, collision_matrix: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # weights = torch.reshape(weights, [self.blocks_num, self.block_size])
        # sm_weights = torch.nn.Softmax(dim=-1)(weights).flatten()
        return torch.matmul(torch.matmul(weights.flatten(), collision_matrix.squeeze()), weights.flatten())

    def opt(self, collision_matirx: np.ndarray, iterations: int) -> np.ndarray:
        block = np.expand_dims(np.expand_dims(collision_matirx, 0), 0)
        block = torch.Tensor(block).float().cpu()

        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for i in range(iterations):
            # data_batch = create_block_matrix(batch_size=batch_size, blocks_num=blocks_num, block_size=block_size,
            #                                  epsilon=epsilon, seed=seed)
            optimizer.zero_grad()
            output = self(block)
            loss = self.loss_fn(collision_matrix=block, weights=output)
            # plot_triple(data, labels, pred)

            # Backpropagation
            loss.backward()
            # print(loss.item())
            optimizer.step()
            logging.info(f"ind: {i}, {loss.item()}")
        return output.squeeze().detach().cpu().numpy()


if __name__ == "__main__":
    set_log(level=logging.WARN)
    blocks_num = 100
    block_size = 20
    batch_size = 1
    epsilon = 0.1
    seed = 1
    full_sol = True
    collision_matrix = examples.create_block_matrix(batch_size=batch_size, blocks_num=blocks_num, block_size=block_size,
                                                    epsilon=0.1,
                                                    seed=1, diagonal_blocks=False).squeeze()
    fname = f"data/yael_dataset2/gnn_k_{blocks_num}_m_{block_size}_e_10{'_full_sol' if full_sol else ''}.p"
    print(fname)

    collision_matrix = pickle.load(open(fname, 'rb'))[0]
    mask = examples.get_diagonal_blocks(blocks_num=blocks_num, block_size=block_size)
    collision_matrix[mask] = 0
    for i in range(100):
        model = BlockModel(blocks_num=blocks_num, block_size=block_size, layers_num=3).float().cpu()
        # summary(model, input_size=(1, 1, block_size * blocks_num, block_size * blocks_num))
        o = model.opt(collision_matrix, iterations=100)
        csol = rm_sol = examples.remove_collisions(collision_matrix, np.round(o))
        print(o.flatten() @ collision_matrix @ o.flatten(), csol.sum())
