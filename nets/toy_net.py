# https://discuss.pytorch.org/t/how-to-print-the-computed-gradient-values-for-a-network/34179/6
import logging

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data import examples
from utils import set_log
from torchinfo import summary

from nets.co import greedy_repair
class ToyModel(nn.Module):
    def __init__(self, blocks_num: int, block_size: int, layers_num: int):
        super().__init__()
        self.blocks_num = blocks_num
        self.block_size = block_size
        self.final_dim: int = blocks_num * block_size
        self.layers_num = layers_num
        self.final_hidden_layer_dim = (int(self.final_dim) // (2 ** self.layers_num)) ** 2
        self.model = self.__create__()

        # self.summer = torch.on-----------es(block_size)
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
        l = torch.Tensor([0]).cuda()
        weights = torch.flatten(weights, start_dim=1)
        for ind, row in enumerate(weights.flatten(start_dim=1)):
            one_block_loss =  torch.matmul(torch.matmul(row, collision_matrix[ind, 0]), row)
            #logging.info(f"one block loss: {one_block_loss}")
            l += one_block_loss

        return l

    def opt(self, blocks: torch.Tensor,i:int) -> np.ndarray:

        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # data_batch = create_block_matrix(batch_size=batch_size, blocks_num=blocks_num, block_size=block_size,
        #                                  epsilon=epsilon, seed=seed)
        optimizer.zero_grad()
        output = self(blocks)
        loss = self.loss_fn(collision_matrix=blocks, weights=output)
        # plot_triple(data, labels, pred)

        # Backpropagation
        loss.backward()
        # print(loss.item())
        optimizer.step()
        if i%50 == 0:
            logging.info(f"ind: {i}, {loss.item()}")
        return output.squeeze().detach().cpu().numpy()


def get_batch_with_sol(blocks_num: int, block_size: int,
                       epsilon: float,
                       batch_size: int, diagonal_blocks: bool, seed:bool) -> torch.Tensor:
    blocks = examples.create_random_batch(blocks_num=blocks_num, block_size=block_size,
                                          epsilon=epsilon,
                                          batch_size=10, diagonal_blocks=True, seed=seed)
    for ind, block in enumerate(blocks):
        blocks[ind] = examples.add_sol_to_data(block, blocks_num=blocks_num, block_size=block_size, sol=1.0)
    return torch.Tensor(np.expand_dims(blocks, 1)).cuda()

def proccess_res(reses:np.ndarray,blocks:torch.Tensor):
    blocks = blocks.cpu().numpy().squeeze()
    for ind,block in enumerate(blocks):
        res = reses[ind]
        res = examples.remove_collisions(block,res)
        rm_val = res.sum()
        res = greedy_repair(block,res)
        repaired_val = res.sum()
        logging.info(res.sum())
        greedy_val = greedy_repair(block,np.zeros_like(res)).sum()
        logging.info(f"rm_val: {rm_val}, repaired_val: {repaired_val}, greedy_val: {greedy_val}")




if __name__ == "__main__":
    set_log()
    blocks_num = 30
    block_size = 10
    batch_size = 1
    epsilon = 0.1
    seed = 1
    # collision_matrix = examples.create_block_matrix(blocks_num=blocks_num, block_size=block_size,
    #                                                 epsilon=0.1,
    #                                                 seed=1).squeeze()
    model = ToyModel(blocks_num=blocks_num, block_size=block_size, layers_num=3).float().cuda()
    for i in range(10000):
        blocks = get_batch_with_sol(blocks_num=blocks_num, block_size=block_size,
                                    epsilon=epsilon,
                                    batch_size=10, diagonal_blocks=True, seed=False)
        old_blocks = blocks.detach().cpu().numpy().copy()
        # summary(model, input_size=(4, 1, block_size * blocks_num, block_size * blocks_num))
        reses = model.opt(blocks,i)
    proccess_res(reses=reses,blocks=blocks)



