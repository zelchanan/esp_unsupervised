# https://discuss.pytorch.org/t/how-to-print-the-computed-gradient-values-for-a-network/34179/6
import logging

import numpy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

from data import examples
from utils import set_log
from torch_geometric.nn import GCNConv, EdgeConv
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything
from gnn.convert_to_graph import convert_to_graph

from torchinfo import summary

from nets.co import greedy_repair
from data.examples import remove_collisions


class GraphModel(nn.Module):
    def __init__(self, blocks_num: int, block_size: int, layers_num: int, init_dim: int):
        seed_everything(1)
        super().__init__()
        self.blocks_num = blocks_num
        self.block_size = block_size
        self.layers_num = layers_num
        self.init_dim = init_dim
        self.__create__()

        # self.summer = torch.on-----------es(block_size)
        # self.selector = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, self.final_dim), padding="same"), nn.ReLU(),

    def __create__(self):
        f = GCNConv
        self.convs = torch.nn.ModuleList()
        self.convs.append(f(self.init_dim, 16))
        for i in range(self.layers_num):
            self.convs.append(f(16, 16))
        self.last_conv = f(16, 1)

    def forward(self, graph: Data) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index
        for ind, conv in enumerate(self.convs):
            x = self.convs[ind](x, edge_index)
            x = torch.nn.ReLU()(x)
        x = self.last_conv(x, edge_index)
        reshaped = torch.reshape(x, (-1, self.blocks_num, self.block_size))
        return torch.nn.Softmax(dim=2)(reshaped)

    # @staticmethod
    # def get_amp(x:torch.Tensor):
    #     return

    # def __create__(self):
    #     self.conv1 = GCNConv(1, 16)
    #     self.conv2 = GCNConv(16, 16)
    #     self.conv3 = GCNConv(16, 1)
    #
    # def forward(self, graph: Data) -> torch.Tensor:
    #     x, edge_index = graph.x, graph.edge_index
    #     x = self.conv1(x, edge_index)
    #     x = torch.nn.ReLU()(x)
    #     x = self.conv2(x, edge_index)
    #     x = torch.nn.ReLU()(x)
    #     x = self.conv3(x, edge_index)
    #     reshaped = torch.reshape(x, (-1, self.blocks_num, self.block_size))
    #     return torch.nn.Softmax(dim=2)(reshaped)

    def loss_fn(self, collision_matrix: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        s = weights.flatten()
        return torch.matmul(torch.matmul(s, collision_matrix), s)

    def opt(self, graph_block: Data, block: torch.Tensor, optimizer: torch.optim.Optimizer, i: int) -> np.ndarray:
        # data_batch = create_block_matrix(batch_size=batch_size, blocks_num=blocks_num, block_size=block_size,
        #                                  epsilon=epsilon, seed=seed)
        optimizer.zero_grad()
        output = self(graph_block)
        loss = self.loss_fn(collision_matrix=block, weights=output)
        # plot_triple(data, labels, pred)

        # Backpropagation
        loss.backward()
        # print(loss.item())
        optimizer.step()
        if i % 50 == 0:
            logging.info(f"ind: {i}, loss: {loss.item()}")
        return output.squeeze().detach().cpu().numpy()


def get_batch_with_sol(blocks_num: int, block_size: int,
                       epsilon: float,
                       batch_size: int, diagonal_blocks: bool, seed: bool) -> torch.Tensor:
    blocks = examples.create_random_batch(blocks_num=blocks_num, block_size=block_size,
                                          epsilon=epsilon,
                                          batch_size=10, diagonal_blocks=True, seed=seed)
    for ind, block in enumerate(blocks):
        blocks[ind] = examples.add_sol_to_data(block, blocks_num=blocks_num, block_size=block_size, sol=1.0)
    return torch.Tensor(np.expand_dims(blocks, 1)).cuda()


def proccess_res(reses: np.ndarray, blocks: torch.Tensor):
    blocks = blocks.cpu().numpy().squeeze()
    for ind, block in enumerate(blocks):
        res = reses[ind]
        res = examples.remove_collisions(block, res)
        rm_val = res.sum()
        res = greedy_repair(block, res)
        repaired_val = res.sum()
        logging.info(res.sum())
        greedy_val = greedy_repair(block, np.zeros_like(res)).sum()
        logging.info(f"rm_val: {rm_val}, repaired_val: {repaired_val}, greedy_val: {greedy_val}")


if __name__ == "__main__":

    set_log()
    blocks_num = 30
    block_size = 10
    batch_size = 1
    epsilon = 0.1
    seed = 1
    init_dim = 5
    # collision_matrix = examples.create_block_matrix(blocks_num=blocks_num, block_size=block_size,
    #                                                 epsilon=0.1,
    #                                                 seed=1).squeeze()
    model = GraphModel(blocks_num=blocks_num, block_size=block_size, layers_num=3, init_dim=init_dim).float()
    block = examples.create_block_matrix(blocks_num=blocks_num, block_size=block_size, epsilon=epsilon, seed=-1,
                                         diagonal_blocks=True)
    block = examples.add_sol_to_data(block=block, blocks_num=blocks_num, block_size=block_size,
                                     sol=True)

    writer = SummaryWriter()
    learning_rate = 1e-3
    l = model.parameters()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(10000):
        graph_block = convert_to_graph(block, d0=init_dim)
        reses = model.opt(graph_block=graph_block, block=torch.Tensor(block), optimizer=optimizer, i=i)
        grad1 = torch.abs(model.convs[0].lin.weight.grad).mean()
        grad2 = torch.abs(model.convs[1].lin.weight.grad).mean()
        grad_middle = torch.abs(model.convs[len(model.convs) // 2].lin.weight.grad).mean()
        grad_last = torch.abs(model.last_conv.lin.weight.grad).mean()
        writer.add_scalar("first_grad", grad1, i)
        writer.add_scalar("second_grad", grad2, i)
        writer.add_scalar("middle_grad", grad_middle, i)
        writer.add_scalar("last_grad", grad_last, i)

        removed_reses = examples.remove_collisions(block, np.round(reses))
        greedy_res = greedy_repair(block, removed_reses)
        if i % 50 == 0:
            logging.info(
                f"removed_reses: {removed_reses.sum()} ,greedy_reses: {greedy_res.sum()}\n"
                f"first layer grad: {grad1:.6f},  second layer grad2: {grad2:.6f},"
                f" middle_layer_grad: {grad_middle:.6f}, last layer grad: {grad_last:.6f}")
    # proccess_res(reses=reses, blocks=blocks)
