import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
from data.examples import create_block_matrix
from utils import set_log


def get_real_val(r_weights: torch.Tensor, block: torch.Tensor) -> float:
    r_weights = r_weights.detach().numpy()
    block = block.numpy().squeeze()
    inds = np.argmax(r_weights, axis=1).flatten()
    selects = np.zeros_like(r_weights)
    selects[:,inds]=1
    selects = selects.flatten()
    return np.dot(selects, np.dot(block, selects))


def optimize(weights: torch.Tensor, block: torch.Tensor):
    optimizer = torch.optim.Adam([weights], lr=1e-2)
    ones = torch.ones(300).float()
    for i in range(10000):
        optimizer.zero_grad()
        r_weights = weights.view(30, 10)
        negative_loss = torch.nn.ReLU()(-r_weights).sum()
        distribution_loss = torch.pow(r_weights.sum(axis=1) - 1, 2).sum()
        collision_loss = torch.mean(torch.mul(torch.matmul(weights, block.squeeze()), ones))
        bl = torch.mean(torch.mul(torch.matmul(ones, block.squeeze()), ones))
        loss = negative_loss + distribution_loss + collision_loss
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            logging.info(f"real_val: {get_real_val(r_weights, block)}")
            logging.info(
                f"bl: {bl}, negative_loss: {negative_loss.item()}, distribution loss: {distribution_loss.item()}, collision loss: {collision_loss.item()}")


if __name__ == "__main__":
    set_log()
    weights = torch.zeros(300, requires_grad=True)
    #ones = torch.ones(300).float() / 10
    block = create_block_matrix(batch_size=1, blocks_num=30, block_size=10, epsilon=0.1, seed=0)
    block = torch.tensor(block).float()
    optimize(weights, block)
