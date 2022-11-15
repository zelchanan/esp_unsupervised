import logging
from typing import Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
from data import examples
from utils import set_log


def get_real_val(r_weights: torch.Tensor, block: torch.Tensor) -> float:
    r_weights = r_weights.detach().numpy()
    block = block.numpy().squeeze()
    maxes = np.argmax(r_weights, axis=1).flatten()
    selects = np.zeros_like(r_weights)
    block_inds = range(r_weights.shape[0])
    selects[block_inds, maxes] = 1
    selects = selects.flatten()
    return np.dot(selects, np.dot(block, selects))


def optimize(block: torch.Tensor, blocks_num: int, block_size: int) -> Dict[float, np.ndarray]:
    # weights_num = block.shape[-1]
    weights = torch.zeros(blocks_num * block_size, requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=1e-2)
    one_weights = torch.ones(blocks_num * block_size) / block_size
    res_dict = {}
    for i in range(10000):
        optimizer.zero_grad()
        # r_weights = weights.view(block.shape[-2], block.shape[-1])
        negative_loss = torch.nn.ReLU()(-weights).sum()
        dist_weights = torch.reshape(weights, (blocks_num, block_size))
        distribution_loss = torch.pow(dist_weights.sum(axis=1) - 1, 2).sum()
        collision_loss = torch.matmul(torch.matmul(weights, block.squeeze()), weights)**2
        bl = torch.mul(torch.matmul(one_weights, block.squeeze()), one_weights)
        loss = negative_loss + distribution_loss + 0.1 * collision_loss
        loss.backward()
        optimizer.step()
        if i % 1 == 0:
            # logging.info()
            real_val = get_real_val(dist_weights, block)
            # logging.info(f"real_val: {get_real_val(dist_weights, block)}")
            if real_val not in res_dict:
                res_dict[real_val] = dist_weights.detach().numpy()
                logging.info(
                    f"ind:  {i}, real_val: {real_val}, loss: {loss}, negative_loss: {negative_loss}, dist_loss: {distribution_loss}, colission loss: {0.1 * collision_loss}")
    return res_dict
    # logging.info(
    #     f"bl: {bl}, negative_loss: {negative_loss.item()}, distribution loss: {distribution_loss.item()}, collision loss: {collision_loss.item()}")


if __name__ == "__main__":
    set_log()
    blocks_num = 30
    block_size = 10
    batch_size = 16
    epsilon = 0.1
    seed = -1

    weights = torch.zeros(blocks_num * block_size, requires_grad=True)

    # ones = torch.ones(300).float() / 10
    block = examples.create_block_matrix(batch_size=1, blocks_num=blocks_num, block_size=block_size, epsilon=0.1,
                                seed=-1).squeeze()
    block = torch.tensor(block).float()
    res_dict = optimize(block, blocks_num=blocks_num, block_size=block_size)
    examples.find_colissions(block.numpy(), res_dict[4])