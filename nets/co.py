import logging
from typing import Dict

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from data import examples
from utils import set_log


def get_selects(r_weights: np.ndarray) -> np.ndarray:
    maxes = np.argmax(r_weights, axis=1).flatten()
    selects = np.zeros_like(r_weights)
    block_inds = range(r_weights.shape[0])
    selects[block_inds, maxes] = 1
    return selects


def get_real_val(r_weights: torch.Tensor, block: torch.Tensor) -> float:
    r_weights = r_weights.detach().numpy()
    block = block.numpy().squeeze()
    selects = get_selects(r_weights)
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
        collision_loss = torch.matmul(torch.matmul(weights, block.squeeze()), weights) ** 2
        bl = torch.mul(torch.matmul(one_weights, block.squeeze()), one_weights)
        loss = negative_loss + distribution_loss + 0.1 * collision_loss
        loss.backward()
        optimizer.step()
        # logging.info()
        real_val = get_real_val(dist_weights, block)
        # logging.info(f"real_val: {get_real_val(dist_weights, block)}")
        if real_val not in res_dict:
            res_dict[real_val] = dist_weights.detach().numpy().copy()
            logging.info(
                f"ind:  {i}, real_val: {real_val}, loss: {loss}, negative_loss: {negative_loss}, dist_loss: {distribution_loss}, colission loss: {0.1 * collision_loss}")
            if real_val == 0:
                return res_dict
    return res_dict
    # logging.info(
    #     f"bl: {bl}, negative_loss: {negative_loss.item()}, distribution loss: {distribution_loss.item()}, collision loss: {collision_loss.item()}")


def greedy_repair(collision_matrix: np.ndarray, selects: np.ndarray) -> np.ndarray:
    blocks_num, blocks_size = selects.shape
    collisions = examples.find_colissions(collision_matrix, selects)
    collisions = [x[0] for x in collisions] + [x[1] for x in collisions]
    collisions = list(set(collisions))
    collisions.sort()
    logging.info(len(collisions))

    rs = []
    cs = []
    for c in collisions:
        rs.append(c // block_size)
        cs.append(c % block_size)
    selects[list(rs), list(cs)] = 0
    logging.info(f"start: {selects.sum()}")
    missing_blocks = np.where((selects == 0).all(axis=1))[0]
    #occupied_cols = np.where(selects.flatten() == 1)[0]

    for ind,m in enumerate(missing_blocks):
        import matplotlib.pyplot as plt
        f,ax = plt.subplots()
        ax.imshow(selects.astype(int))
        ax.set_title(f"{m}")
        submatrix = collision_matrix[m * blocks_size:(m + 1) * block_size, :].astype(bool)
        bool_selects = selects.flatten().astype(bool)
        logging.info(f"#: {bool_selects.sum()}")
        candidates = ((submatrix & bool_selects) == False).all(axis=1)
        #logging.info(f"block: {m}, candidates: {candidates}, sum: {candidates.sum()}")
        if candidates.any():
            selected_for_block = np.random.choice(np.where(candidates)[0])
            selects[m, selected_for_block] = 1
        #logging.info(f"collisions num: {examples.count_collisions(block,selects)}")
    #logging.info(f"new_collision: {examples.find_colissions(block,selects)}")
    logging.info(f"end: {selects.sum()}")

    plt.pause(1000)
    return selects


if __name__ == "__main__":
    set_log()
    blocks_num = 30
    block_size = 10
    batch_size = 16
    epsilon = 0.1
    seed = -1
    block = examples.create_block_matrix(batch_size=1, blocks_num=blocks_num, block_size=block_size, epsilon=0.1,
                                         seed=1)[0].squeeze()
    #block = torch.tensor(block).float()

    res_dict = optimize(torch.tensor(block).float(), blocks_num=blocks_num, block_size=block_size)
    key = min(list(res_dict.keys()))
    collisions = examples.find_colissions(collision_matrix=block, paths_dist=res_dict[key])
    print(f"collisions: {collisions}")
    #for i in range(10):
    if collisions:
        greedy_repair(collision_matrix=block, selects=get_selects(res_dict[key]))
