import sys

import pandas as pd

sys.path.append(r"res")
from typing import Tuple

import numpy as np
import logging

import torch
from torch import nn

from data import examples
from utils import set_log

def optim(block: np.ndarray, init_weight: np.ndarray) -> Tuple[float, np.ndarray]:
    weights = torch.Tensor(init_weight).requires_grad_()
    tblock = torch.Tensor(block)
    optimizer = torch.optim.Adam([weights], lr=1e-2)

    previous_loss = block.size
    for i in range(100000):
        sm_weights = torch.nn.Softmax(dim=-1)(weights).flatten()
        optimizer.zero_grad()
        collision_loss = torch.matmul(torch.matmul(sm_weights, tblock), sm_weights)
        # collision_loss = torch.matmul(torch.matmul(sm_weights, tapprox_block), sm_weights)

        collision_loss.backward(retain_graph=True)
        loss = collision_loss.item()
        optimizer.step()
        if (np.abs(loss - previous_loss) < 0.01) and (np.modf(loss)[0] < 0.01):
            return loss, sm_weights.detach().numpy().copy()
        previous_loss = loss
    return loss, sm_weights.detach().numpy().copy()

if __name__ == "__main__":
    set_log(level = logging.INFO)
    blocks_num = 30
    block_size = 10
    batch_size = 1
    epsilon = 0.1
    seed = -1
    block = examples.create_block_matrix(batch_size=1,blocks_num=blocks_num,block_size=block_size,epsilon=0.1)
    losses = []
    all_sm_weights = []

    for i in range(1000):
        init_weights = np.random.randn(blocks_num,block_size)
        loss, sm_weights = optim(block, init_weights)
        losses.append(int(loss))
        all_sm_weights.append(sm_weights)
        logging.info(f"i: {i}, loss: {loss}")
    print(pd.Series(losses).value_counts().sort_index())


