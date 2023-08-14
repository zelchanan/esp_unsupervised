import os
import sys
from pathlib import Path
import pandas as pd

sys.path.append(r"res")
from typing import Tuple

import numpy as np
import logging

import torch
from torch import nn
import matplotlib.pyplot as plt

from data import examples
from utils import set_log
from nets import co


def optim(block: np.ndarray, init_weight: np.ndarray, priority: np.ndarray = np.empty(0), toplot=False) -> Tuple[
    float, np.ndarray]:
    weights = torch.Tensor(init_weight).requires_grad_()
    tblock = torch.Tensor(block)
    if len(priority):
        priority = torch.Tensor(priority)
    optimizer = torch.optim.Adam([weights], lr=1e-2)

    previous_loss = block.size

    p = Path("ds_movs")
    p.mkdir(exist_ok=True)
    fnames = list(p.glob("*.png"))
    for f in fnames:
        f.unlink()

    for i in range(100000):
        sm_weights = torch.nn.Softmax(dim=-1)(weights).flatten()
        optimizer.zero_grad()
        collision_loss = torch.matmul(torch.matmul(sm_weights, tblock), sm_weights)
        if len(priority):
            collision_loss = collision_loss + torch.inner(priority, sm_weights.flatten())
        # collision_loss = torch.matmul(torch.matmul(sm_weights, tapprox_block), sm_weights)

        if (i % 25 == 0) and toplot:
            dist_weights = sm_weights.detach().numpy().reshape(init_weight.shape)
            f, ax = plt.subplots(figsize=(3, 6))
            im = ax.imshow(dist_weights, vmin=0.0, vmax=1.0)
            plt.colorbar(im)
            plt.savefig(p / f"img_{str(i // 25).zfill(3)}")
        collision_loss.backward(retain_graph=True)
        loss = collision_loss.item()
        # logging.info(f"ind: {i}, loss: {loss}")
        optimizer.step()
        if (np.abs(loss - previous_loss) < 0.01):
            output_fname = f"{p.name}{os.path.sep}out.mp4"
            pat = f"{p.name}{os.path.sep}img_%03d.png"
            # cmd = f"ffmpeg -y -framerate 2 -i {pat} -c:v libx264 -pix_fmt yuv420p {output_fname}"
            # os.system(cmd)
            return loss, sm_weights.detach().numpy().copy()
        previous_loss = loss

    return loss, sm_weights.detach().numpy().copy()


if __name__ == "__main__":
    set_log(level=logging.INFO)
    blocks_num = 30
    block_size = 10
    batch_size = 1
    epsilon = 0.1
    seed = -1
    block = examples.create_block_matrix(batch_size=1, blocks_num=blocks_num, block_size=block_size, epsilon=0.1)
    losses = []
    all_sm_weights = []

    for i in range(1000):
        init_weights = np.random.randn(blocks_num, block_size)
        loss, sm_weights = optim(block, init_weights)
        losses.append(int(loss))
        all_sm_weights.append(sm_weights)
        logging.info(f"i: {i}, loss: {loss}")
    print(pd.Series(losses).value_counts().sort_index())
