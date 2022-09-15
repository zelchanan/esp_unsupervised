import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_block_matrix(batch_size: int, blocks_num: int, block_size: int, epsilon: float,
                        seed: int = -1) -> np.ndarray:
    l = blocks_num * block_size
    if seed >= 0:
        np.random.seed(seed)
    m = (np.random.rand(batch_size, 1, l, l) < epsilon/2.0)
    y, x = np.meshgrid(range(l), range(l))
    mask = y // block_size == x // block_size
    mask = np.expand_dims(mask, [0, 1])
    mask = np.repeat(mask, batch_size, axis=0)
    # logging.info(f"blocks: {m.shape}, mask: {mask.shape}")
    m[mask] = True
    m = (m | np.transpose(m, [0, 1, 3, 2])).astype(int)
    m = m - np.eye(l)
    # logging.info(f"shape: {m.shape}")
    return m
