import logging
from typing import Set, Tuple
import itertools

import numpy as np
import torch
import matplotlib.pyplot as plt

#from nets import co


def create_block_matrix(batch_size: int, blocks_num: int, block_size: int, epsilon: float,
                        seed: int = -1) -> np.ndarray:
    l = blocks_num * block_size
    if seed >= 0:
        np.random.seed(seed)
    m = (np.random.rand(batch_size, 1, l, l) < epsilon / 2.0)
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


def find_colissions(collision_matrix: np.ndarray, paths_dist: np.ndarray) -> Set[Tuple[int, int]]:
    existing_blocks = np.where((paths_dist > 0.01).any(axis=1))[0]

    rs, cs = np.where(collision_matrix > 0)
    coordinates = list(zip(rs, cs))
    ps = np.arange(paths_dist.shape[0])*paths_dist.shape[1]+np.argmax(paths_dist,axis=1)
    mask =np.isin(ps//10,existing_blocks)
    ps =ps[mask]
    pairs = list(itertools.combinations(ps, 2))
    return set(pairs) & set(coordinates)


def count_collisions(collision_matrix: np.ndarray, paths_dist: np.ndarray) -> int:
    s = (paths_dist.flatten() > 0.5).astype(float)
    return int(s @ collision_matrix @ s)


