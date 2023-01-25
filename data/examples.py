import logging
from typing import Set, Tuple, List
import itertools

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# from nets import co


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


def get_block_from_file(df: pd.DataFrame,vec_size:int) -> List[np.ndarray]:
    ms = []
    for id in range(len(df)//2):
        rs = df.iloc[2 * id, 5:].values
        cs = df.iloc[2 * id + 1, 5:].values
        block = np.ones((vec_size, vec_size))
        block[rs, cs] = 0
        block[np.eye(block.shape[0]).astype(bool)] = 0
        #min_eig = np.linalg.eig(block)[0].min()
        #block[np.eye(block.shape[0]).astype(bool)] = np.abs(min_eig)

        ms.append(block)
    return ms


def remove_collisions(collision_matrix: np.ndarray, selects: np.ndarray) -> np.ndarray:
    selects = selects.copy()
    block_size = selects.shape[1]
    collisions = find_collisions(collision_matrix, selects)
    collisions_0 = {x[0] for x in collisions}
    collisions_1 = {x[1] for x in collisions}
    # collisions = collisions_0 if len(collisions_0) < len(collisions_1) else collisions_1
    collisions = collisions_0.union(collisions_1)
    collisions = list(collisions)
    collisions.sort()
    logging.info(f"collsions_num: {len(collisions)}")

    rs = []
    cs = []
    for c in collisions:
        rs.append(c // block_size)
        cs.append(c % block_size)
    selects[list(rs), list(cs)] = 0
    return selects


def find_collisions(collision_matrix: np.ndarray, paths_dist: np.ndarray) -> Set[Tuple[int, int]]:
    existing_blocks = np.where((paths_dist > 0.01).any(axis=1))[0]

    rs, cs = np.where(collision_matrix > 0)
    coordinates = list(zip(rs, cs))
    ps = np.arange(paths_dist.shape[0]) * paths_dist.shape[1] + np.argmax(paths_dist, axis=1)
    mask = np.isin(ps // 10, existing_blocks)
    ps = ps[mask]
    pairs = list(itertools.combinations(ps, 2))
    return set(pairs) & set(coordinates)


def count_collisions(collision_matrix: np.ndarray, paths_dist: np.ndarray) -> int:
    s = (paths_dist.flatten() > 0.5).astype(float)
    return int(s @ collision_matrix @ s)
