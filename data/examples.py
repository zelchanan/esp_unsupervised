import logging
from typing import Set, Tuple, List
import itertools

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')


# from nets import co

def create_block_matrix(blocks_num: int, block_size: int, epsilon: float,
                        seed: int = -1, diagonal_blocks: bool = True) -> np.ndarray:
    l = blocks_num * block_size
    if seed >= 0:
        np.random.seed(seed)
    m = (np.random.rand(l, l) < epsilon / 2.0)
    mask = get_diagonal_blocks(blocks_num, block_size)

    m = (m | np.transpose(m)).astype(int)
    if diagonal_blocks:
        m[mask] = 1
        m = m - np.eye(l)
    else:
        m[mask] = 0

    return m


def create_random_batch(blocks_num: int, block_size: int, epsilon: float, batch_size: int, sol: bool,
                        seed: int = -1, diagonal_blocks: bool = True) -> np.ndarray:
    blocks = []
    for i in range(batch_size):
        block = create_block_matrix(blocks_num=blocks_num, block_size=block_size, epsilon=epsilon,
                                    seed=seed, diagonal_blocks=diagonal_blocks)
        if sol:
            block = add_sol_to_data(block=block, blocks_num=blocks_num, block_size=block_size)
        blocks.append(block)
    return np.stack(blocks, axis=0)


def get_diagonal_blocks(blocks_num: int, block_size: int) -> np.ndarray:
    l = blocks_num * block_size
    y, x = np.meshgrid(range(l), range(l))
    mask = y // block_size == x // block_size
    # mask = np.expand_dims(mask, [0, 1])
    return mask


def get_blocks_from_df(df: pd.DataFrame, vec_size: int) -> Tuple[List[np.ndarray], List[int]]:
    errs = df.isnull().sum().sum()
    if errs != 0:
        logging.info(f"Errs error: {errs} ")
        df[df.isnull()] = 0
        df = df.astype(int)

    blocks = []
    sizes = []
    for id in range(len(df) // 2):
        orig_size = df.iloc[2 * id, 3]
        rs = df.iloc[2 * id, 4:].values
        cs = df.iloc[2 * id + 1, 4:].values
        block = np.ones((vec_size, vec_size))
        block[rs, cs] = 0
        block[np.eye(block.shape[0]).astype(bool)] = 0
        # min_eig = np.linalg.eig(block)[0].min()
        # block[np.eye(block.shape[0]).astype(bool)] = np.abs(min_eig)

        blocks.append(block)
        sizes.append(orig_size)

    return blocks, sizes


def get_blocks_from_raw(path: str, vec_size: int, block_size: int, diagonal_blocks: bool = True) -> Tuple[
    np.ndarray, List[int]]:
    f = open(path)
    lines = []
    for line in f:
        lines.append(pd.Series(line.split(",")).astype(int))
    df = pd.concat(lines, axis=1).T

    blocks, sizes = get_blocks_from_df(df, vec_size)
    blocks = np.array(blocks)
    if not diagonal_blocks:
        # block_size = blocks[0, 0, :].sum() + 1
        blocks_num = len(blocks[0]) // block_size
        mask = get_diagonal_blocks(blocks_num, block_size)
        mask = np.expand_dims(mask, 0)
        mask = np.repeat(mask, len(blocks), axis=0)
        logging.info(f"{mask.shape}, {blocks.shape}")

        blocks[mask] = 0
    return blocks, sizes


def add_sol_to_data(block: np.ndarray, blocks_num: int, block_size: int) -> np.ndarray:
    selects = np.zeros((blocks_num, block_size))
    rs = range(blocks_num)
    cs = np.random.randint(0, block_size, blocks_num)
    selects[rs, cs] = 1
    mask = np.outer(selects.flatten(), selects.flatten()) == 1
    block[mask] = 0
    return block


def remove_collisions(collision_matrix: np.ndarray, selects: np.ndarray) -> np.ndarray:
    csol = selects.copy()
    collisions_num = csol.flatten() @ collision_matrix @ csol.flatten()
    # logging.info(selects.shape)
    while collisions_num > 0:
        idxmax = ((csol.flatten() @ collision_matrix) * (csol.flatten())).argmax() // selects.shape[1]
        csol[idxmax, :] = 0
        collisions_num = csol.flatten() @ collision_matrix @ csol.flatten()
    return csol
    # selects = selects.copy()
    # block_size = selects.shape[1]
    # collisions = find_collisions(collision_matrix, selects)
    # collisions_0 = {x[0] for x in collisions}
    # collisions_1 = {x[1] for x in collisions}
    # # collisions = collisions_0 if len(collisions_0) < len(collisions_1) else collisions_1
    # collisions = collisions_0.union(collisions_1)
    # collisions = list(collisions)
    # collisions.sort()
    # logging.info(f"collsions_num: {len(collisions)}")
    #
    # rs = []
    # cs = []
    # for c in collisions:
    #     rs.append(c // block_size)
    #     cs.append(c % block_size)
    # selects[list(rs), list(cs)] = 0
    # return selects


def find_collisions(collision_matrix: np.ndarray, paths_dist: np.ndarray) -> Set[Tuple[int, int]]:
    existing_blocks = np.where((paths_dist > 0.01).any(axis=1))[0]

    rs, cs = np.where(collision_matrix > 0)
    coordinates = list(zip(rs, cs))
    ps = np.arange(paths_dist.shape[0]) * paths_dist.shape[1] + np.argmax(paths_dist, axis=1)
    mask = np.isin(ps // 10, existing_blocks)
    ps = ps[mask]
    pairs = list(itertools.combinations(ps, 2))
    return set(pairs) & set(coordinates)


def get_init_selects(blocks_num: int, block_size: int) -> np.ndarray:
    selects = np.zeros((blocks_num, block_size))
    inds = np.random.randint(low=0, high=block_size, size=blocks_num)
    selects[list(range(blocks_num)), inds] = 1
    return selects


def count_collisions(collision_matrix: np.ndarray, paths_dist: np.ndarray) -> int:
    s = (paths_dist.flatten() > 0.5).astype(float)
    return int(s @ collision_matrix @ s)


if __name__ == "__main__":
    blocks_num = 5
    block_size = 3
    epsilon = 0.1
    blocks = create_random_batch(blocks_num, block_size, epsilon, batch_size=10, sol=True)
    # add_sol_to_data(blocks, blocks_num, block_size)
    # p = "data/yael_dataset2/gnn_k_100_m_20_e_10_full_sol.csv"
    # blocks, sizes = get_blocks_from_raw(p, 2000)
    # f, ax = plt.subplots()
    # ax.imshow(blocks[0])
    # plt.pause(100)
