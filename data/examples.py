import sys
sys.path.append(".")
import logging
from typing import Set, Tuple, List, Optional
import itertools

import numpy as np
import pandas as pd
import matplotlib
from scipy import optimize
from scipy.optimize import milp
import matplotlib.pyplot as plt
from scipy.sparse import bsr_array

from utils import set_log

matplotlib.use('Qt5Agg')


# from nets import co

class CollisionsMatrix():
    def __init__(self, collision_matrix: np.ndarray, sizes: Optional[List[int]] = None):
        self.collision_matrix = collision_matrix
        self.total_size = len(self.collision_matrix)
        self.start_inds, self.end_inds = self.extract_blocks(sizes)
        self.sizes = self.end_inds - self.start_inds + 1
        self.blocks_num: int = len(self.sizes)

    def extract_blocks(self, sizes: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        if sizes:
            cumsum = np.cumsum(sizes)
            end_inds = cumsum - 1
            start_inds = np.hstack([np.array([0]), cumsum[:-1]])
        else:
            sub_diag = np.diag(self.collision_matrix, -1)
            inds = np.where(sub_diag == 0)[0]
            start_inds = np.hstack([np.array([0]), inds + 1])
            end_inds = np.hstack([inds, np.array(len(self.collision_matrix) - 1)])
        return start_inds, end_inds

    def get_block_and_inner_ind(self, total_ind: int) -> Tuple[int, int]:
        m = (total_ind >= self.start_inds) & (total_ind <= self.end_inds)
        block = np.where(m)[0][0]
        inner_ind = total_ind - self.start_inds[block]
        return block, inner_ind


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


def create_random_batch(blocks_num: int, block_size: int, epsilon: float, batch_size: int,
                        seed: bool = False, diagonal_blocks: bool = True) -> np.ndarray:
    blocks = []
    for i in range(batch_size):
        seed_val = i + 1 if seed else -1

        block = create_block_matrix(blocks_num=blocks_num, block_size=block_size, epsilon=epsilon, seed=seed_val)
        blocks.append(block)
    return np.stack(blocks, axis=0)


def create_lexical_matrix(blocks_num: int, block_size: int, epsilon: float,
                          seed: int = -1) -> np.ndarray:
    block = create_block_matrix(blocks_num=blocks_num, block_size=block_size, epsilon=epsilon,
                                seed=seed, diagonal_blocks=False)
    block = add_priorities(block, block_size, blocks_num)
    return -block


def add_priorities(blocks: np.ndarray, block_size: int, blocks_num: int):
    for ind, block in enumerate(blocks):
        prices = np.random.rand(blocks_num * block_size, blocks_num * block_size)
        block = -block * prices
        mask = get_diagonal_blocks(blocks_num, block_size)
        block[mask] = -1
        block[np.diag(np.ones(blocks_num * block_size)).astype(bool)] = 1
        priorities_vec = np.zeros(blocks_num * block_size)
        group_size = blocks_num // 4
        for i in range(4):
            priorities_vec[i * group_size * block_size:(i + 1) * group_size * block_size] = i + 1
        priorities_vec[priorities_vec == 0] = 4
        priority_matrix = np.diag(priorities_vec)
        blocks[ind] = np.dot(priority_matrix, block)
    return -blocks


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


def add_sol_to_data(block: np.ndarray, blocks_num: int, block_size: int, sol: float) -> np.ndarray:
    diag_vals = np.diag(block).copy()
    diag_mask = np.eye(blocks_num * block_size).astype(bool)
    selects = np.zeros((blocks_num, block_size))
    num = int(blocks_num * sol)
    rs = np.random.choice(range(blocks_num), num, replace=False)
    cs = np.random.randint(0, block_size, num)
    # mask = (np.random.rand(blocks_num) < sol).astype(int)
    # cs *= mask

    selects[rs, cs] = 1
    mask = np.outer(selects.flatten(), selects.flatten()) == 1

    block[mask] = 0
    block[diag_mask] = diag_vals
    return block


def remove_collisions(collision_matrix: np.ndarray, selects: np.ndarray) -> np.ndarray:
    csol = selects.copy()
    # collisions_num = csol.flatten() @ collision_matrix @ csol.flatten()
    # logging.info(selects.shape)
    vals = ((csol.flatten() @ collision_matrix) * (csol.flatten())).copy()
    max_val = vals.max()
    idxmax = ((csol.flatten() @ collision_matrix) * (csol.flatten())).argmax() // selects.shape[1]
    while max_val > 0:
        # logging.info(f"max val: {max_val}")
        csol[idxmax, :] = 0
        # collisions_num = csol.flatten() @ collision_matrix @ csol.flatten()
        max_val = ((csol.flatten() @ collision_matrix) * (csol.flatten())).max()
        idxmax = ((csol.flatten() @ collision_matrix) * (csol.flatten())).argmax() // selects.shape[1]
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


def ilp():
    blocks_num = 20
    block_size = 10
    epsilon = 0.2
    blocks = create_random_batch(blocks_num, block_size, epsilon, batch_size=10)
    for ind, block in enumerate(blocks):
        blocks[ind] = add_sol_to_data(block=block, blocks_num=blocks_num, block_size=block_size,
                                               sol=True)

    cm = CollisionsMatrix(blocks[0], sizes=blocks_num * [block_size])
    num_of_vars = cm.total_size + cm.total_size * (cm.total_size - 1) // 2
    num_of_equations = cm.blocks_num + cm.total_size * (cm.total_size - 1)
    above_diagonal = np.array(
        [cm.collision_matrix[i, j] for i in range(cm.total_size) for j in range(i + 1, cm.total_size)])
    above_inds = [[i, j] for i in range(cm.total_size) for j in range(i + 1, cm.total_size)]
    integrality = np.ones(num_of_vars).astype(bool)
    # constaraints_matrix = np.zeros((num_of_equations, num_of_vars))
    rows = []
    cols = []
    vals = []
    for i in range(cm.blocks_num):
        block_size = cm.sizes[i]

        rows += [i] * block_size
        cols += range(cm.start_inds[i], cm.end_inds[i] + 1)
        vals += [1] * block_size
        # constaraints_matrix[i, cm.start_inds[i]:cm.end_inds[i] + 1] = 1
    for k in range(cm.blocks_num, num_of_equations - 1, 2):
        c, r = above_inds[(k - cm.blocks_num) // 2]
        block_ind, inner_ind = cm.get_block_and_inner_ind(c)
        # logging.info(f"k: {k}, c: {c}, r: {r}, block_ind: {block_ind}, inner_ind: {inner_ind}")
        first_block_ind = cm.start_inds[block_ind] + inner_ind
        block_ind, inner_ind = cm.get_block_and_inner_ind(r)
        second_block_ind = cm.start_inds[block_ind] + inner_ind
        collison_ind = (k - cm.blocks_num) // 2 + cm.total_size
        rows += [k, k, k + 1, k + 1]
        cols += [collison_ind, first_block_ind, collison_ind, second_block_ind]
        vals += [1, -1, 1, -1]
        # constaraints_matrix[k, (k - cm.blocks_num) // 2 + cm.total_size] = 1
        # constaraints_matrix[k, first_block_ind] = -1
        # constaraints_matrix[k + 1, (k - cm.blocks_num) // 2 + cm.total_size] = 1
        # constaraints_matrix[k + 1, second_block_ind] = -1
    constraints_matrix = bsr_array((vals, (rows, cols)), shape=(num_of_equations, num_of_vars))
    f, ax = plt.subplots()
    #ax.imshow(constraints_matrix.toarray())
    bounds = optimize.Bounds(0, 1)  # 0 <= x_i <= 1
    lb = np.zeros(num_of_equations)
    lb[:cm.blocks_num] = 1
    lb[cm.blocks_num:] = -np.inf
    ub = np.zeros(num_of_equations)
    ub[:cm.blocks_num] = 1
    constraints = optimize.LinearConstraint(constraints_matrix, lb=lb, ub=ub)
    # c = cm.collision_matrix.flatten()
    c = np.hstack([np.zeros(cm.total_size), above_diagonal]) - 1
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options = {"disp":True})
    f, axes = plt.subplots(2, 2)
    selected_vars = res.x[:cm.total_size].reshape((cm.blocks_num, cm.total_size // cm.blocks_num))
    axes[0, 0].imshow(selected_vars)
    axes[0, 1].imshow(np.outer(selected_vars.flatten(), selected_vars.flatten()))
    collision_vars = res.x[cm.total_size:]
    m = np.zeros((cm.total_size, cm.total_size))
    counter = 0
    for i in range(cm.total_size):
        for j in range(i + 1, cm.total_size):
            m[i, j] = collision_vars[counter]
            counter += 1
    axes[1, 0].imshow(m + m.T)
    axes[1, 1].imshow(cm.collision_matrix)
    logging.info(res.x)
    plt.pause(1000)


if __name__ == "__main__":
    set_log()
    ilp()

# create_lexical_matrix(blocks_num=blocks_num, block_size=block_size, epsilon=epsilon, sol=True, seed=1)
# add_sol_to_data(blocks, blocks_num, block_size)
# p = "data/yael_dataset2/gnn_k_100_m_20_e_10_full_sol.csv"
# blocks, sizes = get_blocks_from_raw(p, 2000)
# f, ax = plt.subplots()
# ax.imshow(blocks[0])
# plt.pause(100)
