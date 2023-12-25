import sys

sys.path.append(".")

import itertools

import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

import networkx as nx
from networkx.algorithms.approximation import maximum_independent_set
from data import examples


def find_match(blocks_num: int, block_size: int, epsilon: float, sol: bool, diagonal_blocks: bool,
               seed: int = -1) -> np.ndarray:
    edges = []
    block = examples.create_block_matrix(blocks_num=blocks_num, block_size=block_size, epsilon=epsilon,
                                         seed=seed, diagonal_blocks=diagonal_blocks)
    block  = examples.add_sol_to_data(block=block, block_size=block_size, blocks_num=blocks_num, sol=1)
    rs, cs = np.where(block == 1)
    l = list(zip(rs, cs))
    edges = np.array(l)
    # mask = l[:, 0] // 10 == l[:, 1] // 10
    # p = np.ones_like(mask).astype(int)
    # p[mask] = -10
    # edges = np.vstack([l.T, p]).T
    g = nx.Graph()
    g.add_edges_from(edges)
    # nx.max_weight_matching()
    res = maximum_independent_set(g)
    # for block_num in range(blocks_num):
    #     edges += list(itertools.combinations(range(block_num*block_size,(block_num+1)*block_size),2))
    return block


if __name__ == "__main__":
    blocks_num = 100
    block_size = 20
    epsilon = 0.1
    sol = False
    diagonal_blocks = True
    seed = -1
    find_match(blocks_num=blocks_num, block_size=block_size, epsilon=epsilon, sol=sol, diagonal_blocks=diagonal_blocks)
