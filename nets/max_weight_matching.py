import sys
sys.path.append(".")

import  itertools

import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

import networkx as nx
from data import examples


def find_match(blocks_num: int, block_size: int, epsilon: float, sol: bool, diagonal_blocks: bool,
               seed: int = -1) ->np.ndarray:
    edges = []
    block = examples.create_block_matrix(blocks_num=blocks_num, block_size=block_size, epsilon=epsilon,
                                seed=seed, diagonal_blocks=diagonal_blocks)
    rs, cs = np.where(block == 0)
    l = list(zip(rs, cs))
    l = np.array(l)
    mask = l[:, 0] // 10 == l[:, 1] // 10
    p = np.ones_like(mask).astype(int)
    p[mask] = -10
    edges = np.vstack([l.T, p]).T
    g = nx.Graph()
    g.add_weighted_edges_from(edges)
    nx.max_weight_matching()
    # for block_num in range(blocks_num):
    #     edges += list(itertools.combinations(range(block_num*block_size,(block_num+1)*block_size),2))
    return block



if __name__ == "__main__":
    blocks_num = 30
    block_size= 10
    epsilon=0.1
    sol = False
    diagonal_blocks = False
    seed = -1
    find_match(blocks_num=blocks_num,block_size=block_size,epsilon=epsilon,sol=sol,diagonal_blocks=diagonal_blocks)




