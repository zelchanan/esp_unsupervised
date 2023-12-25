import matplotlib.pyplot as plt
import numpy as np
import torch

from data import examples
from torch_geometric.data import Data


def convert_to_graph(block: np.ndarray, d0: int) -> Data:
    # block = examples.create_block_matrix(blocks_num=blocks_num, block_size=block_size, epsilon=0.1,
    #                                      diagonal_blocks=True)
    rs, cs = np.where(block == 1)
    edge_index = torch.tensor(np.array([rs, cs]), dtype=torch.long)
    x = torch.Tensor(np.ones((block.shape[0], d0)))
    x.normal_()
    return Data(x=x, edge_index=edge_index)


if __name__ == "__main__":
    blocks_num = 30
    block_size = 10
    d0=5
    block = examples.create_block_matrix(blocks_num=blocks_num, block_size=block_size, epsilon=0.1,
                                         diagonal_blocks=True)
    convert_to_graph(block, d0 = d0)
