import os.path
import sys

sys.path.append(".")
import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt

import glob
from pathlib import Path
from typing import Tuple, List

import logging
from datetime import datetime, timedelta

import cvxpy as cp
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 300)

from nets import toy_net, co, direct_softmax_opt as ds, smart_greedy
from data import examples
from utils import set_log


def lp(blocks_num: int, block_size: int, epsilon: float):
    block = examples.create_block_matrix(blocks_num=blocks_num, block_size=block_size, epsilon=epsilon,
                                         seed=-1, diagonal_blocks=True)
    block = examples.add_sol_to_data(block=block, blocks_num=blocks_num, block_size=block_size)
    rs, cs = np.where(block == 1)
    v = cp.Variable(len(block))
    M = np.zeros((len(rs), len(block)))
    M[range(len(cs)), cs] = 1
    M[range(len(cs)), rs] = 1
    CL = np.zeros((blocks_num, blocks_num * block_size))
    for i in range(blocks_num):
        CL[i, i * block_size:(i + 1) * block_size] = 1
    constaints = [v >= 0, v <= 1, M @ v <= 1, CL @ v == 1]
    prob = cp.Problem(cp.Maximize(cp.sum(v)), constaints)
    q = prob.solve()
    logging.info(v)


if __name__ == "__main__":
    set_log()
    blocks_num = 100
    block_size = 20
    epsilon = 0.1
    lp(blocks_num, block_size, epsilon)
