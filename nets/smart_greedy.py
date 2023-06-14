import sys

sys.path.append(r"res")

import numpy as np
import logging

from data import examples
from utils import set_log


class SmartGreedy:
    def __init__(self, height: int, width: int, collision_matrix: np.ndarray, min_required_sol: int = -1):
        if min_required_sol == -1:
            self.min_required_sol = height
        else:
            self.min_required_sol = min_required_sol
        self.max_sol_size: int = 0
        self.max_selects: np.ndarray = np.zeros((height, width))

        self.iteration: int = 0
        self.scanned: float = 0.0
        self.height = height
        self.width = width
        self.collision_matrix = collision_matrix

    def smart_greedy(self, selects: np.ndarray, row: int):
        self.iteration = self.iteration + 1

        if self.iteration % 1000000 == 0:
            logging.info(
                f"iteration: {self.iteration//1000000}M, scanned: {self.scanned}, max_sol_size: {self.max_sol_size}, row{row}, no recursion: {self.get_pairs_of_inds(selects)}")
        if ((selects.sum() + self.height - row) <= self.min_required_sol - 1) or (self.max_sol_size >= self.height):
            self.scanned += self.width ** -(row + 1)
        else:
            logging.debug(f"row: {row}, inside recursion")

            for col in range(self.width):
                tmp_selects = selects.copy()

                if np.dot(self.collision_matrix[row * self.width + col], selects.flatten()) == 0:
                    tmp_selects[row, col] = 1

                    self.max_sol_size = max(self.max_sol_size, tmp_selects.sum())
                    if tmp_selects.sum() == self.max_sol_size:
                        self.max_selects = tmp_selects
                    logging.debug(
                        f"iteration: {self.iteration}, row: {row}, col: {col}, max_sol_size: {self.max_sol_size}, selects: {self.get_pairs_of_inds(tmp_selects)}")
                    # if row < height - 1:
                self.smart_greedy(tmp_selects, row + 1)

    @staticmethod
    def get_pairs_of_inds(selects: np.ndarray) -> list:
        rows, cols = np.where(selects)
        return list(zip(rows, cols))


if __name__ == "__main__":
    set_log()
    BLOCKS_NUM = 30
    BLOCK_SIZE = 10

    block = examples.create_block_matrix(batch_size=1, blocks_num=BLOCKS_NUM, block_size=BLOCK_SIZE, epsilon=0.1,
                                         seed=1).squeeze()
    # scanned = 0.0
    selects = np.zeros((BLOCKS_NUM, BLOCK_SIZE))
    # max_selects = selects.copy()
    # iteration = 0
    sg = SmartGreedy(collision_matrix=block, height=BLOCKS_NUM, width=BLOCK_SIZE)
    sg.smart_greedy(selects=selects, row=0)
    logging.info(sg.max_selects)
    logging.info(sg.max_selects.flatten() @ block @ sg.max_selects.flatten())
