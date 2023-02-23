import sys

sys.path.append(r"res")

import numpy as np
import logging

from data import examples
from utils import set_log

class SmartGreedy:
    def __init__(self, height: int, width: int, collision_matrix: np.ndarray):
        self.max_sol_size: int = 0
        self.max_selects: np.ndarray = np.zeros((height, width))

        self.iteration: int = 0
        self.scanned: float = 0.0
        self.height = height
        self.width = width
        self.collision_matrix = collision_matrix

    def smart_greedy(self, selects: np.ndarray, row: int):
        self.iteration = self.iteration + 1

        if ((selects.sum() + self.height - row) <= self.height - 1) or (self.max_sol_size >= self.height):
            self.scanned += self.width ** -(row + 1)
            logging.info(
                f"iteration: {self.iteration}, scanned: {self.scanned}, row{row}, no recursion: {self.get_pairs_of_inds(selects)}")
        else:
            logging.info(f"row: {row}, inside recursion")

        # sols = []
        #     logging.info(
        #         f"iteration: {self.iteration}, row: {row}, col: {-1}, max_sol_size: {(self.max_sol_size, self.max_selects.sum())},selects: {self.get_pairs_of_inds(selects)}")
            #self.smart_greedy(selects.copy(), row + 1)  
            # sols.append(sol)
            for col in range(self.width):
                tmp_selects = selects.copy()

                if np.dot(self.collision_matrix[row * self.width + col], selects.flatten()) == 0:
                    tmp_selects[row, col] = 1

                    self.max_sol_size = max(self.max_sol_size, tmp_selects.sum())
                    if tmp_selects.sum() == self.max_sol_size:
                        self.max_selects = tmp_selects
                    logging.info(
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
