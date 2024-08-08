from scipy.optimize import LinearConstraint

from data import examples


def create_problem():
    b = examples.create_random_batch(blocks_num=20, block_size=10, epsilon=0.3, batch_size=1)

if __name__ == "__main__":
    create_problem()

