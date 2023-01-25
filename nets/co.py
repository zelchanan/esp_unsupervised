import logging
import os
from typing import Dict, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from data import examples
from utils import set_log

DISTRIBUTION_LOSS_COEFF = 5
LASSO_LOSS_COEFF = 1
HIGH_LOSS_COEFF = 1
NEGATIVE_LOSS_COEFF = 1


def convexify(orig_block: np.ndarray) -> np.ndarray:
    vals, eigs = np.linalg.eig(orig_block)
    eigs = np.real(eigs)
    vals = np.real(vals)
    d = np.diag(vals)
    d[d < 0] = 0
    con = np.real(eigs @ d @ eigs.T)
    max_diag_element = np.diag(con).max()
    con[np.eye(len(con)).astype(bool)] = max_diag_element
    return con

def sqrt_block(block:np.ndarray)->np.ndarray:
    vals, eigs = np.linalg.eig(block)
    eigs = np.real(eigs)
    vals = np.real(vals)**0.5
    d = np.diag(vals)

    return  np.real(eigs @ d @ eigs.T)


def get_selects(r_weights: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    maxes = np.argmax(r_weights, axis=1).flatten()
    zeros_mask = r_weights.max(axis=1) < threshold
    selects = np.zeros_like(r_weights)
    block_inds = range(r_weights.shape[0])
    selects[block_inds, maxes] = 1
    selects[zeros_mask, :] = 0
    #logging.info(f"zeros: {zeros_mask.sum()}, sum: {selects.sum()}")

    return selects


def get_real_val(r_weights: np.ndarray, block: np.ndarray) -> float:
    # r_weights = r_weights.detach().numpy()
    # block = block.numpy().squeeze().copy()
    block = block.copy()
    block[np.eye(block.shape[0]).astype(bool)] = 0
    selects = get_selects(r_weights)
    selects = selects.flatten()
    return selects @ block @ selects


def rm_tree(pth: Path):
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()


def optimize(orig_block: np.ndarray, approx_block: np.ndarray, blocks_num: int, block_size: int, plot=False,
             init_vals=None) -> Tuple[
    Dict[float, np.ndarray], pd.DataFrame, float, np.ndarray]:
    # weights_num = block.shape[-1]
    approx_block = torch.Tensor(approx_block)
    if init_vals is not None:
        weights = torch.Tensor(init_vals).requires_grad_()
    else:
        weights = torch.zeros(blocks_num * block_size, requires_grad=True)
    # logging.info(weights)
    optimizer = torch.optim.Adam([weights], lr=1e-2)
    res_dict = {}
    old_loss = 100000
    min_loss = 100000
    res_dir = Path(r"nets/res")
    if res_dir.exists():
        rm_tree(res_dir)
    else:
        res_dir.mkdir()
    counter = 0
    scores = []
    for i in range(10000):
        dist_weights = torch.reshape(weights, (blocks_num, block_size))
        optimizer.zero_grad()
        # r_weights = weights.view(block.shape[-2], block.shape[-1])
        negative_loss = NEGATIVE_LOSS_COEFF * torch.nn.ReLU()(-weights).sum()
        high_loss = HIGH_LOSS_COEFF * torch.nn.ReLU()(weights - 1).sum()
        # distribution_loss = DISTRIBUTION_LOSS_COEFF*torch.pow(dist_weights.sum(axis=1) - 1, 2).sum()
        distribution_loss = DISTRIBUTION_LOSS_COEFF * torch.pow(dist_weights.sum(axis=1) - 1, 2).sum()
        lasso_loss = LASSO_LOSS_COEFF * (torch.sqrt(torch.abs(dist_weights) + 1e-5)).sum()
        abs_weights = torch.abs(dist_weights)+1e-5
        lasso_loss = LASSO_LOSS_COEFF * torch.sum(torch.abs(abs_weights * torch.log(1 / abs_weights)))
        collision_loss = torch.matmul(torch.matmul(weights, approx_block), weights)
        # logging.info(collision_loss)
        # collision_loss = weights @ approx_block @ weights
        constrains_loss = negative_loss + distribution_loss + lasso_loss + high_loss
        loss = collision_loss + constrains_loss
        if (loss < (0.93 * old_loss)) and plot:
            old_loss = plot_dist(counter, dist_weights, loss, old_loss, res_dir)
            counter += 1
        loss.backward()
        selects = get_selects(dist_weights.detach().numpy().copy(), threshold=0.5)
        real_val = get_real_val(selects, orig_block)
        approx_real_val = get_real_val(selects, approx_block.numpy())
        sum_ = selects.sum()

        scores.append([i, real_val, approx_real_val, sum_, loss.item(), negative_loss.item(), high_loss.item(),
                       distribution_loss.item(), collision_loss.item(),
                       lasso_loss.item()])
        if (i % 200 == 0) or (i<100):
            logging.info(
                f"ind:  {i}, real_val: {real_val}, sum: {sum_}, loss: {loss}, negative_loss: {negative_loss}, dist_loss: {distribution_loss}, "
                f"colission loss: {collision_loss}, lasso loss: {lasso_loss.item()}, real_val: {real_val}")
        if loss.item() < min_loss:
            min_loss_res = dist_weights.detach().numpy().copy()
            min_loss = np.copy(loss.item())
        if real_val not in res_dict:
            res_dict[real_val] = dist_weights.detach().numpy().copy()
        optimizer.step()
    logging.info(
        f"ind:  {i}, real_val: {real_val}, sum: {sum_}, loss: {loss}, negative_loss: {negative_loss}, dist_loss: {distribution_loss}, "
        f"colission loss: {collision_loss}, lasso loss: {lasso_loss.item()}")
    if plot:
        inputs = res_dir / 'weights_%04d.png'
        output = res_dir / 'output.mp4'
        cmd = f'ffmpeg -framerate 10 -i {inputs} {output}'
        logging.info(f"cmd: {cmd}")
        os.system(cmd)
    scores_df = pd.DataFrame(scores)
    scores_df.columns = ["ind", "real_val", "approx_real_val", "sum_", "loss", "negative_loss", "high_loss",
                         "dist_loss",
                         "collision_loss",
                         "lasso_loss"]
    return res_dict, scores_df, min_loss, min_loss_res


def plot_dist(counter, dist_weights, loss, old_loss, res_dir):
    # logging.info(f"loss: {loss}, old_loss: {old_loss}")
    old_loss = loss
    f, ax = plt.subplots(figsize=(6, 12))
    im = ax.imshow(dist_weights.detach().numpy(), vmin=-0.2, vmax=1.2)
    plt.colorbar(im)
    img = f"weights_{str(counter).zfill(4)}.png"
    output_fname = res_dir / img
    plt.savefig(output_fname)
    plt.close()
    return old_loss


def greedy_repair(collision_matrix: np.ndarray, selects: np.ndarray) -> Tuple:
    blocks_num, block_size = selects.shape
    missing_blocks = np.where((selects == 0).all(axis=1))[0]
    selects_dict = dict([])
    candidates_dict = dict([])
    for k in range(100):
        candidates_list = []
        tmp_selects = selects.copy()
        missing_blocks = np.random.permutation(missing_blocks)
        for ind, m in enumerate(missing_blocks):
            logging.info(
                f"k: {k}, ind: {ind}, sum: {tmp_selects.sum()}, val: {tmp_selects.flatten() @ collision_matrix @ tmp_selects.flatten()}")
            submatrix = collision_matrix[m * block_size:(m + 1) * block_size, :].astype(bool)
            bool_selects = tmp_selects.flatten().astype(bool)
            # logging.info(f"#: {bool_selects.sum()}")
            candidates = ((submatrix & bool_selects) == False).all(axis=1)
            # logging.info(f"block: {m}, candidates: {candidates}, sum: {candidates.sum()}")
            if candidates.any():
                selected_for_block = np.random.choice(np.where(candidates)[0])
                tmp_selects[m, selected_for_block] = 1
                candidates_list.append((m, selected_for_block))
        selects_dict[tmp_selects.sum()] = tmp_selects
        candidates_dict[tmp_selects.sum()] = candidates_list
        if tmp_selects.sum() == len(selects):
            break

    # plt.pause(1000)
    logging.info(list(selects_dict.keys()))
    min_key = max(list(selects_dict.keys()))
    return selects_dict[min_key], candidates_dict[min_key]


def get_losses(approx_block: np.ndarray, weights: np.ndarray) -> Tuple[float]:
    negative_loss = NEGATIVE_LOSS_COEFF * -(weights[weights < 0].sum())
    high_loss = HIGH_LOSS_COEFF * (weights[(weights - 1) > 0] - 1).sum()
    distribution_loss = DISTRIBUTION_LOSS_COEFF * ((weights.sum(axis=1) - 1) ** 2).sum()
    lasso_loss = LASSO_LOSS_COEFF * np.abs(np.sum(np.abs(weights) * np.log(1 / (np.abs(weights) + 1e-5))))
    collision_loss = weights.flatten() @ approx_block @ weights.flatten()
    loss = negative_loss + high_loss + distribution_loss + lasso_loss + collision_loss
    logging.info(
        f"loss: {loss}, collision_loss:  {collision_loss}, negative_loss: {negative_loss}, high_loss: {high_loss}, "
        f"distribution_loss: {distribution_loss}, lasso_loss: {lasso_loss}")
    return loss, collision_loss, negative_loss, high_loss, distribution_loss, lasso_loss


if __name__ == "__main__":
    import pickle

    blocks_num = 30
    block_size = 10
    batch_size = 16
    epsilon = 0.15
    seed = -1
    orig_block = examples.create_block_matrix(batch_size=1, blocks_num=blocks_num,
                                              block_size=block_size, epsilon=epsilon,
                                              seed=1)[0].squeeze()
    # greedy_sol, greedy_pairs = greedy_repair(orig_block, np.zeros((blocks_num, block_size)))
    # pickle.dump(orig_block, file = open(f"data/block_{epsilon}.p", mode="wb"))
    # pickle.dump(greedy_sol, file= open(f"data/greedy_sol_{epsilon}.p", mode="wb"))
    orig_block = pickle.load(open(f"data/block_{epsilon}.p", mode="rb"))
    greedy_sol = pickle.load(open(f"data/greedy_sol_{epsilon}.p", mode="rb"))
    approx_block = convexify(orig_block)
    print(greedy_sol.sum())
    res_dict, scores_df, min_loss, min_loss_res = optimize(orig_block=orig_block, approx_block=approx_block,
                                                           blocks_num=blocks_num, block_size=block_size,
                                                           init_vals=greedy_sol.flatten())
    print(scores_df)
