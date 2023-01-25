from pathlib import Path
import os
from typing import Set, Tuple, List
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nets import co


def draw_remove(opt_sol: np.ndarray, opt_collisions: Set[Tuple]):
    res_dir = Path(r"nets/res/remove")
    if res_dir.exists():
        co.rm_tree(res_dir)
    else:
        res_dir.mkdir(parents=True)
    selects = co.get_selects(opt_sol)
    df = pd.DataFrame(list(opt_collisions))
    gb = df[1].groupby(df[0])
    size_ts = gb.size().sort_values(ascending=False)
    counter = 0
    for ind, val in size_ts.items():
        plt.close("all")
        inds = list(gb.get_group(ind).values) + [ind]
        selects = selects.flatten()
        selects[inds] = 0.5
        f, ax = plt.subplots(figsize=(6, 12))
        ax.imshow(selects.reshape(30, 10))
        output_fname = res_dir / f"{str(counter).zfill(4)}.png"
        plt.savefig(output_fname)
        counter += 1
        selects[inds] = 0
        f, ax = plt.subplots(figsize=(6, 12))
        ax.imshow(selects.reshape(30, 10))
        output_fname = res_dir / f"{str(counter).zfill(4)}.png"
        plt.savefig(output_fname)
        counter += 1

    inputs = res_dir / '%04d.png'
    output = res_dir / 'output.mp4'
    cmd = f'ffmpeg -framerate 2 -i {inputs} {output}'
    os.system(cmd)


def draw_repair(removed_sol: np.ndarray, candidates: List[Tuple]):
    res_dir = Path(r"nets/res/repair")
    if res_dir.exists():
        co.rm_tree(res_dir)
    else:
        res_dir.mkdir(parents=True)
    counter = 0
    f, ax = plt.subplots(figsize=(6, 12))
    ax.imshow(removed_sol)
    output_fname = res_dir / f"{str(counter).zfill(4)}.png"
    plt.savefig(output_fname)
    for candidate in candidates:
        counter += 1
        removed_sol[candidate[0], candidate[1]] = 0.5
        output_fname = res_dir / f"{str(counter).zfill(4)}.png"
        f, ax = plt.subplots(figsize=(6, 12))
        ax.imshow(removed_sol)
        #logging.info(f"output_fname: {output_fname}")
        plt.savefig(output_fname)
        counter += 1
        removed_sol[candidate[0], candidate[1]] = 1

        f, ax = plt.subplots(figsize=(6, 12))
        ax.imshow(removed_sol)
        output_fname = res_dir / f"{str(counter).zfill(4)}.png"
        plt.savefig(output_fname)
    inputs = res_dir / '%04d.png'
    output = res_dir / 'output.mp4'
    cmd = f'ffmpeg -framerate 2 -i {inputs} {output}'
    os.system(cmd)

    inputs = res_dir / '%04d.png'
    output = res_dir / 'output.mp4'
    cmd = f'ffmpeg -framerate 2 -i {inputs} {output}'
    os.system(cmd)
