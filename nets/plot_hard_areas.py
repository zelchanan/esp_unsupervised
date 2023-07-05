import os.path
import sys

from matplotlib import pyplot as plt

sys.path.append(".")

import glob
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

import logging
from datetime import datetime, timedelta

pd.set_option("display.max_rows", 300)

from nets import toy_net, co, direct_softmax_opt as ds, smart_greedy
from data import examples
from utils import set_log

from nets.run_various_models import get_name, trials_dict


def show_hardness(algo:str):
    pairs = [(100, 20), (30, 10), (10, 10)]
    f, axes = plt.subplots(2, 1, figsize=(12, 20))
    for ind, full_sol in enumerate([True,False]):
        vals_list = []
        for blocks_num, block_size in pairs:
            stats_df = pd.read_csv(f"time_results/stats_{blocks_num}_{block_size}.csv", index_col=0)
            gb = stats_df.groupby(["algo", "full_sol"])["mean_half"]
            vals = gb.get_group((algo, full_sol)).values / blocks_num
            v = vals[-1]
            if len(vals)<5:
                for i in range(5-len(vals)):
                    vals = np.append(vals,v)


            vals_list.append((vals * 100).astype(int))

        hard_map = np.stack(vals_list)
        if ind == 0:
            s = axes[ind].imshow(hard_map, vmax=100, vmin=0)
        else:
            axes[ind].imshow(hard_map, vmax=100, vmin=0)
            logging.info(hard_map)
        axes[ind].set_xticks(range(5))
        axes[ind].set_xticklabels([5, 10, 20, 30, 50], fontsize=24)
        axes[ind].set_yticks(range(3))
        axes[ind].set_yticklabels([100,30,10], fontsize=24)
        axes[ind].set_xlabel("collision percents", fontsize=24)
        axes[ind].set_ylabel("num of targets", fontsize=24)

        axes[ind].set_title(f"{'with' if full_sol else 'without'} full sol", fontsize=30)
    f.colorbar(s, ax=axes)
    plt.suptitle(algo,fontsize=48)
    plt.savefig(f"time_results/imgs/hard_{algo}_{full_sol}")


def time_by_algo_plot(blocks_num: int, block_size: int):
    algos = ["greedy", "direct_softmax", "simplex"]
    precentses = [5, 10, 20, 30, 50]
    full_sols = [True, False]

    stats = []
    for percents in precentses:
        for full_sol in full_sols:
            f, ax = plt.subplots()
            title = f"block num {blocks_num}, block size: {block_size}, percents: {percents}, {'with' if full_sol else 'without'} full soll"
            output_fname = title.replace(",", "").replace(":", "").replace(" ", "_")

            ax.set_title(title)
            ax.set_xlabel("seconds")
            ax.set_ylabel("matches num")

            for algo in algos:
                marker = "*" if algo == "simplex" else None
                fname = get_name(root="time_results", algo=algo, token="vals", blocks_num=blocks_num,
                                 block_size=block_size, percents=percents, full_sol=full_sol)
                fname = Path(fname)
                df = pd.read_csv(fname, index_col=0)
                secs = trials_dict[blocks_num][algo]
                max_ts = df.cummax().mean(axis=1)
                max_ts.index = np.arange(1, len(max_ts) + 1) * secs
                max_ts.plot(ax=ax, label=algo, legend=True, marker=marker)
            ax.grid(True)
            d = fname.parent / 'imgs'
            d.mkdir(exist_ok=True)
            plt.savefig(d / output_fname, bbox_inches="tight", pad_inches=0.25)

    # plt.pause(1000)


if __name__ == "__main__":
    set_log()
    # time_by_algo_plot(blocks_num=100, block_size=20)
    #show_hardness("greedy")
    #show_hardness("direct_softmax")
    show_hardness("simplex")
