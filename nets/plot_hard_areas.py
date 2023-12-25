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


def show_hardness(algo: str, random: bool):
    pairs = [(100, 20), (30, 10), (10, 10)]
    f, axes = plt.subplots(2,1)#,figsize=(12, 22))
    for ind, full_sol in enumerate([True, False]):
        vals_list = []
        for blocks_num, block_size in pairs:
            stats_df = pd.read_csv(f"time_results{'_random' if random else ''}/stats_{blocks_num}_{block_size}{'_random' if random else ''}.csv", index_col=0)
            gb = stats_df.groupby(["algo", "full_sol"])["mean_half"]
            vals = gb.get_group((algo, full_sol)).values / blocks_num
            v = vals[-1]
            if len(vals) < 5:
                for i in range(5 - len(vals)):
                    vals = np.append(vals, v)

            vals_list.append((vals * 100).astype(int))

        hard_map = np.stack(vals_list)
        if ind == 0:
            s = axes[ind].imshow(hard_map, vmax=100, vmin=0)
        else:
            axes[ind].imshow(hard_map, vmax=100, vmin=0)
            logging.info(hard_map)
        axes[ind].set_xticks(range(5))
        axes[ind].set_xticklabels([5, 10, 20, 30, 50])#, fontsize=24)
        axes[ind].set_yticks(range(3))
        axes[ind].set_yticklabels([100, 30, 10])#, fontsize=24)
        if ind == 0:
            axes[ind].set_xticks([])
        if ind == 1:
            axes[ind].set_xlabel("collision percents")#, fontsize=24)
        axes[ind].set_ylabel("# targets")#, fontsize=24)

        axes[ind].set_title(f"{'with' if full_sol else 'without'} full sol")#, fontsize=30)
    f.colorbar(s, ax=axes)
    plt.suptitle(algo, fontsize=18,weight='bold')
    #plt.pause(10000)
    plt.savefig(f"time_results{'_random' if random else ''}/imgs/hard_{algo}_{full_sol}")


def get_dict() -> dict:
    d = {}
    for blocks_num, block_size in [(100, 20), (30, 10), (10, 10)]:
        for full_sol in [True, False]:
            for random in [True, False]:
                for percents in [5, 10, 20, 30, 50]:
                    for algo in ("direct_softmax", "greedy","simplex"):
                        fname = get_name(root="time_results", algo=algo, token="times", blocks_num=blocks_num,
                                         block_size=block_size, percents=percents, full_sol=full_sol, random=random)
                        if os.path.exists(fname):
                            df = pd.read_csv(fname, index_col=0)
                            if len(df.unstack())>=50:
                                m = df.unstack().sort_values().iloc[10:-10].mean()
                            else:
                                m = df.unstack().sort_values().mean()
                            # logging.info(f"fname: {fname}, mean: {m:.5f}")
                            d[(blocks_num, block_size, full_sol, random, percents, algo)] = m
                            logging.info(f"dict val: {m}")
    return d


def rename_files():
    fnames = list(Path('time_results/older').glob("*.csv"))
    for fname in fnames:
        new_fname = fname.parent / (fname.name.replace("data__data", "data"))
        fname.replace(new_fname)
        print(new_fname)


def time_by_algo_plot(blocks_num: int, block_size: int):
    algos = ["greedy", "direct_softmax", "simplex"]
    # algos = ["simplex"]
    precentses = [5, 10, 20, 30, 50]
    precentses = [20]
    full_sols = [True, False]
    full_sols = [True]
    trials_dict = get_dict()

    stats = []
    for percents in precentses:
        for full_sol in full_sols:
            f, ax = plt.subplots(figsize=(16, 12))

            for random in [True, False]:
                title = f"block num {blocks_num}, block size: {block_size}, percents: {percents}, {'with' if full_sol else 'without'} full soll"

                ax.set_title(label=title, fontsize="xx-large")
                ax.set_xlabel(xlabel="seconds", fontsize="xx-large")
                ax.set_ylabel(ylabel="matches num", fontsize="xx-large")

                for algo in algos:
                    marker = "None"#"*" #if algo == "simplex" else None
                    fname = get_name(root="time_results", algo=algo, token="vals", blocks_num=blocks_num,
                                     block_size=block_size, percents=percents, full_sol=full_sol, random=random)
                    logging.info(fname)
                    fname = Path(fname)
                    if os.path.exists(fname):
                        df = pd.read_csv(fname, index_col=0)
                        df = df.cummax()
                        l = df.notnull().sum().max()
                        df = df.iloc[:l, :]
                        df = df.ffill()
                        # if algo != "simplex":
                        secs = trials_dict[(blocks_num, block_size, full_sol, random, percents, algo)]
                        # else:
                        #secs = 120
                        max_ts = df.cummax().mean(axis=1)
                        max_ts.index = (max_ts.index+1) * secs
                        max_ts.loc[120] =  max_ts.iloc[-1]
                        max_ts[l * secs] = max_ts.iloc[-1]
                        max_ts = max_ts.loc[:120]
                        if random:
                            max_ts = max_ts+2
                        color = "r" if random else "b"
                        ls = "-" if algo == "greedy" else "-." if algo == "direct_softmax" else ":"
                        label = f"{algo} {'random' if random else ''}"
                        # if algo == "simplex":
                        max_ts.plot(ax=ax, label=label, color=color, legend=True, marker=marker, ls=ls)
                        # else:
                        #     max_ts.index = np.arange(1, len(max_ts) + 1) * secs
                        #     max_ts.plot(ax=ax, label=algo, legend=True, marker=marker)
            ax.grid(True)
            #ax.set_ylim([-5,105])
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.set_ylim([0,115])
            plt.legend(loc=4,fontsize = "xx-large")
            output_fname = f"img_{blocks_num}_{block_size}_{percents}_{'_full_sol' if full_sol else ''}"
            d = fname.parent / 'imgs'
            d.mkdir(exist_ok=True)
            plt.savefig(d / output_fname, bbox_inches="tight", pad_inches=0.25)

    plt.pause(10000)


if __name__ == "__main__":
    set_log()
    time_by_algo_plot(blocks_num=100, block_size=20)
    #show_hardness("direct_softmax",True)
    #show_hardness("direct_softmax",False)
    # show_hardness("direct_softmax")
    # show_hardness("simplex")
