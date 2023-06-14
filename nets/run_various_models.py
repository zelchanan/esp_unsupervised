import sys

from matplotlib import pyplot as plt

sys.path.append(".")

import glob
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

import logging
from datetime import datetime

pd.set_option("display.max_rows", 300)

from nets import toy_net, co, direct_softmax_opt as ds, smart_greedy
from data import examples
from utils import set_log


def read_data(blocks_num: int, block_size: int, percents: int, full_sol: bool, diag: bool = False) -> Tuple[
    np.ndarray, List[int]]:
    fname = f"data/yael_dataset2/gnn_k_{blocks_num}_m_{block_size}_e_{percents}{'_full_sol' if full_sol else ''}.csv"
    # output_fname = f"data/yael_dataset2/gnn_k_{blocks_num}_m_{block_size}_diag_{'on' if diag else 'of'}_e_{'_full_sol' if full_sol else ''}.csv"
    return examples.get_blocks_from_raw(fname, vec_size=blocks_num * block_size, block_size=block_size,
                                        diagonal_blocks=diag)


def run_direct_softmax(blocks: np.ndarray, blocks_num: int, block_size: int, full_sol: bool, trials_num: int) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    vals_dict = {}
    times_dict = {}
    t0 = datetime.now()
    for block_ind, block in enumerate(blocks):
        logging.info(f"###################### DirectSM {block_ind} #############################")
        vals_dict[block_ind] = []
        times_dict[block_ind] = []

        all_sm_weights = []

        for i in range(trials_num):
            t0 = datetime.now()
            init_weights = np.random.randn(blocks_num, block_size)
            loss, sm_weights = ds.optim(block, init_weights)

            rm_sol = examples.remove_collisions(block, np.round(sm_weights.reshape(blocks_num, block_size)))
            selects, candidates, stats = co.greedy_repair(block, rm_sol)
            sol_size = selects.sum()
            if selects.sum() not in vals_dict[block_ind]:
                # losses.append(selects.sum())
                all_sm_weights.append(selects)
                print(selects.sum())
            vals_dict[block_ind].append(sol_size)
            times_dict[block_ind].append(((datetime.now() - t0).microseconds) / 1e6)

        # print(pd.Series(losses).value_counts().sort_index())
    vals_df = pd.DataFrame(vals_dict)
    times_df = pd.DataFrame(times_dict)
    res_dir = Path("results")
    vals_df.to_csv(res_dir / f"direct_softmax_vals_{blocks_num}_{block_size}{'_full_sol' if full_sol else ''}.csv")
    times_df.to_csv(res_dir / f"direct_softmax_times_{blocks_num}_{block_size}{'_full_sol' if full_sol else ''}.csv")
    return vals_df, times_df


def run_simplex_algoriyhm(blocks: np.ndarray, blocks_num: int, block_size: int, full_sol: bool, trials_num: int) -> \
        Tuple[
            pd.DataFrame, pd.DataFrame]:
    vals_dict = {}
    times_dict = {}
    for block_ind, block in enumerate(blocks[:]):
        vals_dict[block_ind] = []
        times_dict[block_ind] = []
        for trial in range(trials_num):
            t0 = datetime.now()
            logging.info(f"############## {block_ind} - {trial} #######################")
            block = block.copy()
            # w = np.zeros((blocks_num,block_size))
            # w[:,0]=1
            w = examples.get_init_selects(blocks_num, block_size)
            val = w.flatten() @ block @ w.flatten()
            logging.info(f"val: {val}")
            # while val != -1:
            w, val = co.find_lower_neighbour(block, w)
            selects = examples.remove_collisions(block, w.reshape(blocks_num, block_size))
            remove_sol_size = selects.sum()
            selects, candidates, stats = co.greedy_repair(block, selects)
            sol_size = selects.sum()
            vals_dict[block_ind].append(sol_size)
            times_dict[block_ind].append(((datetime.now() - t0).microseconds) / 1e6)
            # t0=datetime.now()
            logging.info(
                f"################## REMOVE SOL_SIZE: {remove_sol_size}, sol_size: {sol_size} #########################")
            if sol_size == blocks_num:
                break
    tss = []
    for key, vals in vals_dict.items():
        ts = pd.Series(vals)
        ts.name = key
        tss.append(ts)
    vals_df = pd.concat(tss, axis=1)
    tss = []
    for key, vals in times_dict.items():
        ts = pd.Series(vals)
        ts.name = key
        tss.append(ts)
    times_df = pd.concat(tss, axis=1)
    res_dir = Path("results")
    vals_df.to_csv(res_dir / f"simplex_vals_{blocks_num}_{block_size}{'_full_sol' if full_sol else ''}.csv")
    times_df.to_csv(res_dir / f"simplex_times_{blocks_num}_{block_size}{'_full_sol' if full_sol else ''}.csv")
    return vals_df, times_df


def run_greedy(blocks: np.ndarray, blocks_num: int, block_size: int, trials_num: int, full_sol: bool):
    vals_dict = {}
    times_dict = {}
    for block_ind, block in enumerate(blocks):
        vals_dict[block_ind] = []
        times_dict[block_ind] = []
        for trial in range(trials_num):
            if trial % 10000 == 0:
                logging.info(f"{block_ind}-{trial}")
            # logging.info(f"trial: {trial}")
            t0 = datetime.now()
            selects, candidates, size = co.greedy_repair(blocks[0], np.zeros((blocks_num, block_size)))
            vals_dict[block_ind].append(size.max())
            times_dict[block_ind].append(((datetime.now() - t0).microseconds) / 1e6)
    vals_df = pd.DataFrame(vals_dict)
    times_df = pd.DataFrame(times_dict)
    res_dir = Path("results")
    vals_df.to_csv(res_dir / f"greedy_vals_{blocks_num}_{block_size}{'_full_sol' if full_sol else ''}.csv")
    times_df.to_csv(res_dir / f"greedy_times_{blocks_num}_{block_size}{'_full_sol' if full_sol else ''}.csv")
    return vals_df, times_df


def read_log():
    ts: pd.Series = pd.Series(open("logs/info - Copy.log").readlines())
    mask = ts.str.contains("REMOVE SOL_SIZE") | ts.str.contains("\d \- \d")

    open("logs/stats.txt", "w").write("\n".join(ts[mask].tolist()))


def read_res(algo, blocks_num: int, block_size: int, full_sol: bool):
    res_path = Path("results") / f"{algo}_vals_{blocks_num}_{block_size}{'_full_sol' if full_sol else ''}.csv"
    df = pd.read_csv(res_path, index_col=0)
    logging.info(f"{df.max()},{df.std()}")


def analyse_times():
    algos = ["greedy", "direct_softmax","simplex"]
    times_dict = {"greedy": 0.02, "direct_softmax": 0.8,"simplex":600}
    for full_sol in [True, False]:
        f, ax = plt.subplots()
        ax.set_title(full_sol)
        for algo in algos:
            times_df = pd.read_csv(f"results/{algo}_times_100_20{'_full_sol' if full_sol else ''}.csv", index_col=0)
            vals_df = pd.read_csv(f"results/{algo}_vals_100_20{'_full_sol' if full_sol else ''}.csv", index_col=0)
            cummax = vals_df.cummax()
            mask = cummax != cummax.shift(1)
            means = cummax.mean(axis=1)
            means.index = means.index*times_dict[algo]
            means.plot(ax=ax, label=algo, legend=True)
            logging.info(f"{algo}-{full_sol}-{times_df.sum()/len(times_df)}")

            # for colname, col in vals_df.iteritems():
            #     logging.info(f"########### {colname} #######################")
            #     logging.info(f"\n{col[mask[colname]]}")
    #plt.pause(1000)


def add_files():
    lines_ts = pd.Series(open("results/no_sol_simplex.txt").readlines())
    mask = lines_ts.str.contains("sol_size")
    vals = lines_ts[mask].str.extract("(?<=sol_size: )(\d+\.\d)")
    times = lines_ts[mask].str.extract("(\d\d):(\d\d):(\d\d),(\d+)").astype(float)
    from datetime import timedelta
    times = times.apply(lambda x: timedelta(hours=x.iloc[0], minutes=x.iloc[1], seconds=x.iloc[2],
                                            microseconds=np.int(1000 * x.iloc[3])), axis=1)
    diffs = times.map(lambda x: x.total_seconds()).diff()


def divide_1000():
    # p1 = Path("results/direct_softmax_times_100_20_full_sol.csv")
    # p2 = Path("results/direct_softmax_times_100_20.csv")
    # p3 = Path("results/greedy_times_100_20_full_sol.csv")
    p4 = Path("results/greedy_times_100_20.csv")
    for p in [p4]:
        df = pd.read_csv(p, index_col=0)
        df = df / 10
        df.to_csv(p)


if __name__ == "__main__":
    set_log()
    analyse_times()
    plt.pause(1000)
    #analyse_times("direct_softmax", False)
    # read_log()
    # divide_1000()
    blocks_num = 100
    block_size = 20
    percents = 10
    full_sol = False
    read_res(algo="direct_softmax",blocks_num=blocks_num, block_size=block_size,full_sol=full_sol)
    algos = ["direct_soft_max","greedy","simplex"]

    # for full_sol in [False]:
    #     blocks, stats = read_data(blocks_num=blocks_num, block_size=block_size, percents=percents, full_sol=full_sol)
    #     run_direct_softmax(blocks=blocks, blocks_num=blocks_num, block_size=block_size,full_sol=full_sol, trials_num=5000)
    #     run_simplex_algoriyhm(blocks=blocks, blocks_num=blocks_num, block_size=block_size,full_sol=full_sol,trials_num=6)
    # run_greedy(blocks=blocks, blocks_num=blocks_num, block_size=block_size,full_sol=full_sol, trials_num=300000)
    #
    # full_sol = True
    # blocks, stats = read_data(blocks_num=blocks_num, block_size=block_size, percents=percents, full_sol=full_sol)
    # run_simplex_algoriyhm(blocks=blocks, blocks_num=blocks_num, block_size=block_size,full_sol=full_sol,trials_num=10)

    # read_res("direct_softmax", blocks_num, block_size, full_sol)
    # logging.info(blocks.shape)
    # df = pd.concat([vals_ts, times_ts], axis=1)
    #
    # res_dir = Path("results")
    # res_dir.mkdir(exist_ok=True)
    # df.to_csv(res_dir / "simplex.csv")
