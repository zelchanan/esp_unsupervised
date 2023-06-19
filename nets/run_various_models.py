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


def read_data(blocks_num: int, block_size: int, percents: int, full_sol: bool, diag: bool = False) -> Tuple[
    np.ndarray, List[int]]:
    fname = f"data/yael_dataset2/gnn_k_{blocks_num}_m_{block_size}_e_{percents}{'_full_sol' if full_sol else ''}.csv"
    # output_fname = f"data/yael_dataset2/gnn_k_{blocks_num}_m_{block_size}_diag_{'on' if diag else 'of'}_e_{'_full_sol' if full_sol else ''}.csv"
    return examples.get_blocks_from_raw(fname, vec_size=blocks_num * block_size, block_size=block_size,
                                        diagonal_blocks=diag)


def get_name(root: str, algo: str, token: str, blocks_num: int, block_size: int, percents: int, full_sol: bool) -> str:
    return f"{root}/{algo}_{token}_{blocks_num}_{block_size}_{percents}{'_full_sol' if full_sol else ''}.csv"


def direct_softmax_one_trial(block: np.ndarray, block_size: int, blocks_num: int) -> Tuple[int, timedelta]:
    t0 = datetime.now()
    init_weights = np.random.randn(blocks_num, block_size)
    loss, sm_weights = ds.optim(block, init_weights)
    rm_sol = examples.remove_collisions(block, np.round(sm_weights.reshape(blocks_num, block_size)))
    sol_size = co.greedy_repair(block, rm_sol)
    return sol_size, datetime.now() - t0


def greedy_one_trial(block: np.ndarray, block_size: int, blocks_num: int) -> Tuple[int, timedelta]:
    t0 = datetime.now()
    size = co.greedy_repair(block, np.zeros((blocks_num, block_size)))
    return size, datetime.now() - t0


def simplex_one_trial(block: np.ndarray, block_size: int, blocks_num: int) -> Tuple[int, timedelta]:
    t0 = datetime.now()
    block = block.copy()
    w = examples.get_init_selects(blocks_num, block_size)
    val = w.flatten() @ block @ w.flatten()
    # logging.info(f"val: {val}")
    w, val = co.find_lower_neighbour(block, w)
    selects = examples.remove_collisions(block, w.reshape(blocks_num, block_size))
    size = co.greedy_repair(block, selects)
    return size, datetime.now() - t0


def run_algo(algo: str, blocks: np.ndarray, blocks_num: int, block_size: int, percents: int, trials_num: int,
             full_sol: bool,
             max_time: int = 1000000):
    algos_dict = {"greedy": greedy_one_trial,
                  "direct_softmax": direct_softmax_one_trial,
                  "simplex": simplex_one_trial}
    vals_tss = []
    times_tss = []
    for block_ind, block in enumerate(blocks):
        t_start = datetime.now()
        vals_list = []
        times_list = []
        for trial in range(trials_num):
            if trial % 10000 == 0:
                logging.info(f"{block_ind}-{trial}")
            # logging.info(f"trial: {trial}")
            t0 = datetime.now()
            size, time = algos_dict[algo](block=block, block_size=block_size, blocks_num=blocks_num)
            vals_list.append(size)
            times_list.append((datetime.now() - t0).total_seconds())
            if (datetime.now() - t_start).total_seconds() > max_time:
                break
        vals_ts = pd.Series(vals_list, name=block_ind)
        vals_tss.append(vals_ts)
        times_ts = pd.Series(times_list, name=block_ind)
        times_tss.append(times_ts)

    vals_df = pd.concat(vals_tss, axis=1)
    times_df = pd.concat(times_tss, axis=1)
    vals_fname = get_name(root="time_results", algo=algo, token="vals", blocks_num=blocks_num, block_size=block_size,
                          percents=percents, full_sol=full_sol)
    times_fname = get_name(root="time_results", algo=algo, token="times", blocks_num=blocks_num, block_size=block_size,
                           percents=percents, full_sol=full_sol)
    vals_df.to_csv(vals_fname)
    times_df.to_csv(times_fname)
    return vals_df, times_df






def read_res(algo: str, blocks_num: int, block_size: int, full_sol: bool):
    res_path = Path("results") / f"{algo}_vals_{blocks_num}_{block_size}{'_full_sol' if full_sol else ''}.csv"
    df = pd.read_csv(res_path, index_col=0)
    logging.info(f"{df.max()},{df.std()}")


def analyse_times():
    algos = ["greedy", "direct_softmax", "simplex"]
    times_dict = {"greedy": 0.02, "direct_softmax": 0.8, "simplex": 600}
    for full_sol in [True, False]:
        f, ax = plt.subplots()
        ax.set_title(full_sol)
        for algo in algos:
            times_df = pd.read_csv(f"results/{algo}_times_100_20{'_full_sol' if full_sol else ''}.csv", index_col=0)
            vals_df = pd.read_csv(f"results/{algo}_vals_100_20{'_full_sol' if full_sol else ''}.csv", index_col=0)
            cummax = vals_df.cummax()
            mask = cummax != cummax.shift(1)
            means = cummax.mean(axis=1)
            means.index = means.index * times_dict[algo]
            means.plot(ax=ax, label=algo, legend=True)
            logging.info(f"{algo}-{full_sol}-{times_df.sum() / len(times_df)}")

            # for colname, col in vals_df.iteritems():
            #     logging.info(f"########### {colname} #######################")
            #     logging.info(f"\n{col[mask[colname]]}")
    # plt.pause(1000)


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


# def run_all_algos(root: str, algo: str, blocks_num: int, block_size: int, ercents: int, full_sol: bool):


if __name__ == "__main__":
    set_log()
    # analyse_times()
    # plt.pause(1000)
    # #analyse_times("direct_softmax", False)
    # # read_log()
    # # divide_1000()
    blocks_num = 100
    block_size = 20
    percents = 10
    full_sol = False
    algos = ["simplex"]
    precentses = [5,20, 30, 50]
    full_sols = [True,False]
    for percents in precentses:
        for full_sol in full_sols:
            blocks, stats = read_data(blocks_num=blocks_num, block_size=block_size, percents=percents, full_sol=full_sol)
            blocks = blocks[:10]
            for algo in algos:
                logging.info(f"algo: {algo}, percents: {percents}, full_sol: {full_sol}")
                run_algo(algo=algo, blocks=blocks, blocks_num=blocks_num, block_size=block_size, percents=percents,
                         trials_num=1,
                         full_sol=full_sol,
                         max_time=900)
    # size, time = greedy_one_trial(block=blocks[0], block_size=block_size, blocks_num=blocks_num)
    # logging.info(f"greedy: {size} - {time}")
    # size, time = direct_softmax_one_trial(block=blocks[0], block_size=block_size, blocks_num=blocks_num)
    # logging.info(f"softmax: {size} - {time}")
    # size, time = simplex_one_trial(block=blocks[0], block_size=block_size, blocks_num=blocks_num)
    # logging.info(f" softmax: {size} - {time}")
    # algo = "direct_softmax"
    # read_res(algo=algo, blocks_num=blocks_num, block_size=block_size, full_sol=full_sol)
    # read_res(algo="direct_softmax",blocks_num=blocks_num, block_size=block_size,full_sol=full_sol)
    # algos = ["direct_soft_max","greedy","simplex"]

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
