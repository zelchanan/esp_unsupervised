import os.path
import sys

import torch

sys.path.append(".")
import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt

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

trials_dict = {100: {"greedy": 0.01, "direct_softmax": 0.5, "simplex": 120},
               30: {"greedy": 0.001, "direct_softmax": 0.25, "simplex": 1},
               # 20: {"greedy": 0.0005, "direct_softmax": 0.1, "simplex": 1},
               10: {"greedy": 0.0001, "direct_softmax": 0.06, "simplex": 0.015}}


def read_data(blocks_num: int, block_size: int, percents: int, full_sol: bool, diag: bool = False) -> Tuple[
    np.ndarray, List[int]]:
    fname = f"data/yael_dataset2/gnn_k_{blocks_num}_m_{block_size}_e_{percents}{'_full_sol' if full_sol else ''}.csv"
    # output_fname = f"data/yael_dataset2/gnn_k_{blocks_num}_m_{block_size}_diag_{'on' if diag else 'of'}_e_{'_full_sol' if full_sol else ''}.csv"
    return examples.get_blocks_from_raw(fname, vec_size=blocks_num * block_size, block_size=block_size,
                                        diagonal_blocks=diag)


def get_name(root: str, algo: str, token: str, blocks_num: int, block_size: int, percents: int, random: bool,
             full_sol: bool) -> str:
    if random:
        root = root + "_random"
    Path(root).mkdir(exist_ok=True)
    return f"{root}/{algo}_{token}_{blocks_num}_{block_size}_{percents}_{'random' if random else 'data'}{'_full_sol' if full_sol else ''}.csv"


def direct_softmax_one_trial(block: np.ndarray, block_size: int, blocks_num: int) -> \
        Tuple[int, timedelta]:
    t0 = datetime.now()
    init_weights = np.random.randn(blocks_num, block_size)
    loss, sm_weights = ds.optim(block, init_weights)
    rm_sol = examples.remove_collisions(block, np.round(sm_weights.reshape(blocks_num, block_size)))
    #sol_size = co.greedy_repair(block, rm_sol)
    return rm_sol.flatten()@block@rm_sol.flatten(), datetime.now() - t0


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
    # w, val = co.find_lower_neighbour(block, w)
    w, val = co.find_any_lower(block, w)
    selects = examples.remove_collisions(block, w.reshape(blocks_num, block_size))
    size = co.greedy_repair(block, selects)
    # logging.info(f"size: {size}")
    return size, datetime.now() - t0


def run_algo(algo: str, blocks: np.ndarray, blocks_num: int, block_size: int, percents: int, trials_num: int,
             full_sol: bool, random: bool,
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
        max_size = 0
        for trial in range(trials_num):
            if trial % 10000 == 0:
                logging.info(f"{block_ind}-{trial}")
            # logging.info(f"trial: {trial}")
            t0 = datetime.now()
            if algo == "direct_softmax":

                size, time = algos_dict[algo](block=block, block_size=block_size, blocks_num=blocks_num)
            else:
                size, time = algos_dict[algo](block=block, block_size=block_size, blocks_num=blocks_num)
            vals_list.append(size)
            logging.info(f"new priority size: {size},max: {max(vals_list)}, min: {min(vals_list)}")
            if max(vals_list) > max_size:
                max_size = max(vals_list)
                logging.info(f"time: {(datetime.now() - t_start).total_seconds()}, new max size: {max_size}")
            times_list.append((datetime.now() - t0).total_seconds())
            if ((datetime.now() - t_start).total_seconds() > max_time) or (max_size == blocks_num):
                break

        vals_ts = pd.Series(vals_list, name=block_ind)
        vals_tss.append(vals_ts)
        times_ts = pd.Series(times_list, name=block_ind)
        times_tss.append(times_ts)

    vals_df = pd.concat(vals_tss, axis=1)
    times_df = pd.concat(times_tss, axis=1)
    vals_fname = get_name(root="time_results", algo=algo, token="vals", blocks_num=blocks_num, block_size=block_size,
                          percents=percents, full_sol=full_sol, random=random)
    times_fname = get_name(root="time_results", algo=algo, token="times", blocks_num=blocks_num, block_size=block_size,
                           percents=percents, full_sol=full_sol, random=random)
    #vals_df.to_csv(vals_fname)
    #times_df.to_csv(times_fname)
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


def rename_res():
    fnames = list(Path("results").glob("*.csv"))
    for fname in fnames:
        new_name = fname.name.replace("100_20", "100_20_10")
        new_fname = fname.parent / new_name
        fname.rename(new_fname)


# def run_all_algos(root: str, algo: str, blocks_num: int, block_size: int, ercents: int, full_sol: bool):


def run_all_algos(algos: Tuple[str, ...], precentses: Tuple[int, ...], full_sols: Tuple[bool, ...], blocks_num: int,
                  block_size: int, random=False, priority: bool = False):
    # algos = ["greedy"]

    for percents in precentses:
        for full_sol in full_sols:

            if random:
                blocks = examples.create_random_batch(blocks_num=blocks_num, block_size=block_size,
                                                      epsilon=percents / 100,
                                                      batch_size=10, sol=full_sol)
            else:
                blocks, stats = read_data(blocks_num=blocks_num, block_size=block_size, percents=percents,
                                          full_sol=full_sol)
            if priority:
                blocks = examples.add_priorities(blocks= blocks, blocks_num=blocks_num,block_size=block_size)
            for algo in algos:
                logging.info(
                    f"blocks num: {blocks_num}, block_size: {block_size}, algo: {algo}, percents: {percents}, full_sol: {full_sol}")

                run_algo(algo=algo, blocks=blocks, blocks_num=blocks_num, block_size=block_size, percents=percents,
                         trials_num=1000000,
                         full_sol=full_sol,
                         max_time=120, random=random)


def add_perfernces_loss(perfernces: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    maxes = torch.max(weights, dim=1)
    return torch.inner(maxes, perfernces)


def summurize_res(blocks_num: int, block_size: int, random: bool):
    algos = ["greedy", "direct_softmax", "simplex"]
    precentses = [5, 10, 20, 30, 50]
    full_sols = [True, False]

    stats = []
    for percents in precentses:
        for full_sol in full_sols:
            for algo in algos:
                fname = get_name(root="time_results", algo=algo, token="vals", blocks_num=blocks_num,
                                 block_size=block_size, percents=percents, full_sol=full_sol, random=random)
                if os.path.exists(fname):
                    df = pd.read_csv(fname, index_col=0)

                    samples_half = int(np.ceil(0.5 / trials_dict[blocks_num][algo]))
                    samples_30 = int(np.ceil(30 / trials_dict[blocks_num][algo]))
                    samples_60 = int(np.ceil(60 / trials_dict[blocks_num][algo]))
                    samples_600 = int(np.ceil(600 / trials_dict[blocks_num][algo]))

                    med_half = df.iloc[:samples_half, :].max().median()
                    med_30 = df.iloc[:samples_30, :].max().median()
                    med_60 = df.iloc[:samples_60, :].max().median()
                    med_600 = df.iloc[:samples_600, :].max().median()

                    mean_half = df.iloc[:samples_half, :].max().mean()
                    mean_30 = df.iloc[:samples_30, :].max().mean()
                    mean_60 = df.iloc[:samples_60, :].max().mean()
                    mean_600 = df.iloc[:samples_600, :].max().mean()

                    logging.info(f"\nmed_half: {med_half}, med_30: {med_30}, med_60: {med_60}, med_600: {med_600}")
                    logging.info("#####################################")
                    stats.append(
                        [algo, percents, full_sol, med_half, med_30, med_60, med_600, mean_half, mean_30, mean_60,
                         mean_600])
    stat_df = pd.DataFrame(stats, columns=["algo", "percents", "full_sol", "med_half", "med_30", "med_60", "med_600",
                                           "mean_half", "mean_30", "mean_60", "mean_600"])
    stat_df = stat_df.sort_values(["percents", "full_sol", "algo"])
    stat_df.to_csv(
        f"time_results{'_random' if random else ''}/stats_{blocks_num}_{block_size}{'_random' if random else ''}.csv")


def bp_min_match():
    from scipy.optimize import linear_sum_assignment
    s = np.random.randint(800, 1000, size=(20, 25))
    linear_sum_assignment(s)


def vertex_cover():
    block = examples.create_block_matrix(blocks_num=100, block_size=20, epsilon=0.1,
                                         seed=-1, diagonal_blocks=True)
    block = examples.add_sol_to_data(block=block, blocks_num=100, block_size=20)
    block_to_manipulate = block.copy()
    counter = 0
    while block_to_manipulate.sum() > 0:
        collisions = block_to_manipulate.sum(axis=1)
        max_collisions_ind = np.argmax(collisions)
        # inds = np.where(block_to_manipulate[max_collisions_ind])[0]
        # len_of_inds = len(inds)
        block_to_manipulate[max_collisions_ind, :] = 0
        block_to_manipulate[:, max_collisions_ind] = 0
        counter += 1
        logging.info(f"{counter} - {block_to_manipulate.sum()}")
        # f,ax = plt.subplots()
        # ax.imshow(block_to_manipulate)
    plt.pause(1000)


def get_dict() -> dict:
    d = {}
    for blocks_num, block_size in [(100, 20), (30, 10), (10, 10)]:
        for full_sol in [True, False]:
            for random in [True, False]:
                for percents in [10, 30, 50]:
                    for algo in ("direct_softmax", "greedy"):
                        fname = get_name(root="time_results", algo=algo, token="times", blocks_num=blocks_num,
                                         block_size=block_size, percents=percents, full_sol=full_sol, random=random)
                        if os.path.exists(fname):
                            df = pd.read_csv(fname, index_col=0)
                            m = df.unstack().sort_values().iloc[3:-3].mean()
                            logging.info(f"fname: {fname}, mean: {m:.5f}")
                            d[(blocks_num, block_size, full_sol, percents, algo)] = m
    return d


def get_pririty(blocks_num: int, block_size: int) -> np.ndarray:
    return np.repeat(np.random.rand(blocks_num), repeats=block_size)


if __name__ == "__main__":
    set_log()

    # get_dict()

    # algos = ("direct_softmax", "greedy")
    algos = ("direct_softmax",)
    precentses = (10,)
    # precentses = (5,)
    full_sols = (True, False)
    # full_sols = (False,)
    random = False
    epsilon = 0.1
    blocks_num = 30
    block_size = 10
    priority = False
    # block = examples.create_lexical_matrix(blocks_num=blocks_num, block_size=block_size, epsilon=epsilon, sol=True, seed=-1)
    # block = examples.add_sol_to_data(block=block,blocks_num=blocks_num,block_size=block_size)
    # init_weights = np.random.rand(blocks_num, block_size)
    # loss,selects = ds.optim(block,init_weights)
    # selects = selects.reshape(init_weights.shape)
    # examples.remove_collisions(block,np.round(selects))
    # priority = get_pririty(blocks_num=blocks_num, block_size=block_size)
    run_all_algos(algos=algos, precentses=precentses, full_sols=full_sols, blocks_num=blocks_num, block_size=block_size,
                  random=random, priority=priority)
    # #
    # # # blocks_num = 30
    # # block_size = 30
    # blocks, stats = read_data(blocks_num=blocks_num, block_size=block_size, percents=10,
    #                           full_sol=False)
    # run_all_algos(algos=algos, precentses=precentses, full_sols=full_sols, blocks_num=blocks_num, block_size=block_size,
    #              random=random)
    # logging.info("done")
    # summurize_res(blocks_num=30, block_size=30, random=True)
    # summurize_res(blocks_num=100, block_size=20, random=False)
    # summurize_res(blocks_num=100, block_size=20, random=True)
    # summurize_res(blocks_num=30, block_size=30,random=random)
    # time_by_algo_plot(blocks_num=100, block_size=20)
    # rename_res()
    # analyse_times()
    # plt.pause(1000)
    # #analyse_times("direct_softmax", False)
    # # read_log()
    # # divide_1000()
    # run_all_algos()
    # blocks_num = 100
    # block_size=20
    # full_sol= True
    # percents=10
    # counter = 0
    # f,axes = plt.subplots(3,3)
    # for blocks_num, block_size in [(10,10),(30,10),(100,20)]:
    #     for percents in [10,30,50]:
    #
    #         blocks, stats = read_data(blocks_num=blocks_num, block_size=block_size, percents=percents, full_sol=full_sol)
    #         ax = axes[counter//3,counter%3]
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.set_title(f"{blocks_num}-{block_size}-{percents}")
    #         ax.imshow(blocks[0])
    #         counter += 1
    # plt.savefig("ds_movs/configg")

    # init_weights = np.random.randn(blocks_num, block_size)
    # loss, sm_weights = ds.optim(blocks[0], init_weights,toplot=True)
    # blocks = blocks[:10]
    # for algo in algos:
    #     run_algo(algo=algo, blocks=blocks, blocks_num=blocks_num, block_size=block_size, percents=percents,
    #              trials_num=1,
    #              full_sol=full_sol,
    #              max_time=900)
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
    # len_of_inds  = len(block_to_manipulate)
    # block_to_manipulate = block.copy()
