import os.path
import sys
import cProfile, pstats

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


def read_data(blocks_num: int, block_size: int, percents: int, diag: bool = False) -> Tuple[
    np.ndarray, List[int]]:
    fname = f"data/yael_dataset2/gnn_k_{blocks_num}_m_{block_size}_e_{percents}.csv"
    # output_fname = f"data/yael_dataset2/gnn_k_{blocks_num}_m_{block_size}_diag_{'on' if diag else 'of'}_e_{'_full_sol' if full_sol else ''}.csv"
    return examples.get_blocks_from_raw(fname, vec_size=blocks_num * block_size, block_size=block_size,
                                        diagonal_blocks=diag)


def get_name(root: str, algo: str, token: str, blocks_num: int, block_size: int, percents: int, random: bool,
             full_sol: bool) -> str:
    if random:
        root = root + "_random"
    Path(root).mkdir(exist_ok=True)
    return f"{root}/{algo}_{token}_{blocks_num}_{block_size}_{percents}_{'random' if random else 'data'}{'_full_sol' if full_sol else ''}.csv"


def direct_softmax_one_trial(block: np.ndarray, block_size: int, blocks_num: int, init_weights=np.empty(0)) -> \
        Tuple[np.ndarray, timedelta]:
    t0 = datetime.now()
    if len(init_weights) == 0:
        init_weights = np.random.randn(blocks_num, block_size)
    before_rm_sol_size, reshaped, tmp_rm_sol, rounded_loss = optim_and_remove(block, block_size, blocks_num,
                                                                              init_weights)
    rm_sol = np.zeros((blocks_num, block_size))
    last_sol_size = -1
    while tmp_rm_sol.sum() > last_sol_size:
        rm_sol = tmp_rm_sol
        last_sol_size = tmp_rm_sol.sum()
        before_rm_sol_size, reshaped, tmp_rm_sol, rounded_loss = optim_and_remove(block, block_size, blocks_num,
                                                                                  rm_sol.copy())

    repaired = co.greedy_repair(block, rm_sol)
    repaired_size = repaired.sum()
    collisions_num = repaired.flatten() @ block @ repaired.flatten()
    # logging.info(
    #     f"loss: {rounded_loss}, raw_size: {before_rm_sol_size}: rm_size: {rm_sol.sum()}, greedy_repaired: {repaired_size}, collisions: {collisions_num}")
    return repaired, datetime.now() - t0


def optim_and_remove(block, block_size, blocks_num, init_weights):
    loss, sm_weights = ds.optim(block, init_weights)
    rounded_loss = np.round(sm_weights) @ block @ np.round(sm_weights)
    before_rm_sol_size = np.round(sm_weights).sum()
    reshaped = np.round(sm_weights.reshape(blocks_num, block_size))
    rm_sol = examples.remove_collisions(block, reshaped)
    return before_rm_sol_size, reshaped, rm_sol, rounded_loss


def greedy_one_trial(block: np.ndarray, block_size: int, blocks_num: int) -> Tuple[np.ndarray, timedelta]:
    t0 = datetime.now()
    sol = co.greedy_repair(block, np.zeros((blocks_num, block_size)))
    size = sol.flatten() @ block @ sol.flatten()
    return sol, datetime.now() - t0


def simplex_one_trial(block: np.ndarray, block_size: int, blocks_num: int) -> Tuple[np.ndarray, timedelta]:
    t0 = datetime.now()
    block = block.copy()
    w = examples.get_init_selects(blocks_num, block_size)
    val = w.flatten() @ block @ w.flatten()
    # logging.info(f"val: {val}")
    # w, val = co.find_lower_neighbour(block, w)
    w, val = co.find_any_lower(block, w)
    selects = examples.remove_collisions(block, w.reshape(blocks_num, block_size))
    sol = co.greedy_repair(block, selects)
    # logging.info(f"size: {size}")
    return sol, datetime.now() - t0


def extract_stat(block: np.ndarray, sol: np.ndarray) -> tuple:
    blocks_num, block_size = sol.shape
    start_blocks_inds = (np.arange(blocks_num) * block_size).astype(int)
    # start_blocks_inds = pd.Series(sol.sum(axis=1) == 1)  # .groupby(np.diag(block))
    grouper = np.diag(block)[start_blocks_inds]
    selected_inds = sol.sum(axis=1) == 1
    counts = pd.Series(grouper[selected_inds]).value_counts()
    counts.index = counts.index.map(np.abs)
    counts = counts.sort_index()
    return tuple(counts.tolist())
    # selected_inds.groupby(start_blocks_inds)


def run_algo(algo: str, blocks: np.ndarray, blocks_num: int, block_size: int, percents: int, trials_num: int,
             full_sol: float, random: bool,
             max_time: int = 1000000, priority: np.ndarray = np.empty(0)):
    full_sol = full_sol > 0
    algos_dict = {"greedy": greedy_one_trial,
                  "direct_softmax": direct_softmax_one_trial,
                  "simplex": simplex_one_trial}
    vals_tss = []
    times_tss = []
    stats_tss = []
    for block_ind, block in enumerate(blocks):
        t_start = datetime.now()
        vals_list = []
        sols_list = []
        times_list = []
        stats_list = []
        max_size = 0
        for trial in range(trials_num):
            if trial % 10000 == 0:
                logging.info(f"{block_ind}-{trial}")
            # logging.info(f"trial: {trial}")
            t0 = datetime.now()
            if algo == "direct_softmax":

                sol, time = algos_dict[algo](block=block, block_size=block_size, blocks_num=blocks_num)
            else:
                sol, time = algos_dict[algo](block=block, block_size=block_size, blocks_num=blocks_num)
            sols_list.append(sol)
            stats_list.append(extract_stat(block, sol))
            vals_list.append(sol.flatten() @ block @ sol.flatten())
            # logging.info(f"new priority size: {size},max: {max(vals_list)}, min: {min(vals_list)}")
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
        stats_ts = pd.Series(stats_list)
        stats_tss.append(stats_ts)

    vals_df = pd.concat(vals_tss, axis=1)
    times_df = pd.concat(times_tss, axis=1)
    stats_df = pd.concat(stats_tss, axis=1)
    if full_sol in [0.0, 1.0]:
        vals_fname = get_name(root="results_priority", algo=algo, token="vals", blocks_num=blocks_num,
                              block_size=block_size,
                              percents=percents, full_sol=full_sol, random=random)
        times_fname = get_name(root="results_priority", algo=algo, token="times", blocks_num=blocks_num,
                               block_size=block_size,
                               percents=percents, full_sol=full_sol, random=random)
        stats_fname = get_name(root="results_priority", algo=algo, token="stats", blocks_num=blocks_num,
                               block_size=block_size,
                               percents=percents, full_sol=full_sol, random=random)

        vals_df.to_csv(vals_fname)
        times_df.to_csv(times_fname)
        stats_df.to_csv(stats_fname)
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


def run_all_algos(algos: Tuple[str, ...], precentses: Tuple[int, ...], full_sols: Tuple[float, ...], blocks_num: int,
                  block_size: int, random=False, priority: bool = False):
    # algos = ["greedy"]

    for percents in precentses:
        for full_sol in full_sols:

            if random:
                blocks = examples.create_random_batch(blocks_num=blocks_num, block_size=block_size,
                                                      epsilon=percents / 100,
                                                      batch_size=10, diagonal_blocks=True)
            else:
                blocks, stats = read_data(blocks_num=blocks_num, block_size=block_size, percents=percents,
                                          diag=True)
            if priority:
                blocks = examples.add_priorities(blocks=blocks, blocks_num=blocks_num, block_size=block_size)
            if full_sol > 0:
                for ind, block in enumerate(blocks):
                    blocks[ind] = examples.add_sol_to_data(block=block, blocks_num=blocks_num, block_size=block_size,
                                                           sol=full_sol)

            for algo in algos:
                logging.info(
                    f"blocks num: {blocks_num}, block_size: {block_size}, random: {random}, algo: {algo}, percents: {percents}, full_sol: {full_sol}")

                run_algo(algo=algo, blocks=blocks, blocks_num=blocks_num, block_size=block_size, percents=percents,
                         trials_num=1000,
                         full_sol=full_sol,
                         max_time=120, random=random)


def summurize_res(blocks_num: int, block_size: int, random: bool):
    algos = ["greedy", "direct_softmax", "simplex"]
    precentses = [5, 10, 20, 30, 50]
    full_sols = [True, False]

    stats = []
    for percents in precentses:
        for full_sol in full_sols:
            for algo in algos:
                fname = get_name(root=f"results_priority", algo=algo, token="vals", blocks_num=blocks_num,
                                 block_size=block_size, percents=percents, full_sol=full_sol, random=random)
                if os.path.exists(fname):
                    df = pd.read_csv(fname, index_col=0)

                    get_quantiles(algo, blocks_num, df, full_sol, percents, stats)
    stat_df = pd.DataFrame(stats, columns=["algo", "percents", "full_sol", "med_half", "med_30", "med_60", "med_600",
                                           "mean_half", "mean_30", "mean_60", "mean_600"])
    stat_df = stat_df.sort_values(["percents", "full_sol", "algo"])
    stat_df.to_csv(
        f"time_results{'_random' if random else ''}/stats_{blocks_num}_{block_size}{'_random' if random else ''}.csv")


def get_quantiles(algo: str, blocks_num: int, df: pd.DataFrame, full_sol: bool, percents: int, stats: list):
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
                        fname = get_name(root="results_priority", algo=algo, token="times", blocks_num=blocks_num,
                                         block_size=block_size, percents=percents, full_sol=full_sol, random=random)
                        if os.path.exists(fname):
                            df = pd.read_csv(fname, index_col=0)
                            m = df.unstack().sort_values().iloc[3:-3].mean()
                            logging.info(f"fname: {fname}, mean: {m:.5f}")
                            d[(blocks_num, block_size, full_sol, percents, algo)] = m
    return d


def get_pririty(blocks_num: int, block_size: int) -> np.ndarray:
    return np.repeat(np.random.rand(blocks_num), repeats=block_size)


def analyze_priority(blocks_num, block_size, random):
    algos = ["greedy", "direct_softmax", "simplex"]
    precentses = [5, 10, 20, 30, 50]
    full_sols = [True, False]

    stats = []
    for percents in precentses:
        for full_sol in full_sols:
            for algo in algos:
                fname = get_name(root=f"results_priority", algo=algo, token="stats", blocks_num=blocks_num,
                                 block_size=block_size, percents=percents, full_sol=full_sol, random=random)
                if os.path.exists(fname):
                    df = pd.read_csv(fname, index_col=0)

                    # df = df.astype(str)
                    try:
                        df = df.applymap(lambda x: list(map(int, x[1:-1].split(','))) if x is not np.nan else [np.nan] * 4)
                    except:
                        print(df)
                    vals_tss = []
                    for name, ts in df.iteritems():
                        category_df = pd.DataFrame(zip(*(ts.values))).T
                        vals = category_df @ np.arange(1, category_df.shape[1] + 1)
                        vals_tss.append(pd.Series(vals))
                    vals_df = pd.concat(vals_tss, axis=1)
                    get_quantiles(algo, blocks_num, vals_df, full_sol, percents, stats)
    stat_df = pd.DataFrame(stats, columns=["algo", "percents", "full_sol", "med_half", "med_30", "med_60", "med_600",
                                           "mean_half", "mean_30", "mean_60", "mean_600"])
    stat_df = stat_df.sort_values(["percents", "full_sol", "algo"])
    stat_df.to_csv(
        f"results_priority{'_random' if random else ''}/stats_{blocks_num}_{block_size}{'_random' if random else ''}.csv")


if __name__ == "__main__":
    set_log()

    # profiler = cProfile.Profile()
    #
    # d = get_dict()
    #
    algos = ("greedy", "direct_softmax", "simplex")
    # #algos = ("direct_softmax",)
    precentses = (5, 10, 20, 30, 50)
    # #precentses = (5,)
    # # precentses = (5,)
    full_sols = (0.0, 1.0)
    priotity = True
    random = False
    blocks_num = 100
    block_size = 20
    # # priority = get_pririty(blocks_num=blocks_num, block_size=block_size)
    # # profiler.enable()
    run_all_algos(algos=algos, precentses=precentses, full_sols=full_sols, blocks_num=blocks_num, block_size=block_size,
                  random=random, priority=priotity)

    # summurize_res(blocks_num=100, block_size=20, random=True)
    analyze_priority(blocks_num=100, block_size=20, random=False)
    #summurize_res(blocks_num=100, block_size=20, random=False)
