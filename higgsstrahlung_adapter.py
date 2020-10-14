import numpy as np
import pandas as pd
from pathlib import Path

from z_rank import z_cols, createZRanking, getZRank


def getData(x_range, y_range):

    # Prepare the two necessary data columns.
    df_path = Path("data")
    for prefix in ["nu", "e1", "e2"]:
        pickled_df = df_path / f"{prefix}_events.pkl"
        if not pickled_df.exists():
            raise Exception(f"{pickled_df} does not exist. "
                "Run `motivation.ipynb` to build it.")

    nu_sample = pd.read_pickle(df_path / "nu_events.pkl")
    e1_sample = pd.read_pickle(df_path / "e1_events.pkl")
    e2_sample = pd.read_pickle(df_path / "e2_events.pkl")

    e1_signal = e1_sample[e1_sample.process == "e1e1h"]
    ranking_table = createZRanking(e1_signal, e1_sample, y_range[:-1],
                        print_every=int(len(y_range)/10), save_as="e1_eLpR")
    e1_sample["zRank"] = getZRank(e1_sample, ranking_table)

    e2_signal = e2_sample[e2_sample.process == "e2e2h"]
    ranking_table = createZRanking(e2_signal, e2_sample, y_range[:-1],
                        print_every=int(len(y_range)/10), save_as="e2_eLpR")
    e2_sample["zRank"] = getZRank(e2_sample, ranking_table)

    nu_signal = nu_sample[nu_sample.process == "nnh"]
    e1_signal = e1_sample[e1_sample.process == "e1e1h"]
    e2_signal = e2_sample[e2_sample.process == "e2e2h"]

    x_edges = np.concatenate((x_range, [1]))
    y_edges = np.concatenate((y_range, [1]))
    s1d, _ = np.histogram(nu_signal.hBDT, bins=x_edges, weights=nu_signal.weight)
    d1d, _ = np.histogram(nu_sample.hBDT, bins=x_edges, weights=nu_sample.weight)
    b1d = d1d - s1d


    s_e1, _, _ = np.histogram2d(e1_signal.hBDT, e1_signal.zRank,
                              bins=(x_edges, y_edges), weights=e1_signal.weight)
    d_e1, _, _ = np.histogram2d(e1_sample.hBDT, e1_sample.zRank,
                              bins=(x_edges, y_edges), weights=e1_sample.weight)
    b_e1 = d_e1 - s_e1

    s_e2, _, _ = np.histogram2d(e2_signal.hBDT, e2_signal.zRank,
                              bins=(x_edges, y_edges), weights=e2_signal.weight)
    d_e2, _, _ = np.histogram2d(e2_sample.hBDT, e2_sample.zRank,
                              bins=(x_edges, y_edges), weights=e2_sample.weight)
    b_e2 = d_e2 - s_e2

    return s1d, b1d, s_e1.T, b_e1.T, s_e2.T, b_e2.T