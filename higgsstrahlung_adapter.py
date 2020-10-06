import numpy as np

import sys
sys.path.append("/home/kunath/iLCSoft/projects/ZH")
from higgsstrahlung.from_root import RootfileHandler

from higgs_only_model import getXGBModel
from z_rank import z_cols, createZRanking, getZRank


def getData(x_range, y_range):

    # Prepare the two necessary data columns.
    e1_sample = RootfileHandler(tupleName="e1Tree")._df
    nu_sample = RootfileHandler(tupleName="nuTree")._df

    model, training_columns = getXGBModel()
    for df in [e1_sample, nu_sample]:
        df["hBDT"] = model.predict_proba(df[training_columns])[:,1]
    # This cut-ranking is a baseline that any BDT definitely must outperform!
    # for df in [e1_sample, nu_sample]:
    #     df["hBDT"] = getZRank(df, ranking_table="nu_eLpR")

    e1_signal = e1_sample[e1_sample.process == "e1e1h"]
    ranking_table = createZRanking(e1_signal, e1_sample, y_range[:-1],
                        print_every=int(len(y_range)/10), save_as="e1_eLpR")
    e1_sample["zRank"] = getZRank(e1_sample, ranking_table)

    e1_signal = e1_sample[e1_sample.process == "e1e1h"]
    nu_signal = nu_sample[nu_sample.process == "nnh"]

    x_edges = np.concatenate((x_range, [1]))
    y_edges = np.concatenate((y_range, [1]))
    s1d, _ = np.histogram(nu_signal.hBDT, bins=x_edges, weights=nu_signal.weight)
    d1d, _ = np.histogram(nu_sample.hBDT, bins=x_edges, weights=nu_sample.weight)
    b1d = d1d - s1d


    s2d, _, _ = np.histogram2d(e1_signal.hBDT, e1_signal.zRank,
                              bins=(x_edges, y_edges), weights=e1_signal.weight)
    d2d, _, _ = np.histogram2d(e1_sample.hBDT, e1_sample.zRank,
                              bins=(x_edges, y_edges), weights=e1_sample.weight)
    b2d = d2d - s2d

    return s1d, b1d, s2d.T, b2d.T