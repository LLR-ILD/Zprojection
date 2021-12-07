from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Make the nu DataFrame available.
sys.path.append(str(Path(__file__).parent / "ZH"))

from higgsstrahlung.from_root import RootfileHandler
import higgsstrahlung.cuts as cuts

rf = "2020-10-08-235437"
rf = Path.home() / "data/ZH/DataFrames" / rf
assert rf.exists()
assert rf.is_absolute()
file_handler = RootfileHandler(tupleName="nuTree", root_folder=rf)
nus = file_handler._getDataFrame(None)
nus = nus[nus.process == "nnh"]
# nus = file_handler._getDataFrame("nnh")


default_var_bins = [
    ("mMiss", np.arange(0, 250, 5)),
    ("mVis", np.arange(0, 180, 5)),
    ("cosTMiss", np.linspace(-1, 1, 50)),
]

def plot_WW_vs_ZH_1D(var_bins=None):
    if var_bins is None:
        var_bins = default_var_bins

    zh = nus[nus.pol == "eRpL"]
    nunuh = nus[nus.pol == "eLpR"]

    nrows = 2
    fx = 7 * len(var_bins)
    fy = 4 * nrows
    fig, axs = plt.subplots(ncols=len(var_bins), nrows=nrows, figsize=(fx, fy))

    for i, (var, bins) in enumerate(var_bins):
        zh_counts, _ = np.histogram(zh[var], bins=bins)
        zh_pdf = zh_counts / zh_counts.sum()
        ww_left_fraction = 0.208
        zh_pdf = zh_pdf * (1 - ww_left_fraction)

        left_counts, _ = np.histogram(nunuh[var], bins=bins)
        left_pdf = left_counts / left_counts.sum()
        ww_pdf = left_pdf - zh_pdf

        x = (bins[1:] + bins[:-1]) / 2
        w = (bins[1:] - bins[:-1])

        axs[0][i].bar(x, zh_pdf, w, bottom=ww_pdf, label="all nnh")
        axs[0][i].bar(x, ww_pdf, w, label="WW-fusion")
        axs[0][i].legend(title="expected counts (arbitrary norm)")
        axs[0][i].set_xlabel(var)

        axs[1][i].bar(x, zh_pdf / zh_pdf.sum(), w, label="ZH nnh", alpha=.8)
        axs[1][i].bar(x, ww_pdf / ww_pdf.sum(), w, label="WW-fusion", alpha=.8)
        axs[1][i].legend(title="pdf")
        axs[1][i].set_xlabel(var)

    return fig
