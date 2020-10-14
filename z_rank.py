import numpy as np
import pandas as pd

try:
    from higgsstrahlung.config import zRankPath
except ModuleNotFoundError:
    from pathlib import Path
    def zRankPath(filename):
        path = Path() / "data/z_rank" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    print("The higgsstrahlung module could not be loaded. The alternative "
         f"zRankPath is {zRankPath}.")

z_cols =  {"mZ", "mRecoil", "abs(cosTZ)", "abs(cosTMiss)"}

def getPreparedZPart(in_df, z_cols):
    """ We change the dataframes. Thus, use copies.
    """
    df = in_df.copy()

    abs_cols = [c.replace("abs(", "")[:-1] for c in z_cols if "abs(" in c]
    for to_abs in abs_cols:
        if f"abs({to_abs})" not in df.columns and to_abs in df.columns:
            df[f"abs({to_abs})"] = abs(df[to_abs])

    if not set(z_cols).issubset(df.columns):
        raise Exception(f"Missing column. Required: {z_cols}. \n"
                        f"Given: {df.columns}.")
    # df = df[list(z_cols) + ["weight"]]
    return df


def createZRanking(sig_df, all_df, steps, print_every=100, save_as=None):
    """
    """
    if print_every is not False:
        print("Create the Z ranking...")
    tmp_sig = getPreparedZPart(sig_df, z_cols)
    tmp_all = getPreparedZPart(all_df, z_cols)

    # Prepare the information storage.
    cut_levels = []
    old_cut = {}
    for z_var in z_cols:
        old_cut[f"{z_var} <= "] = max(tmp_sig[z_var])
        if "abs(" not in z_var:
            old_cut[f"{z_var} >= "] = min(tmp_sig[z_var])

    n_signal = tmp_sig.weight.sum()

    if isinstance(steps, int): steps = np.arange(0, 1, 1/steps)
    if max(steps > 1) or min(steps < 0): raise Exception()
    for i, fraction_removed in enumerate(steps):
        min_remaining_signal = n_signal * (1 - fraction_removed)
        purity, new_cut = {}, {}
        for z_var in z_cols:
            order = tmp_sig.weight.iloc[tmp_sig[z_var].argsort()].cumsum()

            # <=
            idx_efficiency_at_least = order.searchsorted(
                min_remaining_signal, side="left")
            if idx_efficiency_at_least == len(order):
                idx_efficiency_at_least -= 1
            val = tmp_sig[z_var].loc[order.index[idx_efficiency_at_least]]
            key = f"{z_var} <= "
            new_cut[key] = val
            purity[key] = (tmp_sig[tmp_sig[z_var] <= val].weight.sum()
                         / tmp_all[tmp_all[z_var] <= val].weight.sum())

            if "abs(" not in z_var:  # >=
                idx_efficiency_at_least = order.searchsorted(
                    tmp_sig.weight.sum() - min_remaining_signal, side="right")
                val = tmp_sig[z_var].loc[order.index[idx_efficiency_at_least]]
                key = f"{z_var} >= "
                new_cut[key] = val
                purity[key] = (tmp_sig[tmp_sig[z_var] >= val].weight.sum()
                             / tmp_all[tmp_all[z_var] >= val].weight.sum())

        best_additional_cut = max(purity, key=lambda k: purity[k])
        val = new_cut[best_additional_cut]
        old_cut[best_additional_cut] = val

        if " >= " in best_additional_cut:
            var = best_additional_cut.split(" >= ")[0]
            tmp_all = tmp_all[tmp_all[var] >= val]
            tmp_sig = tmp_sig[tmp_sig[var] >= val]
        elif  " <= " in best_additional_cut:
            var = best_additional_cut.split(" <= ")[0]
            tmp_all = tmp_all[tmp_all[var] <= val]
            tmp_sig = tmp_sig[tmp_sig[var] <= val]
        else:
            raise Exception(best_additional_cut)

        old_cut["eff"] = tmp_sig.weight.sum() / n_signal
        old_cut["pur"] = purity[best_additional_cut]
        cut_levels.append(old_cut.copy())
        if (print_every is not False) and (not i%print_every):
            print(f"{best_additional_cut:>17}{val:>6.2f}, "
                  f"eff={old_cut['eff']*100:>6.2f} %, "
                  f"pur={old_cut['pur']*100:>6.2f} %, "
                  f"{min_remaining_signal:>9.4f}")

    ranking_table = pd.DataFrame(cut_levels)
    if save_as is not None:
        ranking_table.to_pickle(zRankPath(f"{save_as}.pkl"))
    return ranking_table


def getZRank(df, ranking_table):
    """Return a value in the range [0, 1] that's composed from the Z decay
    dependent variables.

    df: pd.DataFrame of events that contains the Z decay dependent columns.
    ranking_table: pd.DataFrame that defines the cut-ranking. Should be of the
        form that is provided by `createZRanking`.
    """
    if not isinstance(ranking_table, pd.DataFrame):
        try:
            ranking_table = pd.read_pickle(zRankPath(f"{ranking_table}.pkl"))
        except:
            raise Exception(f"{ranking_table=}")

    ranking_columns = ranking_table.columns.drop(["eff", "pur"],
                                                 errors="ignore")

    ev = getPreparedZPart(df, z_cols)
    # for var in z_cols:
    # ev = getPreparedZPart(df, {"abs(cosTZ)", "abs(cosTMiss)"})
    for var in ev.columns:
        for key, search_side in [(f"{var} >= ", "left"),
                                 (f"{var} <= ", "right")
        ]:
            if key not in ranking_columns:
                continue
            if " >= " in key:
                ev[key] = ranking_table[key].searchsorted(
                          ev[var], side=search_side)
            elif " <= " in key: # Searchsorted needs ascending order -> negate.
                ev[key] = (-ranking_table[key]).searchsorted(
                           -ev[var], side=search_side)
            else:
                raise Exception(key)

    ev["z_rank"] = ev[ranking_columns].values.min(axis=1) / len(ranking_table)
    return ev["z_rank"]