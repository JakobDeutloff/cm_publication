# %% import
import numpy as np
import matplotlib.pyplot as plt
from src.read_data import load_cre
import pandas as pd
import xarray as xr
import pickle

# %% load data
cre_binned, cre_mean = load_cre()
ds_monsoon = xr.open_dataset("/work/bm1183/m301049/iwp_framework/mons/data/full_snapshot_proc.nc")
path = "/work/bm1183/m301049/iwp_framework/mons/model_output/"
run = "prefinal"
with open(path + run + ".pkl", "rb") as f:
    result = pickle.load(f)

# %% plot mean CRE vs IWP
IWP_points = cre_mean["IWP"]
fig, axes = plt.subplots(1, 2, sharey="row", figsize=(10, 2.5), sharex="row")
plt.rcParams.update({"font.size": 8})

# plot C for LWP < 1e-4
def plot_cre(ax, modus):
    ax.axhline(0, color="grey", linestyle="--")
    ax.plot(
        IWP_points,
        cre_mean[f"{modus}_sw"],
        label="SW",
        color="blue",
        linestyle="-",
    )
    ax.plot(
        IWP_points,
        cre_mean[f"{modus}_lw"],
        label="LW",
        color="red",
    )
    ax.plot(
        IWP_points,
        cre_mean[f"{modus}_net"],
        label="Net",
        color="k",
    )

modes = ['no_lc', 'ice_over_lc']
for ax, mode in zip(axes, modes):
    plot_cre(ax, mode)


labels = ["a", "b"]
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlabel("$I$ / kg m$^{-2}$")
    ax.set_xlim(1e-5, 1e0)
    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticks([-200, 200])
    ax.text(
        0.1,
        1.12,
        labels.pop(0),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )


axes[0].set_ylabel(r"$C(I)$ / W m$^{-2}$")
axes[0].set_title("LWP < $10^{-4}$ kg m$^{-2}$")
axes[1].set_title("LWP > $10^{-4}$ kg m$^{-2}$")

# legend outside of axes
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.18))
fig.savefig("plots/paper/CREs.png", dpi=300, bbox_inches="tight")

# %% calculate cumulative HCRE for table
n_cells = len(ds_monsoon.lat) * len(ds_monsoon.lon)

hist, edges = np.histogram(
    ds_monsoon["IWP"].where(ds_monsoon["mask_height"]), bins=cre_mean["IWP_bins"]
)
hist = hist / n_cells
weighted_sw = cre_mean["connected_sw"] * hist
weighted_lw = cre_mean["connected_lw"] * hist
weighted_net = weighted_sw + weighted_lw
weighted_sw_cm = result["SW_cre"] * hist
weighted_lw_cm = result["LW_cre"] * hist
weighted_net_cm = weighted_sw_cm + weighted_lw_cm

table_cre = pd.DataFrame(
    index=["ARTS",  "Concept"], columns=["SW", "LW", "Net", "Net Percent", "SW percent", "LW percent"]
)
table_cre.loc["ARTS"]["SW"] = np.round(
    float((cre_mean["connected_sw"] * hist).sum().values), 2
)
table_cre.loc["ARTS"]["LW"] = np.round(
    float((cre_mean["connected_lw"] * hist).sum().values), 2
)
table_cre.loc["ARTS"]["Net"] = (
    table_cre.loc["ARTS"]["SW"] + table_cre.loc["ARTS"]["LW"]
)
table_cre.loc["Concept"]["SW"] = np.round(float((result["SW_cre"] * hist).sum()), 2)
table_cre.loc["Concept"]["LW"] = np.round(float((result["LW_cre"] * hist).sum()), 2)
table_cre.loc["Concept"]["Net"] = table_cre.loc["Concept"]["SW"] + table_cre.loc["Concept"]["LW"]
table_cre.loc["Concept"]["Net Percent"] = np.round(
    ((table_cre.loc["Concept"]["Net"] - table_cre.loc["ARTS"]["Net"])/table_cre.loc['ARTS']['Net']) * 100, 2
)
table_cre.loc["Concept"]["SW percent"] = np.round(
    ((table_cre.loc["Concept"]["SW"] - table_cre.loc["ARTS"]["SW"])/table_cre.loc['ARTS']['SW']) * 100, 2
)
table_cre.loc["Concept"]["LW percent"] = np.round(
    ((table_cre.loc["Concept"]["LW"] - table_cre.loc["ARTS"]["LW"])/table_cre.loc['ARTS']['LW']) * 100, 2
)
table_cre

# %%
