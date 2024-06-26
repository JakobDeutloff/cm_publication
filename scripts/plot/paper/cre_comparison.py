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
run = "frozen_only"
with open(path + run + ".pkl", "rb") as f:
    result = pickle.load(f)

# %% plot mean CRE vs IWP
IWP_points = cre_mean["IWP"]
fig, axes = plt.subplots(1, 3, sharey="row", figsize=(10, 2.5), sharex="row")
plt.rcParams.update({"font.size": 8})


def plot_cre(ax, mode, iwp):
    ax.axhline(0, color="grey", linestyle="--")
    ax.plot(
        iwp,
        cre_mean[mode + "_net"],
        label="Net",
        color="k",
    )
    ax.plot(
        iwp,
        cre_mean[mode + "_sw"],
        label="SW",
        color="blue",
    )
    ax.plot(
        iwp,
        cre_mean[mode + "_lw"],
        label="LW",
        color="r",
    )


# ice only
cre_mean["ice_only_net"][IWP_points > 2.3] = np.nan
cre_mean["ice_only_sw"][IWP_points > 2.3] = np.nan
cre_mean["ice_only_lw"][IWP_points > 2.3] = np.nan
plot_cre(axes[0], mode="ice_only", iwp=IWP_points)

# ice over low clouds
plot_cre(axes[1], mode="ice_over_lc", iwp=IWP_points)

# all high clouds
plot_cre(axes[2], mode="all", iwp=IWP_points)

labels = ["a", "b", "c"]
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlabel("$I$ / kg m$^{-2}$")
    ax.set_xlim(1e-5, 1e1)
    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
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

axes[0].set_ylabel(r"$C_{\mathrm{frozen}}(I)$ / W m$^{-2}$")
axes[0].set_title("LWP < $10^{-4}$ kg m$^{-2}$")
axes[1].set_title("LWP > $10^{-4}$ kg m$^{-2}$")
axes[2].set_title("All LWP")

# legend outside of axes
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.18))
#fig.savefig("plots/paper/CREs.png", dpi=300, bbox_inches="tight")

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
    index=["ARTS Conn", "ARTS No Conn", "Concept"], columns=["SW", "LW", "Net", "Net Percent"]
)
table_cre.loc["ARTS Conn"]["SW"] = np.round(
    float((cre_mean["connected_sw"] * hist).sum().values), 2
)
table_cre.loc["ARTS Conn"]["LW"] = np.round(
    float((cre_mean["connected_lw"] * hist).sum().values), 2
)
table_cre.loc["ARTS Conn"]["Net"] = (
    table_cre.loc["ARTS Conn"]["SW"] + table_cre.loc["ARTS Conn"]["LW"]
)
table_cre.loc["ARTS No Conn"]["SW"] = np.round(
    float((cre_mean["all_sw"] * hist).sum().values), 2
)
table_cre.loc["ARTS No Conn"]["LW"] = np.round(
    float((cre_mean["all_lw"] * hist).sum().values), 2
)
table_cre.loc["ARTS No Conn"]["Net"] = (
    table_cre.loc["ARTS No Conn"]["SW"] + table_cre.loc["ARTS No Conn"]["LW"]
)
table_cre.loc["ARTS No Conn"]["Net Percent"] = np.round(
    ((table_cre.loc["ARTS No Conn"]["Net"] - table_cre.loc["ARTS Conn"]["Net"])/table_cre.loc['ARTS Conn']['Net']) * 100, 0
)
table_cre.loc["Concept"]["SW"] = np.round(float((result["SW_cre"] * hist).sum()), 2)
table_cre.loc["Concept"]["LW"] = np.round(float((result["LW_cre"] * hist).sum()), 2)
table_cre.loc["Concept"]["Net"] = table_cre.loc["Concept"]["SW"] + table_cre.loc["Concept"]["LW"]
table_cre.loc["Concept"]["Net Percent"] = np.round(
    ((table_cre.loc["Concept"]["Net"] - table_cre.loc["ARTS Conn"]["Net"])/table_cre.loc['ARTS Conn']['Net']) * 100, 0
)
table_cre

# %%
