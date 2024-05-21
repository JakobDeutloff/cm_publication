# %% import
import numpy as np
import matplotlib.pyplot as plt
from src.read_data import load_cre
import pandas as pd
import xarray as xr
import pickle

# %% load data
cre_binned, cre_interpolated, cre_interpolated_average = load_cre()
ds = xr.open_dataset(
    "/work/bm1183/m301049/nextgems_profiles/cycle3/interp_representative_sample.nc"
)
ds_monsoon = xr.open_dataset("/work/bm1183/m301049/nextgems_profiles/monsoon/raw_data_converted.nc")
path = "/work/bm1183/m301049/cm_results/"
run = "icon_mons_const_lc"
with open(path + run + ".pkl", "rb") as f:
    result = pickle.load(f)

# %% plot mean CRE vs IWP
IWP_points = cre_interpolated_average["IWP"]
fig, axes = plt.subplots(1, 4, sharey="row", figsize=(12, 2.5), sharex='row')
plt.rcParams.update({'font.size': 8})


def plot_cre(ax, mode, iwp):
    ax.plot(
        iwp,
        cre_interpolated_average[mode + "_net"],
        label="Net CRE",
        color="k",
    )
    ax.plot(
        iwp,
        cre_interpolated_average[mode + "_sw"],
        label="SW CRE",
        color="blue",
    )
    ax.plot(
        iwp,
        cre_interpolated_average[mode + "_lw"],
        label="LW CRE",
        color="r",
    )

# ice only
plot_cre(axes[0], mode='ice_only', iwp=IWP_points)

# ice over low clouds
plot_cre(axes[1], mode="ice_over_lc", iwp=IWP_points)

# all high clouds
plot_cre(axes[2], mode="all", iwp=IWP_points)

# considering connectedness
plot_cre(axes[3], mode="connected", iwp=IWP_points)


for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlabel("IWP / kg m$^{-2}$")
    ax.axhline(0, color="k", linestyle="--")
    ax.set_xlim(1e-5, 1e1)
    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])

axes[0].set_ylabel("HCRE / W m$^{-2}$")
axes[0].set_title("(a) High Clouds over Clear Sky")
axes[1].set_title("(b) High Clouds over Low Clouds")
axes[2].set_title("(c) ARTS No Connect")
axes[3].set_title("(d) ARTS Connect")

# legend outside of axes
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.18))
fig.savefig("plots/paper/CREs.png", dpi=300, bbox_inches="tight")

# %% calculate cumulative HCRE for table 
n_cells = len(ds_monsoon.lat) * len(ds_monsoon.lon)

hist, edges = np.histogram(
    ds_monsoon["IWP"].where(ds_monsoon["mask_height"]), bins=cre_interpolated_average["IWP_bins"]
)
hist = hist / n_cells
weighted_sw = cre_interpolated_average["connected_sw"] * hist
weighted_lw = cre_interpolated_average["connected_lw"] * hist
weighted_net = weighted_sw + weighted_lw
weighted_sw_cm = result["SW_cre"] * hist
weighted_lw_cm = result["LW_cre"] * hist
weighted_net_cm = weighted_sw_cm + weighted_lw_cm
 
table_cre = pd.DataFrame(index=['ARTS Conn', 'ARTS No Conn', 'Concept'], columns=['SW', 'LW', 'Net'])
table_cre.loc['ARTS Conn']['SW'] = np.round(float((cre_interpolated_average["connected_sw"] * hist).sum().values), 1)
table_cre.loc['ARTS Conn']['LW'] = np.round(float((cre_interpolated_average["connected_lw"] * hist).sum().values), 1)
table_cre.loc['ARTS Conn']['Net'] = table_cre.loc['ARTS Conn']['SW'] + table_cre.loc['ARTS Conn']['LW']
table_cre.loc['ARTS No Conn']['SW'] = np.round(float((cre_interpolated_average["all_sw"] * hist).sum().values), 1)
table_cre.loc['ARTS No Conn']['LW'] = np.round(float((cre_interpolated_average["all_lw"] * hist).sum().values), 1)
table_cre.loc['ARTS No Conn']['Net'] = table_cre.loc['ARTS No Conn']['SW'] + table_cre.loc['ARTS No Conn']['LW']
table_cre.loc['Concept']['SW'] = np.round(float((result["SW_cre"] * hist).sum()), 1)
table_cre.loc['Concept']['LW'] = np.round(float((result["LW_cre"] * hist).sum()), 1)
table_cre.loc['Concept']['Net'] = table_cre.loc['Concept']['SW'] + table_cre.loc['Concept']['LW']
table_cre

# %%
