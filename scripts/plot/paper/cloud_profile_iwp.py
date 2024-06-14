# %% import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from src.read_data import load_cre
import pickle

# %%
def control_plot(ax):
    ax.set_xlim(1e-5, 10)
    ax.set_xticks([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ax.spines[["top", "right"]].set_visible(False)

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

# %% bin by IWP and average
IWP_bins = np.logspace(-6, 2, 70)
IWP_points = (IWP_bins[:-1] + np.diff(IWP_bins)) / 2
ds_binned = ds.groupby_bins("IWP", IWP_bins).mean(["stacked_time_cell"])

# %% find the two model levels closest to temperature and interpolate the pressure_lev coordinate between them
temps = [273.15]
levels = pd.DataFrame(index=ds_binned.IWP_bins.values, columns=temps)
for temp in temps:
    for iwp in ds_binned.IWP_bins.values:
        try:
            ta_vals = ds_binned.sel(IWP_bins=iwp, pressure_lev=slice(10000, 100000))["temperature"]
            temp_diff = np.abs(ta_vals - temp)
            level1_idx = temp_diff.argmin("pressure_lev").values
            level1 = ta_vals.pressure_lev.values[level1_idx]
            if ta_vals.sel(pressure_lev=level1) > temp:
                level2 = ta_vals.pressure_lev.values[level1_idx - 1]
                level_interp = np.interp(
                    temp,
                    [
                        ta_vals.sel(pressure_lev=level2).values,
                        ta_vals.sel(pressure_lev=level1).values,
                    ],
                    [level2, level1],
                )
            else:
                level2 = ta_vals.pressure_lev.values[level1_idx + 1]
                level_interp = np.interp(
                    temp,
                    [
                        ta_vals.sel(pressure_lev=level1).values,
                        ta_vals.sel(pressure_lev=level2).values,
                    ],
                    [level1, level2],
                )
            levels.loc[iwp, temp] = level_interp
        except:
            levels.loc[iwp, temp] = np.nan

# %% calculate IWP Hist
n_cells = len(ds_monsoon.lat) * len(ds_monsoon.lon)
hist, edges = np.histogram(
    ds_monsoon["IWP"].where(ds_monsoon["mask_height"]), bins=cre_interpolated_average["IWP_bins"]
)
hist = hist / n_cells


# %% plot cloud occurence vs IWP percentiles
fig, axes = plt.subplots(3, 1, figsize=(9, 10), height_ratios=[2, 1, 1], sharex=True)

# plot cloud fraction
cf = axes[0].contourf(
    IWP_points,
    ds.pressure_lev / 100,
    ds_binned["cf"].T,
    cmap="Blues",
    levels=np.arange(0.1, 1.1, 0.1),
)
axes[0].plot(IWP_points, levels[273.15].values / 100, color="grey", linestyle="--")
axes[0].text(2e-6, 590, "0°C", color="grey", fontsize=11)
axes[0].invert_yaxis()
axes[0].set_ylabel("Pressure / hPa")
axes[0].set_yticks([1000, 600, 200])

# plot CRE
axes[1].axhline(0, color="grey", linestyle="--")
axes[1].plot(
    cre_interpolated_average.IWP,
    cre_interpolated_average["connected_sw"],
    label="SW",
    color="blue",
    linestyle="-",
)
axes[1].plot(
    cre_interpolated_average.IWP,
    cre_interpolated_average["connected_lw"],
    label="LW",
    color="red",
    linestyle="-",
)
axes[1].plot(
    cre_interpolated_average.IWP,
    cre_interpolated_average["connected_net"],
    label="Net",
    color="k",
    linestyle="-",
)
axes[1].plot(np.linspace(1 - 6, cre_interpolated_average.IWP.min(), 100), np.ones(100)*cre_interpolated_average['connected_net'][0].values, color="k")
axes[1].set_ylabel("HCRE / W m$^{-2}$")

# plot IWP dist
axes[2].stairs(hist, edges, label="IWP", color="black")
axes[2].set_xscale("log")
axes[2].set_ylabel("P")

# add colorbar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.84, 0.54, 0.02, 0.34])
cb = fig.colorbar(cf, cax=cax, label="Cloud Cover")
cb.set_ticks([0.1, 0.4, 0.7, 1])

# add legend for axes[1]
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    labels=labels,
    handles=handles,
    bbox_to_anchor=(0.92, 0.468),
    frameon=False,
)

# format axes
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1e-6, 10)
    ax.set_xticks([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ax.set_xscale("log")

axes[2].set_xlabel("Ice Water Path / kg m$^{-2}$")

fig.savefig("plots/paper/cloud_profile_iwp.png", dpi=500, bbox_inches="tight")

# %% calculate numbers for text 

# max net CRE 
max_net_cre = cre_interpolated_average["connected_net"].max().values
iwp_max_net_cre = cre_interpolated_average.IWP[cre_interpolated_average["connected_net"].argmax()].values
print(f"Max net CRE: {max_net_cre} at {iwp_max_net_cre}")
#max IWP distribution
iwp_max_hist = cre_interpolated_average.IWP[hist.argmax()].values
print(f"Max IWP distribution: {iwp_max_hist}")
# net HCRE at max IWP distribution
net_hcre_max_hist = cre_interpolated_average["connected_net"][hist.argmax()].values
print(f"Net HCRE at max IWP distribution: {net_hcre_max_hist}")

# %% plot just CRE and IWP dist for poster 
fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

# CRE 
axes[0].axhline(0, color="grey", linestyle="--")
axes[0].plot(
    cre_interpolated_average.IWP,
    cre_interpolated_average["connected_sw"],
    label="SW",
    color="blue",
    linestyle="--",
)
axes[0].plot(
    cre_interpolated_average.IWP,
    cre_interpolated_average["connected_lw"],
    label="LW",
    color="red",
    linestyle="--",
)
axes[0].plot(
    cre_interpolated_average.IWP,
    cre_interpolated_average["connected_net"],
    label="Net",
    color="k",
    linestyle="--",
)
axes[0].plot(np.linspace(1 - 6, cre_interpolated_average.IWP.min(), 100), np.zeros(100), color="k")
axes[0].plot([], [], color="grey", linestyle="--", label="ARTS")
axes[0].plot([], [], color="grey", linestyle="-", label="Concept")
axes[0].set_ylabel("HCRE / W m$^{-2}$")  

# IWP dist
axes[1].stairs(hist, edges, label="IWP", color="black")
axes[1].set_xscale("log")
axes[1].set_ylabel("P")
axes[1].set_xlabel("Ice Water Path / kgm$^{-2}$")

for ax in axes:
    control_plot(ax)

# add legend 
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    labels=labels,
    handles=handles,
    bbox_to_anchor=(0.9, -0.02),
    ncols=5
)

#fig.savefig("plots/paper/cre_weighting_talk_without.png", dpi=500, bbox_inches="tight")

# %%
