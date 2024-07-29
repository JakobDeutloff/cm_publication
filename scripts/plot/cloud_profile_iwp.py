# %% import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from src.read_data import load_cre, load_icon_snapshot, get_data_path


# %%
def control_plot(ax):
    ax.set_xlim(1e-5, 10)
    ax.set_xticks([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ax.spines[["top", "right"]].set_visible(False)


# %% load data
cre_binned, cre_mean = load_cre()
path = get_data_path()
ds = xr.open_dataset(path + "data/interp_cf.nc")
ds_monsoon = load_icon_snapshot()

# %% bin by IWP and average
IWP_bins_cf = np.logspace(-5, np.log10(30), 50)
IWP_bins_cre = np.logspace(-5, 1, 50)
IWP_points_cf = (IWP_bins_cf[:-1] + np.diff(IWP_bins_cf)) / 2
IWP_points_cre = (IWP_bins_cre[:-1] + np.diff(IWP_bins_cre)) / 2
ds_binned = ds.groupby_bins("IWP", IWP_bins_cf).mean()

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
    ds_monsoon["IWP"].where(ds_monsoon["mask_height"]), bins=IWP_bins_cre
)
hist = hist / n_cells


# %% plot cloud occurence vs IWP percentiles
fig, axes = plt.subplots(4, 1, figsize=(10, 10), height_ratios=[3, 2, 1, 1], sharex=True)

# plot cloud fraction
cf = axes[0].contourf(
    IWP_points_cf,
    ds.pressure_lev / 100,
    ds_binned["cf"].T,
    cmap="Blues",
    levels=np.arange(0.1, 1.1, 0.1),
)
axes[0].plot(IWP_points_cf, levels[273.15].values / 100, color="grey", linestyle="--")
axes[0].text(2e-5, 560, "0°C", color="grey", fontsize=11)
axes[0].invert_yaxis()
axes[0].set_ylabel("Pressure / hPa")
axes[0].set_yticks([1000, 600, 200])

# plot CRE
axes[1].axhline(0, color="grey", linestyle="--")
axes[1].plot(
    cre_mean.IWP,
    cre_mean["connected_sw"],
    label="SW",
    color="blue",
    linestyle="-",
)
axes[1].plot(
    cre_mean.IWP,
    cre_mean["connected_lw"],
    label="LW",
    color="red",
    linestyle="-",
)
axes[1].plot(
    cre_mean.IWP,
    cre_mean["connected_net"],
    label="net",
    color="k",
    linestyle="-",
)
axes[1].plot(
    np.linspace(1 - 6, cre_mean.IWP.min(), 100),
    np.ones(100) * cre_mean["connected_net"][0].values,
    color="k",
)
axes[1].set_ylabel("$C(I)$ / W m$^{-2}$")
axes[1].set_yticks([-200, 0, 200])

# plot IWP dist
axes[2].stairs(hist, edges, label="IWP", color="black")
axes[2].set_ylabel("$P(I)$")
axes[2].set_yticks([0, 0.02])

# plot P time C 
P_times_C = hist * cre_mean["connected_net"]
axes[3].stairs(P_times_C, IWP_bins_cre, color="k", fill=True, alpha=0.5)
axes[3].stairs(P_times_C, IWP_bins_cre, color="k")
axes[3].set_ylabel("$P(I) ~ \cdot ~ C(I)$ / W m$^{-2}$")

# add colorbar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.84, 0.59, 0.02, 0.3])
cb = fig.colorbar(cf, cax=cax, label="Cloud Cover")
cb.set_ticks([0.1, 0.4, 0.7, 1])

# add legend for axes[1]
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    labels=labels,
    handles=handles,
    bbox_to_anchor=(0.9, 0.52),
    frameon=False,
)

# format axes
labels = ["a", "b", "c", "d"]
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1e-5, 10)
    ax.set_xticks([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ax.set_xscale("log")
    ax.text(
        0.05,
        1.12,
        labels.pop(0),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )

axes[3].set_xlabel("$I$ / kg m$^{-2}$")
fig.savefig("plots/cloud_profile_iwp_mons.png", dpi=500, bbox_inches="tight")

# %% calculate numbers for text

# max net CRE
max_net_cre = cre_mean["connected_net"].max().values
iwp_max_net_cre = cre_mean.IWP[cre_mean["connected_net"].argmax()].values
print(f"Max net CRE: {max_net_cre} at {iwp_max_net_cre}")
# max IWP distribution
iwp_max_hist = cre_mean.IWP[hist.argmax()].values
print(f"Max IWP distribution: {iwp_max_hist}")
# net HCRE at max IWP distribution
net_hcre_max_hist = cre_mean["connected_net"][hist.argmax()].values
print(f"Net HCRE at max IWP distribution: {net_hcre_max_hist}")


# %%
