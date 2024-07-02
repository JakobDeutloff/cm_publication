# %% import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from src.read_data import load_cre

# %% load data
cre_binned, cre_interpolated, cre_interpolated_average = load_cre()
ds = xr.open_dataset(
    "/work/bm1183/m301049/nextgems_profiles/cycle3/interp_representative_sample.nc"
)
ds_monsoon = xr.open_dataset("/work/bm1183/m301049/nextgems_profiles/monsoon/raw_data_converted.nc")

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

# %% plot cloud occurence vs IWP percentiles
fig, axes = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[2, 1], sharex=True)

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

# plot CRE
axes[1].axhline(0, color="grey", linestyle="--")
axes[1].plot(
    cre_interpolated_average.IWP,
    cre_interpolated_average["connected_sw"],
    label="SW",
    color="blue",
)
axes[1].plot(
    cre_interpolated_average.IWP,
    cre_interpolated_average["connected_lw"],
    label="LW",
    color="red",
)
axes[1].plot(
    cre_interpolated_average.IWP,
    cre_interpolated_average["connected_net"],
    label="Net",
    color="k",
)
axes[1].plot(np.linspace(1 - 6, cre_interpolated_average.IWP.min(), 100), np.zeros(100), color="k")
axes[1].set_ylabel("HCRE / W m$^{-2}$")
axes[1].set_xlabel("Ice Water Path / kg m$^{-2}$")

# add colorbar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.41, 0.02, 0.48])
fig.colorbar(cf, cax=cax, label="Cloud Cover")

# add legend for axes[1]
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(
    labels=labels,
    handles=handles,
    bbox_to_anchor=(0.92, 0.3),
    frameon=False,
)

# format axes
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1e-6, 10)
    ax.set_xticks([1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ax.set_xscale("log")

fig.savefig("plots/paper/cloud_profile.png", dpi=500, bbox_inches="tight")


# %%
