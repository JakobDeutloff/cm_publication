# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib as mpl


# %% def functions
def read_cloudsat(year):
    """
    Function to read CloudSat for a given year
    """

    path_cloudsat = "/work/bm1183/m301049/cloudsat/"
    cloudsat = xr.open_dataset(
        path_cloudsat + year + "-07-01_" + str(int(year) + 1) + "-07-01_fwp.nc"
    )
    # convert ot pandas
    cloudsat = cloudsat.to_pandas()
    # select tropics
    lat_mask = (cloudsat["lat"] <= 30) & (cloudsat["lat"] >= -30)

    return cloudsat[lat_mask]


def plot_hist(year, ax, bins, linestyle, color):
    hist, edges = np.histogram(
        cloudsat[year]["ice_water_path"] * 1e-3, bins=bins, density=False
    )
    hist_norm = hist / (
        np.diff(edges) * len(cloudsat[year]["ice_water_path"]) * (1 - zeros[year])
    )
    ax.stairs(hist_norm, edges, label=year, linestyle=linestyle, color=color)


def calc_hist(year, bins):
    hist, edges = np.histogram(
        cloudsat[year]["ice_water_path"] * 1e-3, bins=bins, density=False
    )
    hist_norm = hist / (
        np.diff(edges) * len(cloudsat[year]["ice_water_path"]) * (1 - zeros[year])
    )
    return hist_norm, edges


def plot_monthly_hists(year, ax, bins, cmap):
    for month in range(1, 13):
        color = cmap(month / 12)
        mask_month = cloudsat[year].time.dt.month == month
        data = cloudsat[year][mask_month]["ice_water_path"] * 1e-3
        hist, edges = np.histogram(data, bins=bins, density=False)
        hist_norm = hist / (np.diff(edges) * len(data))
        ax.stairs(hist_norm, edges, label=month, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-5, 1e2)
        ax.set_ylim(1e-4, 1e3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(str(year))


# %% read cloudsat data for all years
cloudsat = {}
for year in range(2006, 2019):
    cloudsat[str(year)] = read_cloudsat(str(year))

# %% get length and percentage of zeros of annual data
length = {}
zeros = {}
for year in range(2006, 2019):
    length[str(year)] = len(cloudsat[str(year)]["ice_water_path"])
    zeros[str(year)] = (
        np.sum(cloudsat[str(year)]["ice_water_path"] == 0) / length[str(year)]
    )

# %% make histogramms for all years
fig, ax = plt.subplots(figsize=(8, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

bins = np.logspace(-5, 2, num=70)

# create a colormap
cmap = mpl.colormaps["viridis"]
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])

for i, year in enumerate(range(2006, 2019)):
    if (
        year in []
    ):  # years with higher occurences: 2006, 2007, 2008, 2009, 2010, years with less data 2011, 2017
        pass
    else:
        plot_hist(str(year), ax, bins, "solid", color=cmap(i / 13))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-5, 1e2)
ax.set_ylim(1e-4, 1e3)
ax.legend()
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Probability Density / (kg m$^{-2}$)$^{-1}$")

fig.savefig("plots/2c_ice_interannual.png", dpi=300, bbox_inches="tight")

# %% make histogram of individual months for four years
years = [2012, 2013, 2014, 2015]
bins = np.logspace(-5, 2, num=70)
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col", sharey="row")
axes = axes.flatten()

# create a colormap
cmap = mpl.colormaps["viridis"]
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array([])

for i, year in enumerate(years):
    plot_monthly_hists(str(year), axes[i], bins, cmap)


# labels
axes[2].set_xlabel("IWP / kg m$^{-2}$")
axes[3].set_xlabel("IWP / kg m$^{-2}$")
axes[0].set_ylabel("Probability Density / (kg m$^{-2}$)$^{-1}$")
axes[2].set_ylabel("Probability Density / (kg m$^{-2}$)$^{-1}$")

# make legend at right side of figure with one column
fig.subplots_adjust(right=0.87)
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="center right", bbox_to_anchor=(0.95, 0.5), ncol=1)
fig.savefig("plots/2c_ice_monthly.png", dpi=300, bbox_inches="tight")


# %% barplot of percentage of zeros
fig, ax = plt.subplots(figsize=(5, 3))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.bar(x=np.arange(2006, 2019), height=list(zeros.values()), color="grey")
ax.set_ylabel("Share of Zeros")
fig.savefig("plots/2c_ice_zeros.png", dpi=300, bbox_inches="tight")

# %% Barplot of number of profiles
fig, ax = plt.subplots(figsize=(5, 3))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.bar(x=np.arange(2006, 2019), height=list(length.values()), color="grey")
ax.set_ylabel("Number of Profiles")
fig.savefig("plots/2c_ice_n_profiles.png", dpi=300, bbox_inches="tight")

# %% histogram mean and std
bins = np.logspace(-5, 2, num=70)
hist = np.zeros((13, len(bins) - 1))

for i, year in enumerate(range(2006, 2019)):
    hist[i, :], edges = calc_hist(str(year), bins)

hist_mean = hist.mean(axis=0)
hist_std = hist.std(axis=0)

fig, ax = plt.subplots(figsize=(5, 3))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.stairs(hist_mean, edges, color="black", label="Mean")
ax.stairs(
    edges=edges,
    values=hist_mean + 2 * hist_std,
    baseline=hist_mean - 2 * hist_std,
    alpha=0.5,
    label=r"$2 \cdot \sigma $",
    fill=True,
    color="grey",
)


ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-5, 1e2)
ax.set_ylim(1e-4, 1e3)
ax.legend()
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Probability Density / (kg m$^{-2}$)$^{-1}$")
fig.savefig("plots/2c_ice_mean_std.png", dpi=300, bbox_inches="tight")

# %%
