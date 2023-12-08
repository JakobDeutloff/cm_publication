# %% import modules
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# %% load data
path = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg/"
atms = xr.open_dataset(path + run + "atms_full.nc")
cre_binned = xr.open_dataset("data/cre_binned.nc")
cre_interpolated = xr.open_dataset("data/cre_interpolated.nc")

# %% find bins with positive CRE at high SW in 
strange_mask = (
    (cre_binned["all_net"] > 0)
    & (cre_binned["all_net"].IWP > 5e-2)
    & (-70 < cre_binned["all_net"].lon)
    & (cre_binned["all_net"].lon < 70)
)

IWP, lon = np.meshgrid(cre_binned["all_net"].IWP, cre_binned["all_net"].lon)
x_coords = IWP[strange_mask.T]
y_coords = lon[strange_mask.T]

# %% plot CRE adn selected bins
fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey="row")

cre_binned["all_net"].plot.pcolormesh(ax=axes[0], x="IWP", vmin=-300, vmax=300, cmap="RdBu_r", add_colorbar=False)
axes[0].scatter(x_coords, y_coords, color='k', marker='x', alpha=0.5)
axes[0].set_title("All clouds")
axes[0].set_ylabel("Longitude")

cmap = cre_binned["ice_only_net"].plot.pcolormesh(ax=axes[1], x="IWP", vmin=-300, vmax=300, cmap="RdBu_r", add_colorbar=False)
axes[1].scatter(x_coords, y_coords, color='k', marker='x', alpha=0.5)
axes[1].set_title("High clouds no low clouds")
axes[1].set_ylabel("")

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1e2)
    ax.set_xlabel("IWP / kg m$^{-2}$")

fig.tight_layout()
fig.colorbar(cmap, ax=axes, label="CRE / W m$^{-2}$", orientation="horizontal", shrink=0.6, pad=0.2)

# %% 
