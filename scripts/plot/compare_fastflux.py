# %% import 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.calc_variables import calc_cre, bin_and_average_cre

# %% load high res and low res data
path = "/work/bm1183/m301049/icon_arts_processed/"
fastflux = "fullrange_flux_mid1deg_fast"
slowflux = "fullrange_flux_mid1deg"

fluxes_3d_fast = xr.open_dataset(path + fastflux + "/fluxes_3d_full.nc")
fluxes_3d_slow = xr.open_dataset(path + slowflux + "/fluxes_3d_full.nc")
atms = xr.open_dataset(path + slowflux + "/atms_full.nc")

# %% calculate cre
cre_fast = calc_cre(fluxes_3d_fast.isel(pressure=-1))
cre_slow = calc_cre(fluxes_3d_slow.isel(pressure=-1))

IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
lon_bins = np.linspace(-180, 180, num=36)

interp_cre_fast = bin_and_average_cre(cre_fast, IWP_bins, lon_bins, atms, modus="all")[2]
interp_cre_slow = bin_and_average_cre(cre_slow, IWP_bins, lon_bins, atms, modus="all")[2]

# %% plot cre and differences 
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

# next to each other 
axes[0].plot(IWP_points, interp_cre_fast["net"], label="net cre fast", color='k')
axes[0].plot(IWP_points, interp_cre_slow["net"], label="net cre slow", linestyle="--", color='k')

axes[0].plot(IWP_points, interp_cre_fast["sw"], label="sw cre fast", color='blue')
axes[0].plot(IWP_points, interp_cre_slow["sw"], label="sw cre slow", linestyle="--", color='blue')

axes[0].plot(IWP_points, interp_cre_fast["lw"], label="lw cre fast", color='red')
axes[0].plot(IWP_points, interp_cre_slow["lw"], label="lw cre slow", linestyle="--", color='red')
axes[0].set_ylabel("CRE / Wm$^{-2}$")

# differences 
axes[1].plot(IWP_points, interp_cre_slow["net"] - interp_cre_fast["net"], label="net cre difference", color='k')
axes[1].plot(IWP_points, interp_cre_slow["sw"] - interp_cre_fast["sw"], label="sw cre difference", color='blue')
axes[1].plot(IWP_points, interp_cre_slow["lw"] - interp_cre_fast["lw"], label="lw cre difference", color='red')

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("IWP / kgm$^{-2}$")
    ax.legend()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

fig.tight_layout()
fig.savefig("plots/cre_comparison_resolutions.png", dpi=300)
# %%
