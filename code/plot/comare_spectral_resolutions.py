# %% import 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from src.icon_arts_analysis import calc_cre, bin_and_average_cre

# %% load high res and low res data
path = "/work/bm1183/m301049/icon_arts_processed/"
low_res = "fullrange_flux_test1deg"
high_res = "fullrange_flux_mid1deg"

fluxes_3d_low = xr.open_dataset(path + low_res + "/fluxes_3d.nc")
fluxes_3d_high = xr.open_dataset(path + high_res + "/fluxes_3d.nc")
atms = xr.open_dataset(path + high_res + "/atms_full.nc")
# %% define functions for cre calculation

# %% calculate cre

cre_low = calc_cre(fluxes_3d_low.isel(pressure=-1))
cre_high = calc_cre(fluxes_3d_high.isel(pressure=-1))

IWP_bins = np.logspace(-5, 2, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
lon_bins = np.linspace(-180, 180, num=36)

interp_cre_low = bin_and_average_cre(cre_low, IWP_bins, lon_bins, atms, modus="ice_only")
interp_cre_high = bin_and_average_cre(cre_high, IWP_bins, lon_bins, atms, modus="ice_only")

# %% plot cre and differences 
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

# next to each other 
axes[0].plot(IWP_points, interp_cre_low["net"], label="net cre low res", color='k')
axes[0].plot(IWP_points, interp_cre_high["net"], label="net cre high res", linestyle="--", color='k')

axes[0].plot(IWP_points, interp_cre_low["sw"], label="sw cre low res", color='blue')
axes[0].plot(IWP_points, interp_cre_high["sw"], label="sw cre high res", linestyle="--", color='blue')

axes[0].plot(IWP_points, interp_cre_low["lw"], label="lw cre low res", color='red')
axes[0].plot(IWP_points, interp_cre_high["lw"], label="lw cre high res", linestyle="--", color='red')
axes[0].set_ylabel("CRE / W/m$^2$")

# differences 
axes[1].plot(IWP_points, interp_cre_high["net"] - interp_cre_low["net"], label="net cre difference", color='k')
axes[1].plot(IWP_points, interp_cre_high["sw"] - interp_cre_low["sw"], label="sw cre difference", color='blue')
axes[1].plot(IWP_points, interp_cre_high["lw"] - interp_cre_low["lw"], label="lw cre difference", color='red')

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("IWP / kg/m$^2$")
    ax.legend()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

fig.tight_layout()
fig.savefig("plots/cre_comparison_resolutions.png", dpi=300)
# %%
