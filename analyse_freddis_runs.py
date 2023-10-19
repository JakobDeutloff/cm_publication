# %%
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# %% load data from freddis runs
path_freddi = "/work/bm1183/m301049/freddi_runs/"
atms = xr.open_dataset(path_freddi + "atms.nc")
fluxes_3d = xr.open_dataset(path_freddi + "fluxes_3d.nc")
fluxes_2D = xr.open_dataset(path_freddi + "fluxes_2d.nc")
aux = xr.open_dataset(path_freddi + "aux.nc")

# claculate IWP
IWP = ((atms["IWC"] + atms["snow"] + atms["graupel"]) * atms["geometric height"]).sum(
    "pressure"
)


# %% load cloudsat data for 2015
path_cloudsat = "/work/bm1183/m301049/cloudsat/"
cloudsat = xr.open_dataset(path_cloudsat + "2015-07-01_2016-07-01_fwp.nc")
cloudsat = cloudsat.to_pandas()

# %% select tropics
lat_mask = (cloudsat["lat"] <= 30) & (cloudsat["lat"] >= -30)
cloudsat_trop = cloudsat[lat_mask]

IWP_trop = IWP.sel(lat=slice(-30, 30))

# %% calculate share of zeros
zeros_freddi = (IWP_trop.where(IWP_trop == 0).count() / IWP_trop.count()).values
zeros_cloudsat = (
    cloudsat_trop["ice_water_path"].where(cloudsat_trop["ice_water_path"] == 0).count()
    / cloudsat_trop["ice_water_path"].count()
)

# %% compare IWP histograms
fig, ax = plt.subplots(figsize=(8, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

bins = np.logspace(-5, 4, num=70)

# cloudsat
hist, edges = np.histogram(
    cloudsat_trop["ice_water_path"] * 1e-3, bins=bins, density=False
)
hist_norm = hist / (
    np.diff(edges) * len(cloudsat_trop["ice_water_path"]) * (1 - zeros_cloudsat)
)
ax.stairs(hist_norm, edges, color="blue", label="CloudSat")

# freddi
hist, edges = np.histogram(IWP_trop, bins=bins, density=False)
hist_norm = hist / (np.diff(edges) * len(IWP_trop) * (1 - zeros_freddi))
ax.stairs(hist_norm, edges, color="red", label="FREDDI")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-6, 1e4)
ax.set_ylim(1e-4, 1e8)
ax.legend()
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Probability Density / (kg m$^{-2}$)$^{-1}$")


# %%
