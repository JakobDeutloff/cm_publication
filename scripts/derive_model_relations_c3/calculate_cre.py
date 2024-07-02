# %% import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.calc_variables import calc_cre

# %% load  data
atms = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/atms_proc.nc")
fluxes_allsky = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/fluxes.nc")
fluxes_noice = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/fluxes_nofrozen.nc")
fluxes_noliquid = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/fluxes_noliquid.nc")

# %% calculate cre
fluxes_allsky_lw = fluxes_allsky['allsky_lw_up'].isel(pressure=-1) + fluxes_allsky['allsky_lw_down'].isel(pressure=-1)
fluxes_allsky_sw = fluxes_allsky['allsky_sw_up'].isel(pressure=-1) + fluxes_allsky['allsky_sw_down'].isel(pressure=-1)
fluxes_clearsky_lw = fluxes_allsky['clearsky_lw_up'].isel(pressure=-1) + fluxes_allsky['clearsky_lw_down'].isel(pressure=-1)
fluxes_clearsky_sw = fluxes_allsky['clearsky_sw_up'].isel(pressure=-1) + fluxes_allsky['clearsky_sw_down'].isel(pressure=-1)
fluxes_noice_lw = fluxes_noice['allsky_lw_up'].isel(pressure=-1) + fluxes_noice['allsky_lw_down'].isel(pressure=-1)
fluxes_noice_sw = fluxes_noice['allsky_sw_up'].isel(pressure=-1) + fluxes_noice['allsky_sw_down'].isel(pressure=-1)
fluxes_no_hc_lw = xr.where(atms["mask_low_cloud"], fluxes_noice_lw, fluxes_clearsky_lw)
fluxes_no_hc_sw = xr.where(atms["mask_low_cloud"], fluxes_noice_sw, fluxes_clearsky_sw)
cre_high_clouds = cre = xr.Dataset(
    coords={
        "iwp_points": fluxes_noice.iwp_points,
        "local_time_points": fluxes_noice.local_time_points,
        "profile": fluxes_noice.profile,
    }
)
cre_high_clouds["sw"] = fluxes_allsky_sw - fluxes_no_hc_sw
cre_high_clouds["lw"] = fluxes_allsky_lw - fluxes_no_hc_lw
cre_high_clouds["net"] = cre_high_clouds["sw"] + cre_high_clouds["lw"]

# %% plot cre for high clouds
fig, ax = plt.subplots()
mean_cre_hc = cre_high_clouds.where(atms["mask_height"] & ~atms['mask_low_cloud']).mean("profile").mean("local_time_points")
ax.plot(mean_cre_hc["iwp_points"], mean_cre_hc["sw"], label="sw hc", color="blue", linestyle="--")
ax.plot(mean_cre_hc["iwp_points"], mean_cre_hc["lw"], label="lw hc", color="red", linestyle="--")
ax.plot(mean_cre_hc["iwp_points"], mean_cre_hc["net"], label="net hc", color="k", linestyle="--")
ax.set_xscale("log")

# %%
fig, ax = plt.subplots()
atms["mask_low_cloud"].mean(["local_time_points", "profile"]).plot(ax=ax)
ax.set_xscale("log")

# %% plot cre in bins
fig, ax = plt.subplots()
mean = cre_high_clouds.mean("profile")
iwp_bins = np.logspace(-6, 0, 51)
local_time_bins = np.linspace(0, 24, 25)
ax.pcolor(
    iwp_bins,
    local_time_bins,
    mean["net"].T,
)
ax.set_xscale("log")

# %% plot sw flux no high clouds in bins 
fig, ax = plt.subplots()
mean_flux = fluxes_no_hc_sw.mean("profile")
ax.pcolor(
    iwp_bins,
    local_time_bins,
    mean_flux.T,
)
ax.set_xscale("log")

# %% save cres
cre_high_clouds.to_netcdf("/work/bm1183/m301049/iwp_framework/ngc3/data/cre_high_clouds.nc")
# %%
