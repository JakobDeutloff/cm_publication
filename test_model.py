# %% import
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt

# %% load freddis data
path_freddi = "/work/bm1183/m301049/freddi_runs/"
atms = xr.open_dataset(path_freddi + "atms_full.nc")
fluxes_3d = xr.open_dataset(path_freddi + "fluxes_3d_full.nc")
fluxes_2d = xr.open_dataset(path_freddi + "fluxes_2d.nc")
aux = xr.open_dataset(path_freddi + "aux.nc")

# %% load coeffs of albedo and emissivity
with open("data/fitted_albedo.pkl", "rb") as f:
    albedo_coeffs = pickle.load(f)

with open("data/fitted_emissivity.pkl", "rb") as f:
    emissivity_coeffs = pickle.load(f)


# %% define functions of albedo and emissivity
def hc_albedo(IWP):
    fitted_vals = np.poly1d(albedo_coeffs)(np.log10(IWP))
    fitted_vals[IWP > 1] = np.poly1d(albedo_coeffs)(np.log10(1))
    return fitted_vals


def hc_emissivity(IWP):
    fitted_vals = np.poly1d(emissivity_coeffs)(np.log10(IWP))
    fitted_vals[fitted_vals > 1] = 1
    fitted_vals[IWP > 0.5] = 1
    return fitted_vals

def sw_cre(IWP, Ts, albedo_s, SW_in):
    return - hc_albedo(IWP) * (1 - albedo_s) * SW_in


# %% find profiles with high clouds and no low clouds below
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)

# %% calculate mean hc temperature
mean_hc_temperature = (
    atms["h_cloud_temperature"]
    .where(mask_hc_no_lc)
    .sel(lat=slice(-30, 30))
    .mean()
    .values
)
mask_temperature = ~np.isnan(atms["h_cloud_temperature"])

# %% calculate LW radiation from below high clouds
LW_up_below_clouds = (
    fluxes_3d["clearsky_lw_up"]
    .isel(pressure=-1)
    .where(mask_hc_no_lc & mask_temperature)
    .sel(lat=slice(-30, 30))
)
mean_LW_up = LW_up_below_clouds.mean().values


# %% plot LW up from below high clouds in scatterplot with IWP

fig, ax = plt.subplots(1, 1)
ax.scatter(
    atms["IWP"].where(mask_hc_no_lc).sel(lat=slice(-30, 30)),
    LW_up_below_clouds,
    s=0.5,
    color="black",
)
ax.set_xscale("log")

# %%
