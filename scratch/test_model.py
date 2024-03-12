# %% import
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt

# %% load freddis data
path = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg_noice/"
atms = xr.open_dataset(path + run + "atms_full.nc")
fluxes_3d = xr.open_dataset(path + run + "fluxes_3d_full.nc")
lw_vars = xr.open_dataset("data/lw_vars.nc")
cre = xr.open_dataset("data/cre_interpolated_average.nc")

# %% load coeffs of albedo and emissivity and cre
with open("data/fitted_albedo.pkl", "rb") as f:
    albedo_coeffs = pickle.load(f)

with open("data/fitted_emissivity.pkl", "rb") as f:
    emissivity_coeffs = pickle.load(f)

with open("data/fitted_correction_factor.pkl", "rb") as f:
    correction_coeffs = pickle.load(f)


# %% define functions of albedo and emissivity
def hc_albedo(IWP):
    fitted_vals = np.poly1d(albedo_coeffs)(np.log10(IWP))
    return fitted_vals


def hc_emissivity(IWP):
    fitted_vals = np.poly1d(emissivity_coeffs)(np.log10(IWP))
    return fitted_vals

def hc_correction_factor(IWP):
    fitted_vals = np.poly1d(correction_coeffs)(np.log10(IWP))
    return fitted_vals


def hc_sw_cre(IWP, albedo_s, SW_in):
    return -hc_albedo(IWP) * (1 - albedo_s) * SW_in


def hc_lw_cre(IWP, T_h, R_t):
    sigma = 5.67e-8
    return hc_emissivity(IWP) * (R_t - sigma * T_h**4) + hc_correction_factor(IWP)

# %% define IWP bins 
IWP_bins = np.logspace(-5, 0, num=50)
IWP_points = (IWP_bins[:-1] + IWP_bins[1:]) / 2

# %% find profiles with high clouds and no low clouds below and above 8 km
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)
mask_height = ~atms["h_cloud_top_pressure"].isnull()

# %%  temperature as function of IWP 
T_h_of_iwp =  np.zeros(len(IWP_bins)-1)
for i in range(len(IWP_bins)-1):
    T_h_of_iwp[i] = (
        atms["h_cloud_temperature"]
        .where(mask_hc_no_lc & mask_height & (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i+1]))
        .sel(lat=slice(-30, 30))
        .mean()
        .values
    )

# %% plot temperature as function of IWP    
fig, ax = plt.subplots(figsize=(6, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.plot(IWP_points, T_h_of_iwp, color='k')
ax.set_xscale("log")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("High-Cloud Temperature / K")

# %% calculate LW radiation from below high clouds
R_t = (
    fluxes_3d["clearsky_lw_up"]
    .where(mask_hc_no_lc & mask_height)
    .isel(pressure=-1)
    .sel(lat=slice(-30, 30))
).mean().values * -1


# %% calculate mean albedo below clouds
alpha_s = (
    atms["clearsky_albedo"]
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30))
    .mean()
    .values
)

# %% calculate mean SW in
SW_in = (
    fluxes_3d["clearsky_sw_down"]
    .isel(pressure=-1)
    .sel(lat=slice(-30, 30))
    .mean()
    .values
)

# %% calculate CRE with model
sw_cre = hc_sw_cre(IWP_points, alpha_s, SW_in)
lw_cre = hc_lw_cre(IWP_points, T_h_of_iwp, R_t)

# %% plot CRE
fig, ax = plt.subplots(figsize=(6, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.plot(IWP_points, sw_cre, label="SW CRE Model", color="blue")
ax.plot(IWP_points, lw_cre, label="LW CRE Model", color="red")
ax.plot(IWP_points, sw_cre + lw_cre, label="Net CRE Model", color="k")
ax.plot(cre[0], cre[1], label="Net CRE", color="k", linestyle="--")
ax.plot(cre_sw[0], cre_sw[1], label="SW CRE", color="blue", linestyle="--")
ax.plot(cre_lw[0], cre_lw[1], label="LW CRE", color="red", linestyle="--")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)

ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Cloud Radiative Effect / W m$^{-2}$")
ax.set_xscale("log")
ax.legend(loc="lower left")
ax.set_xlim(1e-5, 1)

# %% Cloud top pressure against IWP
fig, ax = plt.subplots()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.scatter(
    atms["IWP"].where(mask_hc_no_lc ).sel(lat=slice(-30, 30)),
    atms["h_cloud_top_pressure"].where(mask_hc_no_lc ).sel(lat=slice(-30, 30)) / 100,
    color="k",
    s=1,
)
ax.set_xscale("log")
ax.invert_yaxis()
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Cloud Top Pressure / hPa")

# %% 
