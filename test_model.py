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

# %% load coeffs of albedo and emissivity and cre
with open("data/fitted_albedo.pkl", "rb") as f:
    albedo_coeffs = pickle.load(f)

with open("data/fitted_emissivity.pkl", "rb") as f:
    emissivity_coeffs = pickle.load(f)

with open("data/hc_cre.pkl", "rb") as f:
    cre = pickle.load(f)

with open("data/hc_cre_sw.pkl", "rb") as f:
    cre_sw = pickle.load(f)

with open("data/hc_cre_lw.pkl", "rb") as f:
    cre_lw = pickle.load(f)


# %% define functions of albedo and emissivity
def hc_albedo(IWP):
    fitted_vals = np.poly1d(albedo_coeffs)(np.log10(IWP))
    fitted_vals[IWP > 1] = np.poly1d(albedo_coeffs)(np.log10(1))
    return fitted_vals


def hc_emissivity(IWP):
    fitted_vals = np.poly1d(emissivity_coeffs)(np.log10(IWP))
    fitted_vals[fitted_vals > 1] = 1
    fitted_vals[IWP > 0.1] = 1
    return fitted_vals


def hc_sw_cre(IWP, albedo_s, SW_in):
    return -hc_albedo(IWP) * (1 - albedo_s) * SW_in


def hc_lw_cre(IWP, T_h, R_t, red):
    sigma = 5.67e-8
    return hc_emissivity(IWP) * (R_t - sigma * T_h**4)


# %% find profiles with high clouds and no low clouds below
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)

# %% calculate mean hc temperature
T_h = (
    atms["h_cloud_temperature_lw"]
    .where(mask_hc_no_lc)
    .sel(lat=slice(-30, 30))
    .mean()
    .values
)
mask_temperature = ~np.isnan(atms["h_cloud_temperature_lw"])

# temperature as function of IWP 
IWP_bins = np.logspace(-5, 1, num=50)
T_h_of_iwp =  np.zeros(len(IWP_bins)-1)
for i in range(len(IWP_bins)-1):
    T_h_of_iwp[i] = (
        atms["h_cloud_temperature_lw"]
        .where(mask_hc_no_lc & (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i+1]))
        .sel(lat=slice(-30, 30))
        .mean()
        .values
    )
fig, ax = plt.subplots()
ax.plot(IWP_bins[:-1], T_h_of_iwp)
ax.set_xscale("log")

# %% claculate reduction above hc
clearsky_lw_out = fluxes_3d["clearsky_lw_up"].mean(["lat", "lon"])
reduction_fraction = (
    clearsky_lw_out - clearsky_lw_out.isel(pressure=-1)
) / clearsky_lw_out
mean_hc_pressure = (
    atms["h_cloud_top_pressure"]
    .where(mask_hc_no_lc)
    .sel(lat=slice(-30, 30))
    .mean()
    .values
)
reduction = (
    1 - reduction_fraction.sel(pressure=mean_hc_pressure, method="nearest").values
)

# %% calculate LW radiation from below high clouds
R_t = (
    fluxes_3d["clearsky_lw_up"]
    .isel(pressure=-1)
    .where(mask_hc_no_lc & mask_temperature)
    .sel(lat=slice(-30, 30))
).mean().values * -1


# %% calculate mean albedo below clouds
alpha_s = (
    atms["clearsky_albedo"]
    .where(mask_hc_no_lc & mask_temperature)
    .sel(lat=slice(-30, 30))
    .mean()
    .values
)

# %% calculate mean surface temperature below clouds
T_s = (
    aux["surface temperature"]
    .where(mask_hc_no_lc & mask_temperature)
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
IWP = (IWP_bins[:-1] + IWP_bins[1:]) / 2
sw_cre = hc_sw_cre(IWP, alpha_s, SW_in)
lw_cre = hc_lw_cre(IWP, T_h_of_iwp, R_t, reduction)

# %% plot CRE
fig, ax = plt.subplots()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.plot(IWP, sw_cre, label="SW CRE Model", color="blue")
ax.plot(IWP, lw_cre, label="LW CRE Model", color="red")
ax.plot(IWP, sw_cre + lw_cre, label="Total CRE Model", color="k")
ax.plot(cre[0], cre[1], label="Total CRE", color="k", linestyle="--")
ax.plot(cre_sw[0], cre_sw[1], label="SW CRE", color="blue", linestyle="--")
ax.plot(cre_lw[0], cre_lw[1], label="LW CRE", color="red", linestyle="--")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.7)

ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Cloud Radiative Effect / W m$^{-2}$")
ax.set_xscale("log")
ax.legend(loc="lower left")

# %% Cloud top pressure against IWP
fig, ax = plt.subplots()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.scatter(
    atms["IWP"].where(mask_hc_no_lc & mask_temperature).sel(lat=slice(-30, 30)),
    atms["h_cloud_top_pressure"].where(mask_hc_no_lc & mask_temperature).sel(lat=slice(-30, 30)) / 100,
    color="k",
    s=1,
)
ax.set_xscale("log")
ax.invert_yaxis()
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Cloud Top Pressure / hPa")

# %% 
