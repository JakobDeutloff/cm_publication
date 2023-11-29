# %% import 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle

# %% load freddis data
path_freddi = "/work/bm1183/m301049/freddi_runs/"
atms = xr.open_dataset(path_freddi + "atms_full.nc")
fluxes_3d = xr.open_dataset(path_freddi + "fluxes_3d_full.nc")
fluxes_2d = xr.open_dataset(path_freddi + "fluxes_2d.nc")
aux = xr.open_dataset(path_freddi + "aux.nc")

# %% find profiles with high clouds and no low clouds below and above 8 km
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)
mask_height = ~atms["h_cloud_top_pressure"].isnull()

# %% aveage over IWP bins 
IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
mean_hc_emissivity = np.zeros(len(IWP_bins) - 1)
mean_correction_factor = np.zeros(len(IWP_bins) - 1)

for i in range(len(IWP_bins) - 1):
    IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i + 1])
    mean_hc_emissivity[i] = float(
        (
            atms["high_cloud_emissivity"]
            .where(IWP_mask & mask_hc_no_lc & mask_height)
            .sel(lat=slice(-30, 30))
        )
        .mean()
        .values
    )
    mean_correction_factor[i] = float(
        (
            atms["rad_correction"]
            .where(IWP_mask & mask_hc_no_lc & mask_height)
            .sel(lat=slice(-30, 30))
            .mean()
            .values
        )
    )

# %% fit polynomial to mean emissivity
p_emm = np.polyfit(np.log10(IWP_points[mean_hc_emissivity <= 1]), mean_hc_emissivity[mean_hc_emissivity <= 1], 9)

def hc_emissivity(IWP, coeffs): 
    fitted_vals = np.poly1d(coeffs)(np.log10(IWP))
    return fitted_vals

fitted_emissivity = hc_emissivity(IWP_points, p_emm)

# %% fit polynomial to mean correction factor
p_cor = np.polyfit(np.log10(IWP_points[~np.isnan(mean_correction_factor)]), mean_correction_factor[~np.isnan(mean_correction_factor)], 9)

def hc_correction_factor(IWP, coeffs):
    fitted_vals = np.poly1d(coeffs)(np.log10(IWP))
    return fitted_vals

fitted_correction_factor = hc_correction_factor(IWP_points, p_cor)

# %% plot mean hv emissivity in scatterplot with IWP
fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"]
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30)),
    atms["high_cloud_emissivity"]
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d["allsky_sw_down"]
    .isel(pressure=-1)
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)

ax.plot(IWP_points, mean_hc_emissivity, color="lime", label="Mean Emissivity")
ax.plot(IWP_points, fitted_emissivity, color="r", label="Fitted Polynomial", linestyle='--')
ax.axhline(1, color="grey", linestyle='--')

cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("High Cloud Emissivity")
ax.set_xscale("log")
ax.set_ylim([0, 1.7])
ax.set_xlim(1e-5, 10)
ax.legend()

# %% plot mean correction factior in scatterplot with IWP
fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"]
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30)),
    atms["rad_correction"]
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d["allsky_sw_down"]
    .isel(pressure=-1)
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)

ax.plot(IWP_points, mean_correction_factor, color="lime", label="Mean Correction Factor")
ax.plot(IWP_points, fitted_correction_factor, color="r", label="Fitted Polynomial", linestyle='--')

cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Correction Factor")
ax.set_xscale("log")
ax.legend()

# %% save coefficients as pkl file
with open("data/fitted_emissivity.pkl", "wb") as f:
    pickle.dump(p_emm, f)

with open("data/fitted_correction_factor.pkl", "wb") as f:
    pickle.dump(p_cor, f)

# %%
