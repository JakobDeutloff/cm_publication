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

# %% find profiles with high clouds and no low clouds below
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)

# %% aveage over IWP bins 
IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
mean_hc_emissivity = np.zeros(len(IWP_bins) - 1)

for i in range(len(IWP_bins) - 1):
    IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i + 1])
    mean_hc_emissivity[i] = float(
        (
            atms["high_cloud_emissivity"]
            .where(IWP_mask & mask_hc_no_lc)
            .sel(lat=slice(-30, 30))
        )
        .mean()
        .values
    )

# %% fit polynomial to mean emissivity
p = np.polyfit(np.log10(IWP_points[mean_hc_emissivity <= 1]), mean_hc_emissivity[mean_hc_emissivity <= 1], 5)

def hc_emissivity(IWP, coeffs): 
    fitted_vals = np.poly1d(coeffs)(np.log10(IWP))
    fitted_vals[fitted_vals > 1] = 1
    fitted_vals[IWP > 0.5] = 1
    return fitted_vals

fitted_emissivity = hc_emissivity(IWP_points, p)

# %% plot mean hv emissivity in scatterplot with IWP
fig, ax = plt.subplots(1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"]
    .where(mask_hc_no_lc)
    .where(atms["h_cloud_top_pressure"])
    .sel(lat=slice(-30, 30)),
    atms["high_cloud_emissivity"]
    .where(mask_hc_no_lc)
    .where(atms["h_cloud_top_pressure"])
    .sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d["allsky_sw_down"]
    .isel(pressure=-1)
    .where(mask_hc_no_lc)
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)

ax.plot(IWP_points, mean_hc_emissivity, color="red")
ax.plot(IWP_points, fitted_emissivity, color="orange")

cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("High Cloud Emissivity")
ax.set_xscale("log")
ax.set_ylim([0, 2])

# %% save coefficients as pkl file
with open("data/fitted_emissivity.pkl", "wb") as f:
    pickle.dump(p, f)

# %%
