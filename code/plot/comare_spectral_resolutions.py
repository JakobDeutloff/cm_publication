# %% import 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# %% load high res and low res data
path = "/work/bm1183/m301049/icon_arts_processed/"
low_res = "fullrange_flux_test1deg"
high_res = "fullrange_flux_mid1deg"

fluxes_3d_low = xr.open_dataset(path + low_res + "/fluxes_3d.nc")
fluxes_3d_high = xr.open_dataset(path + high_res + "/fluxes_3d.nc")
atms = xr.open_dataset(path + high_res + "/atms_full.nc")
# %% define functions for cre calculation

def calc_cre(fluxes_toa):
    cre = xr.Dataset(coords={"lat": fluxes_toa.lat, "lon": fluxes_toa.lon})

    # clearsky and allsky from normal simulations
    cre["net"] = (
    fluxes_toa["allsky_sw_down"]
    + fluxes_toa["allsky_sw_up"]
    + fluxes_toa["allsky_lw_down"]
    + fluxes_toa["allsky_lw_up"]
    - (fluxes_toa["clearsky_sw_down"] + fluxes_toa["clearsky_sw_up"])
    - (fluxes_toa["clearsky_lw_down"] + fluxes_toa["clearsky_lw_up"])
    )

    cre["sw"] = (
    fluxes_toa["allsky_sw_down"]
    + fluxes_toa["allsky_sw_up"]
    - (fluxes_toa["clearsky_sw_down"] + fluxes_toa["clearsky_sw_up"])
    )

    cre["lw"] = (
    fluxes_toa["allsky_lw_down"]
    + fluxes_toa["allsky_lw_up"]
    - (fluxes_toa["clearsky_lw_down"] + fluxes_toa["clearsky_lw_up"])
    )

    return cre

def interpolate(data):
    non_nan_indices = np.array(np.where(~np.isnan(data)))
    non_nan_values = data[~np.isnan(data)]
    nan_indices = np.array(np.where(np.isnan(data)))

    interpolated_values = griddata(
        non_nan_indices.T, non_nan_values, nan_indices.T, method="linear"
    )

    copy = data.copy()
    copy[np.isnan(data)] = interpolated_values
    return copy


def bin_and_average_cre(cre, IWP_bins, lon_bins, atms, modus="ice_only"):

    if modus == "ice_only":
        mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)
    elif modus == "ice_over_lc":
        mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] > 1e-10)
    else:
        mask_hc_no_lc = True

    dummy = np.zeros([len(IWP_bins) - 1, len(lon_bins) - 1])
    cre_arr = {"net": dummy.copy(), "sw": dummy.copy(), "lw": dummy.copy()}

    for i in range(len(IWP_bins) - 1):
        IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i + 1])
        for j in range(len(lon_bins) - 1):
            lon_mask = (atms.lon > lon_bins[j]) & (
                atms.lon <= lon_bins[j + 1]
            )

            cre_arr["net"][i, j] = float(
                (cre["net"].where(IWP_mask & lon_mask & mask_hc_no_lc))
                .mean()
                .values
            )
            cre_arr["sw"][i, j] = float(
                (cre["sw"].where(IWP_mask & lon_mask & mask_hc_no_lc))
                .mean()
                .values
            )
            cre_arr["lw"][i, j] = float(
                (cre["lw"].where(IWP_mask & lon_mask & mask_hc_no_lc))
                .mean()
                .values
            )

    # Interpolate
    interp_cre = {
        "net": cre_arr["net"].copy(),
        "sw": cre_arr["sw"].copy(),
        "lw": cre_arr["lw"].copy(),
    }
    for key in interp_cre.keys():
        interp_cre[key] = interpolate(cre_arr[key])

    # average over lat
    for key in interp_cre.keys():
        interp_cre[key] = np.nanmean(interp_cre[key], axis=1)

    return interp_cre

# %% calculate cre

cre_low = calc_cre(fluxes_3d_low.isel(pressure=-1))
cre_high = calc_cre(fluxes_3d_high.isel(pressure=-1))

IWP_bins = np.logspace(-5, 2, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
lon_bins = np.linspace(-180, 180, num=36)

interp_cre_low = bin_and_average_cre(cre_low, IWP_bins, lon_bins, atms, modus="ice_only")
interp_cre_high = bin_and_average_cre(cre_high, IWP_bins, lon_bins, atms, modus="ice_only")

# %% plot cre and differences 
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

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
# %%
