import numpy as np
import xarray as xr
from scipy.interpolate import griddata

def calc_cre(fluxes_toa, fluxes_toa_noice=None, mode="clearsky"):

    cre = xr.Dataset(coords={"lat": fluxes_toa.lat, "lon": fluxes_toa.lon})

    if mode == "clearsky":
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

    elif mode == "noice":
        cre["net"] = (
        fluxes_toa["allsky_sw_down"]
        + fluxes_toa["allsky_sw_up"]
        + fluxes_toa["allsky_lw_down"]
        + fluxes_toa["allsky_lw_up"]
        - (fluxes_toa_noice["allsky_sw_down"] + fluxes_toa_noice["allsky_sw_up"])
        - (fluxes_toa_noice["allsky_lw_down"] + fluxes_toa_noice["allsky_lw_up"])
        )

        cre["sw"] = (
        fluxes_toa["allsky_sw_down"]
        + fluxes_toa["allsky_sw_up"]
        - (fluxes_toa_noice["allsky_sw_down"] + fluxes_toa_noice["allsky_sw_up"])
        )

        cre["lw"] = (
        fluxes_toa["allsky_lw_down"]
        + fluxes_toa["allsky_lw_up"]
        - (fluxes_toa_noice["allsky_lw_down"] + fluxes_toa_noice["allsky_lw_up"])
        )

    else:
        raise ValueError("mode must be either clearsky or noice")
    
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
    interp_cre_average = {}
    for key in interp_cre.keys():
        interp_cre_average[key] = np.nanmean(interp_cre[key], axis=1)

    return cre_arr, interp_cre, interp_cre_average