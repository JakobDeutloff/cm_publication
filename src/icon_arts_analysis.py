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

def cut_data(data, mask=True):
    return data.sel(lat=slice(-30, 30)).where(mask)

def cut_data_mixed(data_cs, data_lc, mask, connected):
    # returns lc data fro not connnected profiles and cs data for connected profiles at mask
    data = xr.where(connected == 0 & mask, x=data_lc.where(mask), y=data_cs.where(mask)).sel(lat=slice(-30, 30))
    return data

def define_connected(atms, frac_no_cloud=0.05, rain=True):
    """
    defines for all profiles with ice above liquid whether
    the high and low clouds are connected (1) or not (0).
    Profiles where not both cloud types are present are filled with nan. 
    Profiles masked aout in atm will also be nan.

    Parameters:
    -----------
    atms : xarray.Dataset
        Dataset containing atmospheric profiles, can be masked if needed
    frac_no_cloud : float
        Fraction of maximum cloud condensate in column to define no cloud

    Returns:
    --------
    connected : xarray.DataArray
        DataArray containing connectedness for each profile
    """

    # define liquid and ice cloud condensate
    if rain:
        liq = atms["LWC"] + atms["rain"]
    else:
        liq = atms["LWC"]
        
    ice = atms["IWC"] + atms["snow"] + atms["graupel"]

    # define ice and liquid content needed for connectedness
    no_ice_cloud = (ice > (frac_no_cloud * ice.max("pressure"))) * 1
    no_liq_cloud = (liq > (frac_no_cloud * liq.max("pressure"))) * 1
    no_cld = no_liq_cloud + no_ice_cloud


    # find all profiles with ice above liquid
    mask_both_clds = (atms['LWP'] > 1e-6) & (atms['IWP'] > 1e-6)
    n_profiles = int((mask_both_clds * 1).sum().values)
    lon, lat = np.meshgrid(mask_both_clds.lon, mask_both_clds.lat)
    lat_valid = lat[mask_both_clds]
    lon_valid = lon[mask_both_clds]

    # loop over all profiles with ice above liquid and define connectedness
    connected = xr.DataArray(np.ones(mask_both_clds.shape), coords=mask_both_clds.coords, dims=mask_both_clds.dims)
    connected = connected.where(mask_both_clds)
    for i in range(n_profiles):
        liq_point = liq.sel(lat=lat_valid[i], lon=lon_valid[i])
        ice_point = ice.sel(lat=lat_valid[i], lon=lon_valid[i])
        p_top_idx = ice_point.argmax("pressure").values
        p_bot_idx = liq_point.argmax("pressure").values
        cld_range = no_cld.sel(lat=lat_valid[i], lon=lon_valid[i]).isel(
            pressure=slice(p_bot_idx, p_top_idx)
        )
        # high and low clouds are not connected if there is a 3-cell deep layer without cloud
        for j in range(len(cld_range.pressure)):
            if cld_range.isel(pressure=j).sum() == 0:
                connected.loc[dict(lat=lat_valid[i], lon=lon_valid[i])] = 0
                break

    return connected