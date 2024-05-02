"""
fuctions used to calculate variables used to parameterise the model and for data analysis
"""

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from tqdm import tqdm


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
            lon_mask = (atms.lon > lon_bins[j]) & (atms.lon <= lon_bins[j + 1])

            cre_arr["net"][i, j] = float(
                (cre["net"].where(IWP_mask & lon_mask & mask_hc_no_lc)).mean().values
            )
            cre_arr["sw"][i, j] = float(
                (cre["sw"].where(IWP_mask & lon_mask & mask_hc_no_lc)).mean().values
            )
            cre_arr["lw"][i, j] = float(
                (cre["lw"].where(IWP_mask & lon_mask & mask_hc_no_lc)).mean().values
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


def calc_connected(atms, frac_no_cloud=0.05, rain=True, convention="icon"):
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

    if (convention == "icon") or (convention == "icon_binned"):
        vert_coord = "level_full"
    elif convention == "arts":
        vert_coord = "pressure"

    ice = atms["IWC"] + atms["snow"] + atms["graupel"]

    # define ice and liquid content needed for connectedness
    no_ice_cloud = (ice > (frac_no_cloud * ice.max(vert_coord))) * 1
    no_liq_cloud = (liq > (frac_no_cloud * liq.max(vert_coord))) * 1
    no_cld = no_liq_cloud + no_ice_cloud

    # find all profiles with ice above liquid
    mask_both_clds = (atms["LWP"] > 1e-4) & (atms["IWP"] > 1e-6)

    # prepare coordinates with liquid clouds below ice clouds for indexing in the loop
    n_profiles = int((mask_both_clds * 1).sum().values)
    if convention == "icon":
        cell, time = np.meshgrid(mask_both_clds.cell, mask_both_clds.time)
        cell_valid = cell[mask_both_clds]
        time_valid = time[mask_both_clds]
    elif convention == "arts":
        lon, lat = np.meshgrid(mask_both_clds.lon, mask_both_clds.lat)
        lat_valid = lat[mask_both_clds]
        lon_valid = lon[mask_both_clds]
    elif convention == "icon_binned":
        time, iwp, profile = np.meshgrid(
            mask_both_clds.local_time_points, mask_both_clds.iwp_points, mask_both_clds.profile
        )
        iwp_valid = iwp[mask_both_clds]
        time_valid = time[mask_both_clds]
        profile_valid = profile[mask_both_clds]

    # create connectedness array
    connected = xr.DataArray(
        np.ones(mask_both_clds.shape), coords=mask_both_clds.coords, dims=mask_both_clds.dims
    )
    connected.attrs = {"units": "1", "long_name": "Connectedness of liquid and frozen clouds"}

    # set all profiles with no liquid cloud to nan
    connected = connected.where(mask_both_clds)

    # loop over all profiles with ice above liquid and check for connectedness
    for i in tqdm(range(n_profiles)):
        if convention == "icon":
            liq_point = liq.sel(cell=cell_valid[i], time=time_valid[i])
            ice_point = ice.sel(cell=cell_valid[i], time=time_valid[i])
            p_top_idx = ice_point.argmax(vert_coord).values
            p_bot_idx = liq_point.argmax(vert_coord).values
            cld_range = no_cld.sel(cell=cell_valid[i], time=time_valid[i]).isel(
                level_full=slice(p_top_idx, p_bot_idx)
            )
            # high and low clouds are not connected if there is a 1-cell deep layer without cloud
            for j in range(len(cld_range.level_full)):
                if cld_range.isel(level_full=j).sum() == 0:
                    connected.loc[dict(cell=cell_valid[i], time=time_valid[i])] = 0
                    break

        elif convention == "arts":
            liq_point = liq.sel(lat=lat_valid[i], lon=lon_valid[i])
            ice_point = ice.sel(lat=lat_valid[i], lon=lon_valid[i])
            p_top_idx = ice_point.argmax(vert_coord).values
            p_bot_idx = liq_point.argmax(vert_coord).values
            cld_range = no_cld.sel(lat=lat_valid[i], lon=lon_valid[i]).isel(
                pressure=slice(p_bot_idx, p_top_idx)
            )
            # high and low clouds are not connected if there is a 1-cell deep layer without cloud
            for j in range(len(cld_range.pressure)):
                if cld_range.isel(pressure=j).sum() == 0:
                    connected.loc[dict(lat=lat_valid[i], lon=lon_valid[i])] = 0
                    break

        elif convention == "icon_binned":
            liq_point = liq.sel(local_time_points=time_valid[i], iwp_points=iwp_valid[i], profile=profile_valid[i])
            ice_point = ice.sel(local_time_points=time_valid[i], iwp_points=iwp_valid[i], profile=profile_valid[i])
            p_top_idx = ice_point.argmax(vert_coord).values
            p_bot_idx = liq_point.argmax(vert_coord).values
            cld_range = no_cld.sel(local_time_points=time_valid[i], iwp_points=iwp_valid[i], profile=profile_valid[i]).isel(
                level_full=slice(p_top_idx, p_bot_idx)
            )
            # high and low clouds are not connected if there is a 1-cell deep layer without cloud
            for j in range(len(cld_range.level_full)):
                if cld_range.isel(level_full=j).sum() == 0:
                    connected.loc[
                        dict(local_time_points=time_valid[i], iwp_points=iwp_valid[i], profile=profile_valid[i])
                    ] = 0
                    break

    return connected


def calc_IWP(atms, convention="icon"):
    """
    Calculate the vertically integrated ice water content.
    """
    if convention == "icon":
        cell_height = atms["dzghalf"]
        vert_coord = "level_full"
    elif convention == "arts":
        cell_height = calc_cell_height(atms)
        vert_coord = "pressure"

    IWP = ((atms["IWC"] + atms["snow"] + atms["graupel"]) * cell_height).sum(vert_coord)
    IWP.attrs = {"units": "kg m^-2", "long_name": "Ice Water Path"}
    return IWP


def calc_LWP(atms, convention="icon"):
    """
    Calculate the vertically integrated liquid water content.
    """
    if convention == "icon":
        cell_height = atms["dzghalf"]
        vert_coord = "level_full"
    elif convention == "arts":
        cell_height = calc_cell_height(atms)
        vert_coord = "pressure"

    LWP = ((atms["rain"] + atms["LWC"]) * cell_height).sum(vert_coord)
    LWP.attrs = {"units": "kg m^-2", "long_name": "Liquid Water Path"}
    return LWP


def calc_cell_height(atms):
    cell_height = atms["geometric height"].diff("pressure")
    cell_height_bottom = xr.DataArray(
        data=cell_height.isel(pressure=0).values,
        dims=["lat", "lon"],
        coords={
            "pressure": atms["pressure"][0],
            "lat": atms["lat"],
            "lon": atms["lon"],
        },
    )
    cell_height = xr.concat([cell_height_bottom, cell_height], dim="pressure")
    return cell_height


def calculate_lc_fraction(atms):
    """
    Calculate the fraction of liquid clouds.
    """
    lc_fraction = (atms["LWP"] > 1e-6) * 1
    return lc_fraction


def calculate_IWC_cumsum(atms, convention="icon"):
    """
    Calculate the vertically integrated ice water content.
    """
    if convention == "icon":
        cell_height = atms["dzghalf"]
        ice_mass = (atms["IWC"] + atms["graupel"] + atms["snow"]) * cell_height
        IWC_cumsum = ice_mass.cumsum('level_full')
    elif convention == "arts":
        cell_height = calc_cell_height(atms)
        # pressure coordinate needs to be reversed for cumsum
        ice_mass = ((atms["IWC"] + atms["graupel"] + atms["snow"]) * cell_height).reindex(
            pressure=list(reversed(atms.pressure))
        )
        IWC_cumsum = ice_mass.cumsum("pressure").reindex(pressure=list(reversed(atms.pressure)))


    IWC_cumsum.attrs = {
        "units": "kg m^-2",
        "long_name": "Cumulative Ice Water Content",
    }
    return IWC_cumsum


def calculate_h_cloud_temperature(atms, IWP_emission=8e-3, convention="icon"):
    """
    Calculate the temperature of high clouds.
    """
    if convention == "icon":
        vert_coord = "level_full"
    elif convention == "arts":
        vert_coord = "pressure"
    top_idx_thin = (atms["IWC"] + atms["snow"] + atms["graupel"]).argmax(vert_coord)
    top_idx_thick = np.abs(atms["IWC_cumsum"] - IWP_emission).argmin(vert_coord)
    top_idx = xr.where(top_idx_thick < top_idx_thin, top_idx_thick, top_idx_thin)
    if convention == "icon":
        top = atms.isel(level_full=top_idx).level_full
        T_h = atms["temperature"].sel(level_full=top)
    elif convention == "arts":
        top = atms.isel(pressure=top_idx).pressure
        T_h = atms["temperature"].sel(pressure=top)
    T_h.attrs = {"units": "K", "long_name": "High Cloud Top Temperature"}
    top.attrs = {"units": "1", "long_name": "Level Index of High CLoud Top"}
    return T_h, top


def calc_dry_air_properties(ds):
    """
    Calculate the properties of dry air based on the given dataset.

    Parameters:
    ds (xarray.Dataset): The dataset containing the necessary variables.

    Returns:
    xarray.Dataset: The dataset with additional variables for dry air properties.
    """

    # Dry Air density
    rho_air = ds.pfull / ds.ta / 287.04
    rho_air.attrs = {"units": "kg/m^3", "long_name": "dry air density"}
    # Dry air specific mass
    dry_air = 1 - (ds.cli + ds.clw + ds.qs + ds.qg + ds.qr + ds.hus)
    dry_air.attrs = {"units": "kg/kg", "long_name": "specific mass of dry air"}

    return rho_air, dry_air


def convert_to_density(ds, key):
    """
    Convert the given variable to density based on the dry air specific mass.

    Parameters:
    ds (xarray.Dataset): The dataset containing the variables.
    var (str): The variable to be converted.

    Returns:
    xarray.Dataset: The dataset with the converted variable.
    """
    var = (ds[key] / ds["dry_air"]) * ds["rho_air"]
    var.attrs["units"] = "kg/m^3"
    return var


def calc_cf(ds):
    """
    Calculate cloud fraction from cloud ice and cloud liquid water content
    If cloud condensate exceeds 10^-6 kg/m^3, it is set to 1, otherwise to 0.
    """
    cf = (
        (ds["IWC"] + ds["LWC"] + ds["rain"] + ds["snow"] + ds["graupel"]) > 5 * 10 ** (-9)
    ).astype(int)
    cf.attrs = {
        "component": "atmo",
        "grid_mapping": "crs",
        "long_name": "cloud_fraction",
        "units": "1/0",
        "vgrid": "reference",
    }
    return cf
