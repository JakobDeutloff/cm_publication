"""
Calculate cloud radiative effect (CRE) for high clouds (connected), high clouds with low clouds (ice_over_lc) 
and high clouds without low clouds (no_lc) and bin and average the CRE by IWP and longitude. 
Save the binned and averaged CRE in a netCDF file.
"""

# %% import
import numpy as np
import xarray as xr
from src.calc_variables import calc_cre, bin_and_average_cre
from src.read_data import load_atms_and_fluxes
import os

# %% load  data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()

# %% calculate cre
fluxes_toa = fluxes_3d.isel(pressure=-1)
fluxes_toa_noice = fluxes_3d_noice.isel(pressure=-1)
cre_all_clouds = xr.Dataset(coords={"lat": atms.lat, "lon": atms.lon})
cre_frozen_clouds = cre_all_clouds.copy()

cre_all_clouds = calc_cre(fluxes_toa, mode="clearsky")
cre_frozen_clouds = calc_cre(fluxes_toa, fluxes_toa_noice, mode="noice")
cre_high_clouds = xr.where(atms["mask_low_cloud"], cre_frozen_clouds, cre_all_clouds)

# %% create bins and initialize dictionaries
IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
lon_bins = np.linspace(-180, 180, num=36)
lon_points = (lon_bins[1:] + lon_bins[:-1]) / 2
cre_binned = {}
cre_interpolated = {}
cre_interpolated_average = {}

# %% bin and interpolate cre
cre_binned["connected"], cre_interpolated["connected"], cre_interpolated_average["connected"] = (
    bin_and_average_cre(
        cre=cre_high_clouds.where(atms["mask_height"]).sel(lat=slice(-30, 30)),
        IWP_bins=IWP_bins,
        lon_bins=lon_bins,
        atms=atms,
    )
)

cre_binned["no_lc"], cre_interpolated["no_lc"], cre_interpolated_average["no_lc"] = (
    bin_and_average_cre(
        cre=cre_high_clouds.where(atms["mask_height"] & ~atms["mask_low_cloud"]).sel(
            lat=slice(-30, 30)
        ),
        IWP_bins=IWP_bins,
        lon_bins=lon_bins,
        atms=atms,
    )
)

(
    cre_binned["ice_over_lc"],
    cre_interpolated["ice_over_lc"],
    cre_interpolated_average["ice_over_lc"],
) = bin_and_average_cre(
    cre=cre_high_clouds.where(atms["mask_height"] & atms["mask_low_cloud"]).sel(lat=slice(-30, 30)),
    IWP_bins=IWP_bins,
    lon_bins=lon_bins,
    atms=atms,
)

# %% build dataset of CREs and save it

cre_binned_xr = xr.Dataset()
cre_binned_xr["connected_net"] = xr.DataArray(
    data=cre_binned["connected"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["connected_sw"] = xr.DataArray(
    data=cre_binned["connected"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["connected_lw"] = xr.DataArray(
    data=cre_binned["connected"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["no_lc_net"] = xr.DataArray(
    data=cre_binned["no_lc"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["no_lc_sw"] = xr.DataArray(
    data=cre_binned["no_lc"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["no_lc_lw"] = xr.DataArray(
    data=cre_binned["no_lc"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["ice_over_lc_net"] = xr.DataArray(
    data=cre_binned["ice_over_lc"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["ice_over_lc_sw"] = xr.DataArray(
    data=cre_binned["ice_over_lc"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["ice_over_lc_lw"] = xr.DataArray(
    data=cre_binned["ice_over_lc"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr = cre_binned_xr.assign_coords({"IWP_bins": IWP_bins, "lon_bins": lon_bins})


cre_interpolated_average_xr = xr.Dataset()
cre_interpolated_average_xr["connected_net"] = xr.DataArray(
    data=cre_interpolated_average["connected"]["net"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["connected_sw"] = xr.DataArray(
    data=cre_interpolated_average["connected"]["sw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["connected_lw"] = xr.DataArray(
    data=cre_interpolated_average["connected"]["lw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["no_lc_net"] = xr.DataArray(
    data=cre_interpolated_average["no_lc"]["net"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["no_lc_sw"] = xr.DataArray(
    data=cre_interpolated_average["no_lc"]["sw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["no_lc_lw"] = xr.DataArray(
    data=cre_interpolated_average["no_lc"]["lw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["ice_over_lc_net"] = xr.DataArray(
    data=cre_interpolated_average["ice_over_lc"]["net"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["ice_over_lc_sw"] = xr.DataArray(
    data=cre_interpolated_average["ice_over_lc"]["sw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["ice_over_lc_lw"] = xr.DataArray(
    data=cre_interpolated_average["ice_over_lc"]["lw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr = cre_interpolated_average_xr.assign_coords({"IWP_bins": IWP_bins})

path = "/work/bm1183/m301049/iwp_framework/mons/data/"
os.remove(path + "cre_binned.nc")
cre_binned_xr.to_netcdf(path + "cre_binned.nc")
os.remove(path + "cre_mean.nc")
cre_interpolated_average_xr.to_netcdf(path + "cre_mean.nc")


# %%
