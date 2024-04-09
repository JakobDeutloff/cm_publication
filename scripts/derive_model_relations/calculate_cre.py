# %% import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.calc_variables import calc_cre, bin_and_average_cre
from src.read_data import load_atms_and_fluxes, load_derived_vars
import os

# %% load  data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()

# %% calculate cre
fluxes_toa = fluxes_3d.isel(pressure=-1)
fluxes_toa_noice = fluxes_3d_noice.isel(pressure=-1)
cre_clearsky = xr.Dataset(coords={"lat": atms.lat, "lon": atms.lon})
cre_noice = cre_clearsky.copy()

cre_clearsky = calc_cre(fluxes_toa, mode="clearsky")
cre_noice = calc_cre(fluxes_toa, fluxes_toa_noice, mode="noice")

# %% calculate cre in bins and interpolate
IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
lon_bins = np.linspace(-180, 180, num=36)
lon_points = (lon_bins[1:] + lon_bins[:-1]) / 2
cre_binned = {}
cre_interpolated = {}
cre_interpolated_average = {}

# %% all clouds - ice over liquid
cre_binned["all"], cre_interpolated["all"], cre_interpolated_average["all"] = bin_and_average_cre(
    cre_noice.where(lw_vars["mask_height"]).sel(lat=slice(-30, 30)),
    IWP_bins,
    lon_bins,
    atms,
    modus="all",
)

# %% all clouds - any cloud over surface
cre_binned["cre"], cre_interpolated["cre"], cre_interpolated_average["cre"] = bin_and_average_cre(
    cre_clearsky.where(lw_vars["mask_height"]).sel(lat=slice(-30, 30)),
    IWP_bins,
    lon_bins,
    atms,
    modus="all",
)
# %% high cloud with no low coud below
cre_binned["ice_only"], cre_interpolated["ice_only"], cre_interpolated_average["ice_only"] = (
    bin_and_average_cre(
        cre_noice.where(lw_vars["mask_height"]).sel(lat=slice(-30, 30)),
        IWP_bins,
        lon_bins,
        atms,
        modus="ice_only",
    )
)
# %% high cloud over low cloud
(
    cre_binned["ice_over_lc"],
    cre_interpolated["ice_over_lc"],
    cre_interpolated_average["ice_over_lc"],
) = bin_and_average_cre(
    cre_noice.where(lw_vars["mask_height"]).sel(lat=slice(-30, 30)),
    IWP_bins,
    lon_bins,
    atms,
    modus="ice_over_lc",
)
# %% hcre including connectedness of clouds - where connected all clouds are removed from cre
cre_binned["connected"], cre_interpolated["connected"], cre_interpolated_average["connected"] = (
    bin_and_average_cre(
        cre=cre_clearsky.where(atms["connected"] == 1, cre_noice)
        .where(lw_vars["mask_height"])
        .sel(lat=slice(-30, 30)),
        IWP_bins=IWP_bins,
        lon_bins=lon_bins,
        atms=atms,
        modus="all",
    )
)
# %% plot interpolated mean CRE
fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey="row")

pcol = axes[0, 0].pcolor(
    IWP_bins,
    lon_bins,
    cre_binned["ice_only"]["sw"].T,
    cmap="seismic",
    vmin=-600,
    vmax=600,
)
axes[0, 0].set_ylabel("Longitude [deg]")
axes[0, 0].set_title("CRE binned ice only")
axes[0, 1].pcolor(
    IWP_bins,
    lon_bins,
    cre_interpolated["ice_only"]["sw"].T,
    cmap="seismic",
    vmin=-600,
    vmax=600,
)
axes[0, 1].set_title("CRE interpolated ice only")
axes[1, 0].pcolor(
    IWP_bins,
    lon_bins,
    cre_binned["connected"]["sw"].T,
    cmap="seismic",
    vmin=-600,
    vmax=600,
)
axes[1, 0].set_ylabel("Longitude [deg]")
axes[1, 0].set_xlabel("IWP [kg m$^{-2}$]")
axes[1, 0].set_title("CRE binned all clouds")
axes[1, 1].pcolor(
    IWP_bins,
    lon_bins,
    cre_interpolated["connected"]["sw"].T,
    cmap="seismic",
    vmin=-600,
    vmax=600,
)
axes[1, 1].set_xlabel("IWP [kg m$^{-2}$]")
axes[1, 1].set_title("CRE interpolated all clouds")

for ax in axes.flatten():
    ax.set_xscale("log")

fig.colorbar(
    pcol,
    label="High Cloud Radiative Effect",
    location="bottom",
    ax=axes[:],
    shrink=0.7,
    extend="min",
    pad=0.1,
)

fig.savefig("plots/cre/CRE_binned_by_IWP_and_lon.png", dpi=300, bbox_inches="tight")

# %% build dataset of CREs and save it
cre_xr = xr.Dataset()
cre_xr["net_clearsky"] = cre_clearsky["net"]
cre_xr["net_noice"] = cre_noice["net"]
cre_xr["sw_clearsky"] = cre_clearsky["sw"]
cre_xr["sw_noice"] = cre_noice["sw"]
cre_xr["lw_clearsky"] = cre_clearsky["lw"]
cre_xr["lw_noice"] = cre_noice["lw"]

cre_binned_xr = xr.Dataset()
cre_binned_xr["all_sw"] = xr.DataArray(
    data=cre_binned["all"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["all_lw"] = xr.DataArray(
    data=cre_binned["all"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["all_net"] = xr.DataArray(
    data=cre_binned["all"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["ice_only_sw"] = xr.DataArray(
    data=cre_binned["ice_only"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["ice_only_lw"] = xr.DataArray(
    data=cre_binned["ice_only"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["ice_only_net"] = xr.DataArray(
    data=cre_binned["ice_only"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["ice_over_lc_sw"] = xr.DataArray(
    data=cre_binned["ice_over_lc"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["ice_over_lc_lw"] = xr.DataArray(
    data=cre_binned["ice_over_lc"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["ice_over_lc_net"] = xr.DataArray(
    data=cre_binned["ice_over_lc"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["all_clouds_net"] = xr.DataArray(
    data=cre_binned["cre"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["all_clouds_sw"] = xr.DataArray(
    data=cre_binned["cre"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["all_clouds_lw"] = xr.DataArray(
    data=cre_binned["cre"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["connected_net"] = xr.DataArray(
    data=cre_binned["connected"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["connected_sw"] = xr.DataArray(
    data=cre_binned["connected"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr["connected_lw"] = xr.DataArray(
    data=cre_binned["connected"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_binned_xr = cre_binned_xr.assign_coords({"IWP_bins": IWP_bins, "lon_bins": lon_bins})

cre_interpolated_xr = xr.Dataset()
cre_interpolated_xr["all_sw"] = xr.DataArray(
    data=cre_interpolated["all"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["all_lw"] = xr.DataArray(
    data=cre_interpolated["all"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["all_net"] = xr.DataArray(
    data=cre_interpolated["all"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["ice_only_sw"] = xr.DataArray(
    data=cre_interpolated["ice_only"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["ice_only_lw"] = xr.DataArray(
    data=cre_interpolated["ice_only"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["ice_only_net"] = xr.DataArray(
    data=cre_interpolated["ice_only"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["ice_over_lc_sw"] = xr.DataArray(
    data=cre_interpolated["ice_over_lc"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["ice_over_lc_lw"] = xr.DataArray(
    data=cre_interpolated["ice_over_lc"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["ice_over_lc_net"] = xr.DataArray(
    data=cre_interpolated["ice_over_lc"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["all_clouds_net"] = xr.DataArray(
    data=cre_interpolated["cre"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["all_clouds_sw"] = xr.DataArray(
    data=cre_interpolated["cre"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["all_clouds_lw"] = xr.DataArray(
    data=cre_interpolated["cre"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["connected_net"] = xr.DataArray(
    data=cre_interpolated["connected"]["net"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["connected_sw"] = xr.DataArray(
    data=cre_interpolated["connected"]["sw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr["connected_lw"] = xr.DataArray(
    data=cre_interpolated["connected"]["lw"], coords={"IWP": IWP_points, "lon": lon_points}
)
cre_interpolated_xr = cre_interpolated_xr.assign_coords(
    {"IWP_bins": IWP_bins, "lon_bins": lon_bins}
)

cre_interpolated_average_xr = xr.Dataset()
cre_interpolated_average_xr["all_sw"] = xr.DataArray(
    data=cre_interpolated_average["all"]["sw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["all_lw"] = xr.DataArray(
    data=cre_interpolated_average["all"]["lw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["all_net"] = xr.DataArray(
    data=cre_interpolated_average["all"]["net"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["ice_only_sw"] = xr.DataArray(
    data=cre_interpolated_average["ice_only"]["sw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["ice_only_lw"] = xr.DataArray(
    data=cre_interpolated_average["ice_only"]["lw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["ice_only_net"] = xr.DataArray(
    data=cre_interpolated_average["ice_only"]["net"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["ice_over_lc_sw"] = xr.DataArray(
    data=cre_interpolated_average["ice_over_lc"]["sw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["ice_over_lc_lw"] = xr.DataArray(
    data=cre_interpolated_average["ice_over_lc"]["lw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["ice_over_lc_net"] = xr.DataArray(
    data=cre_interpolated_average["ice_over_lc"]["net"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["all_clouds_net"] = xr.DataArray(
    data=cre_interpolated_average["cre"]["net"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["all_clouds_sw"] = xr.DataArray(
    data=cre_interpolated_average["cre"]["sw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["all_clouds_lw"] = xr.DataArray(
    data=cre_interpolated_average["cre"]["lw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["connected_net"] = xr.DataArray(
    data=cre_interpolated_average["connected"]["net"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["connected_sw"] = xr.DataArray(
    data=cre_interpolated_average["connected"]["sw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr["connected_lw"] = xr.DataArray(
    data=cre_interpolated_average["connected"]["lw"], coords={"IWP": IWP_points}
)
cre_interpolated_average_xr = cre_interpolated_average_xr.assign_coords({"IWP_bins": IWP_bins})

path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"
os.remove(path + "cre.nc")
os.remove(path + "cre_binned.nc")
os.remove(path + "cre_interpolated.nc")
os.remove(path + "cre_interpolated_average.nc")
cre_xr.to_netcdf(path + "cre.nc")
cre_binned_xr.to_netcdf(path + "cre_binned.nc")
cre_interpolated_xr.to_netcdf(path + "cre_interpolated.nc")
cre_interpolated_average_xr.to_netcdf(path + "cre_interpolated_average.nc")


# %%
