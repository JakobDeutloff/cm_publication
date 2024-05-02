# %%
import xarray as xr
import numpy as np
from src.calc_variables import calc_connected
import os

# %% load data from freddis runs
path_freddi = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg/"
atms = xr.open_dataset(path_freddi + run + "atms.nc")
fluxes_3d = xr.open_dataset(path_freddi + run + "fluxes_3d.nc")
aux = xr.open_dataset(path_freddi + run + "aux.nc")

# %% change convention of fluxes - down is positive
fluxes = [
    "allsky_sw_down",
    "allsky_sw_up",
    "allsky_lw_down",
    "allsky_lw_up",
    "clearsky_sw_down",
    "clearsky_sw_up",
    "clearsky_lw_down",
    "clearsky_lw_up",
]
for flux in fluxes:
    fluxes_3d[flux] = fluxes_3d[flux] * -1

# %% calculate celll height
cell_height = atms["geometric height"].diff("pressure")
# not correct, we would need height at half levels
# set cell heigth in lowest box to the value above
cell_height_bottom = xr.DataArray(
    data=cell_height.isel(pressure=0).values,
    dims=["lat", "lon"],
    coords={
        "pressure": fluxes_3d["pressure"][0],
        "lat": fluxes_3d["lat"],
        "lon": fluxes_3d["lon"],
    },
)
cell_height = xr.concat([cell_height_bottom, cell_height], dim="pressure")
# %% Calculate LWP and IWP
atms["IWP"] = ((atms["IWC"] + atms["snow"] + atms["graupel"]) * cell_height).sum("pressure")
atms["LWP"] = ((atms["rain"] + atms["LWC"]) * cell_height).sum("pressure")

# ice mass needs to be reversed along pressure coord to calculate cumsum from the toa
ice_mass = ((atms["IWC"] + atms["graupel"] + atms["snow"]) * cell_height).reindex(
    pressure=list(reversed(atms.pressure))
)
atms["IWC_cumsum"] = ice_mass.cumsum("pressure").reindex(pressure=list(reversed(atms.pressure)))

# %% calculate lc fraction
lc_fraction = (atms["LWP"] > 1e-4) * 1
atms["lc_fraction"] = lc_fraction

# %% calculate heating rates from fluxes (vertical levels are not quite correct)
g = 9.81
cp = 1005
seconds_per_day = 24 * 60 * 60
p = fluxes_3d["pressure"]
p_half = (p[1:].values + p[:-1].values) / 2
fluxes_3d = fluxes_3d.assign_coords(p_half=p_half)

allsky_hr_lw = (
    (g / cp)
    * (
        (fluxes_3d["allsky_lw_up"] + fluxes_3d["allsky_lw_down"]).diff("pressure")
        / fluxes_3d["pressure"].diff("pressure")
    )
    * seconds_per_day
)
allsky_hr_lw["pressure"] = p_half
allsky_hr_lw = allsky_hr_lw.rename({"pressure": "p_half"})
fluxes_3d["allsky_hr_lw"] = allsky_hr_lw

clearsky_hr_lw = (
    (g / cp)
    * (
        (fluxes_3d["clearsky_lw_up"] + fluxes_3d["clearsky_lw_down"]).diff("pressure")
        / fluxes_3d["pressure"].diff("pressure")
    )
    * seconds_per_day
)
clearsky_hr_lw["pressure"] = p_half
clearsky_hr_lw = clearsky_hr_lw.rename({"pressure": "p_half"})
fluxes_3d["clearsky_hr_lw"] = clearsky_hr_lw

allsky_hr_sw = (
    (g / cp)
    * (
        (fluxes_3d["allsky_sw_up"] + fluxes_3d["allsky_sw_down"]).diff("pressure")
        / fluxes_3d["pressure"].diff("pressure")
    )
    * seconds_per_day
)
allsky_hr_sw["pressure"] = p_half
allsky_hr_sw = allsky_hr_sw.rename({"pressure": "p_half"})
fluxes_3d["allsky_hr_sw"] = allsky_hr_sw

clearsky_hr_sw = (
    (g / cp)
    * (
        (fluxes_3d["clearsky_sw_up"] + fluxes_3d["clearsky_sw_down"]).diff("pressure")
        / fluxes_3d["pressure"].diff("pressure")
    )
    * seconds_per_day
)
clearsky_hr_sw["pressure"] = p_half
clearsky_hr_sw = clearsky_hr_sw.rename({"pressure": "p_half"})
fluxes_3d["clearsky_hr_sw"] = clearsky_hr_sw

# %% calculate albedo
fluxes_3d["albedo_allsky"] = np.abs(
    fluxes_3d["allsky_sw_up"].isel(pressure=-1) / fluxes_3d["allsky_sw_down"].isel(pressure=-1)
)
fluxes_3d["albedo_clearsky"] = np.abs(
    fluxes_3d["clearsky_sw_up"].isel(pressure=-1) / fluxes_3d["clearsky_sw_down"].isel(pressure=-1)
)

# %% calculate connectedness
atms["connected"] = calc_connected(atms, frac_no_cloud=0.05, rain=True, convention='arts')

# %% calculate high cloud temperature from vertically integrated IWP
IWP_emission = 8e-3  # IWP where high clouds become opaque

p_top_idx_thin = (atms["IWC"] + atms['snow'] + atms['graupel']).argmax("pressure")
p_top_idx_thick = np.abs(atms["IWC_cumsum"] - IWP_emission).argmin("pressure")
p_top_idx = xr.where(p_top_idx_thick > p_top_idx_thin, p_top_idx_thick, p_top_idx_thin)
p_top = atms.isel(pressure=p_top_idx).pressure
T_h_lw = atms["temperature"].sel(pressure=p_top)
atms["hc_temperature"] = T_h_lw
atms["hc_top_pressure"] = p_top

# %% find profiles with high cloud tops above 350 hPa
mask_hc_no_lc = (atms["IWP"] > 1e-7) & (atms["LWP"] < 1e-7)
mask_height = p_top < 35000
atms["mask_height"] = mask_height
atms["mask_hc_no_lc"] = mask_hc_no_lc

# %% save results
os.remove(path_freddi + run + "atms_full.nc")
atms.to_netcdf(path_freddi + run + "atms_full.nc")
os.remove(path_freddi + run + "fluxes_3d_full.nc")
fluxes_3d.to_netcdf(path_freddi + run + "fluxes_3d_full.nc")

# %%
