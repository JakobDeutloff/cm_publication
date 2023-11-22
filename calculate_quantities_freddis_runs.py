# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

# %% load data from freddis runs
path_freddi = "/work/bm1183/m301049/freddi_runs/"
atms = xr.open_dataset(path_freddi + "atms.nc")
fluxes_3d = xr.open_dataset(path_freddi + "fluxes_3d.nc")
fluxes_2d = xr.open_dataset(path_freddi + "fluxes_2d.nc")
aux = xr.open_dataset(path_freddi + "aux.nc")

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

# %% calculate IWP and LWP
cell_height = atms["geometric height"].diff(
    "pressure"
)  # not correct, we would need height ad half levels
atms["IWP"] = ((atms["IWC"] + atms["snow"] + atms["graupel"]) * cell_height).sum(
    "pressure"
)
atms["LWP"] = ((atms["rain"] + atms["LWC"]) * cell_height).sum("pressure")

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

# %% Calculate T_h as T at maximum of IWC
min_height = 8e3  # m
p_max = atms["IWC"].idxmax("pressure")  # maximum should not be below min height
p_max = p_max.where(atms["geometric height"].sel(pressure=p_max) > min_height)
valid = p_max.notnull()
p_max = p_max.fillna(atms.pressure.max())
T_h = atms["temperature"].sel(pressure=p_max).where(valid)
atms["h_cloud_temperature"] = T_h
atms["h_cloud_top_pressure"] = p_max.where(valid)

# %% calculate high cloud albedo
atms["clearsky_albedo"] = (
    np.abs(
        fluxes_3d.isel(pressure=-1)["clearsky_sw_up"]
        / fluxes_3d.isel(pressure=-1)["clearsky_sw_down"]
    )
)
atms["allsky_albedo"] = (
    np.abs(
        fluxes_3d.isel(pressure=-1)["allsky_sw_up"]
        / fluxes_3d.isel(pressure=-1)["allsky_sw_down"]
    )
)
atms["high_cloud_albedo"] = (atms["allsky_albedo"] - atms["clearsky_albedo"]) / (
    1 - atms["clearsky_albedo"]
)

# %% calculate high cloud emissivity
sigma = 5.67e-8  # W m-2 K-4
LW_out_as = fluxes_3d.isel(pressure=-1)["allsky_lw_up"]
LW_out_cs = fluxes_3d.isel(pressure=-1)["clearsky_lw_up"]
rad_hc = - atms["h_cloud_temperature"] ** 4 * sigma
atms["high_cloud_emissivity"] = (LW_out_as - LW_out_cs) / (rad_hc - LW_out_cs)

# %% calculate high cloud radiative effect
fluxes_toa = fluxes_3d.isel(pressure=-1)
atms["cloud_rad_effect"] = (
    fluxes_toa["allsky_sw_down"]
    + fluxes_toa["allsky_sw_up"]
    + fluxes_toa["allsky_lw_down"]
    + fluxes_toa["allsky_lw_up"]
    - (fluxes_toa["clearsky_sw_down"] + fluxes_toa["clearsky_sw_up"])
    - (fluxes_toa["clearsky_lw_down"] + fluxes_toa["clearsky_lw_up"])
)

# %% save results
atms.to_netcdf(path_freddi + "atms_full.nc")
fluxes_3d.to_netcdf(path_freddi + "fluxes_3d_full.nc")


# %%
