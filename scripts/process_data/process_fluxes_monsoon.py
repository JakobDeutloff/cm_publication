# %%
import xarray as xr
import numpy as np
from src.calc_variables import calc_connected
import os

# %% load data from freddis runs
path_freddi = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg/"
fluxes_3d = xr.open_dataset(path_freddi + run + "fluxes_3d.nc")

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


# %% save results
os.remove(path_freddi + run + "fluxes_3d_full.nc")
fluxes_3d.to_netcdf(path_freddi + run + "fluxes_3d_full.nc")

# %%
