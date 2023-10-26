# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs  #

# %% load data from freddis runs
path_freddi = "/work/bm1183/m301049/freddi_runs/"
atms = xr.open_dataset(path_freddi + "atms.nc")
fluxes_3d = xr.open_dataset(path_freddi + "fluxes_3d.nc")
fluxes_2d = xr.open_dataset(path_freddi + "fluxes_2d.nc")
aux = xr.open_dataset(path_freddi + "aux.nc")

# %% calculate IWP and LWP
cell_height = atms["geometric height"].diff("pressure")  # not correct, we would need height ad half levels
atms["IWP"] = ((atms["IWC"] + atms["snow"] + atms["graupel"]) * cell_height).sum("pressure")
atms["LWP"] = ((atms["rain"] + atms["LWC"]) * cell_height).sum("pressure")

# %% calculate heating rates from fluxes (vertical levels are not quite correct) 
g = 9.81 
cp = 1005
seconds_per_day = 24 * 60 * 60
p = fluxes_3d["pressure"]
p_half = (p[1:].values + p[:-1].values) / 2
fluxes_3d = fluxes_3d.assign_coords(p_half=p_half)

allsky_hr_lw = (g/cp) * ((fluxes_3d["allsky_lw_up"] + fluxes_3d["allsky_lw_down"]).diff("pressure") / fluxes_3d["pressure"].diff("pressure")) * seconds_per_day
allsky_hr_lw['pressure'] = p_half
allsky_hr_lw = allsky_hr_lw.rename({'pressure': 'p_half'})
fluxes_3d['allsky_hr_lw'] = allsky_hr_lw

clearsky_hr_lw = (g/cp) * ((fluxes_3d["clearsky_lw_up"] + fluxes_3d["clearsky_lw_down"]).diff("pressure") / fluxes_3d["pressure"].diff("pressure")) * seconds_per_day
clearsky_hr_lw['pressure'] = p_half
clearsky_hr_lw = clearsky_hr_lw.rename({'pressure': 'p_half'})
fluxes_3d['clearsky_hr_lw'] = clearsky_hr_lw

allsky_hr_sw = (g/cp) * ((fluxes_3d["allsky_sw_up"] + fluxes_3d["allsky_sw_down"]).diff("pressure") / fluxes_3d["pressure"].diff("pressure")) * seconds_per_day
allsky_hr_sw['pressure'] = p_half
allsky_hr_sw = allsky_hr_sw.rename({'pressure': 'p_half'})
fluxes_3d['allsky_hr_sw'] = allsky_hr_sw

clearsky_hr_sw = (g/cp) * ((fluxes_3d["clearsky_sw_up"] + fluxes_3d["clearsky_sw_down"]).diff("pressure") / fluxes_3d["pressure"].diff("pressure")) * seconds_per_day
clearsky_hr_sw['pressure'] = p_half
clearsky_hr_sw = clearsky_hr_sw.rename({'pressure': 'p_half'})
fluxes_3d['clearsky_hr_sw'] = clearsky_hr_sw

# %% Calculate T_h as T at maximum of IWC
min_height = 8e3  # m
p_max = atms['IWC'].idxmax("pressure")
# maximum should not be below min height
p_max = p_max.where(atms["geometric height"].sel(pressure=p_max) < min_height)
p_max_indexer = p_max 
p_max_indexer[np.isnan(p_max_indexer)] = atms.pressure.max()
T_h = atms["temperature"].sel(pressure=p_max_indexer)
T_h = T_h.where(T_h < 280)
atms["h_cloud_temperature"] = T_h
atms['h_cloud_top_pressure'] = p_max

# %% save results 
atms.to_netcdf(path_freddi + "atms_full.nc")
fluxes_3d.to_netcdf(path_freddi + "fluxes_3d_full.nc")

# %%
