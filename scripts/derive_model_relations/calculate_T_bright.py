# %% import
import numpy as np
import matplotlib.pyplot as plt
from src.read_data import load_atms_and_fluxes, load_derived_vars
from matplotlib.colors import LogNorm

# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()


# %% calculate brightness temp from fluxes
def calc_brightness_temp(flux):
    return (-flux / 5.67e-8) ** (1 / 4)


# only look at clouds with cloud tops above 350 hPa and IWP > 1e-1 so that e = 1 can be assumed
mask = atms["mask_height"] & (atms["IWP"] > 1e-1) 

flux = fluxes_3d["allsky_lw_up"].isel(pressure=-1)
T_bright = calc_brightness_temp(flux)
T_bright = T_bright.where(mask, 500)

# tropopause pressure
p_trop_ixd = atms["temperature"].argmin("pressure")
p_trop = atms["pressure"].isel(pressure=p_trop_ixd)

# get pressure levels and ice_cumsum where T == T_bright in Troposphere
T_profile = atms["temperature"].where(atms["pressure"] > p_trop, 0)
p_bright_idx = np.abs(T_profile - T_bright).argmin("pressure")  # fills with 0 for NaNs
ice_cumsum = atms["IWC_cumsum"].isel(pressure=p_bright_idx)
ice_cumsum = ice_cumsum.where(mask)
p_bright = atms.isel(pressure=p_bright_idx)["pressure"].where(mask)
T_bright = T_bright.where(mask)
mean_ice_cumsum = ice_cumsum.sel(lat=slice(-30, 30)).median()
print(mean_ice_cumsum.values)

# %%
