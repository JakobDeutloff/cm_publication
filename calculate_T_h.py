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
cell_height = atms["geometric height"].diff("pressure")
atms["IWP"] = ((atms["IWC"] + atms["snow"] + atms["graupel"]) * cell_height).sum("pressure")
atms["LWP"] = ((atms["rain"] + atms["LWC"]) * cell_height).sum("pressure")
# %% find maximum of 
