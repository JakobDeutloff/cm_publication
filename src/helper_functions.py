"""
Quick and dirty helper functions
"""
import numpy as np
import xarray as xr
    
def logistic(x, L, x0, k):
    return L / (1 + np.exp(-k*(x-x0)))

def cut_data(data, mask=True):
    return data.sel(lat=slice(-30, 30)).where(mask)

def cut_data_mixed(data_cs, data_lc, mask, connected):
    # returns lc data fro not connnected profiles and cs data for connected profiles at mask
    data = xr.where(connected == 0 & mask, x=data_lc.where(mask), y=data_cs.where(mask)).sel(
        lat=slice(-30, 30)
    )
    return data

def hor_mean(data, mode="icon"):
    if mode == "icon":
        return data.mean(["cell", "time"])
    else:
        return data.sel(lat=slice(-30, 30)).mean(["lat", "lon"])
