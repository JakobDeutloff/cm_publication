"""
All the functions to calculate variables used in the model
"""
# %% import 
import xarray as xr
import numpy as np

# %% functions

def calculate_lc_fraction(atms):
    """
    Calculate the fraction of liquid clouds.
    """
    lc_fraction = (atms["LWP"] > 1e-6) * 1
    return lc_fraction

def calculate_IWC_cumsum(atms):
    """
    Calculate the vertically integrated ice water content.
    """
    ice_mass = ((atms["IWC"] + atms['graupel'] + atms['snow']) * atms['dzghalf'])
    IWC_cumsum = ice_mass.cumsum("level_full")
    return IWC_cumsum

def calculate_h_cloud_temperature(atms, IWP_emission):
    """
    Calculate the temperature of high clouds.
    """
    top_idx_thin = (atms["IWC"] + atms['snow'] + atms['graupel']).argmax("level_full")
    top_idx_thick = np.abs(atms["IWC_cumsum"] - IWP_emission).argmin("level_full")
    top_idx = xr.where(top_idx_thick < top_idx_thin, top_idx_thick, top_idx_thin)
    top = atms.isel(level_full=top_idx).level_full
    T_h = atms["temperature"].sel(level_full=top)
    return T_h, top