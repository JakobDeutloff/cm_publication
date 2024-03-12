"""
Script to derive variables for the binned atmospheric data and save them in a new file.
"""
# %% import
import xarray as xr
from calc_variables import calculate_lc_fraction, calculate_IWC_cumsum, calculate_h_cloud_temperature

# %% load data
path = "/work/bm1183/m301049/nextgems_profiles/"
atms = xr.open_dataset(path + "all_profiles_processed.nc")

# %% calculate lc fraction
atms["lc_fraction"] = calculate_lc_fraction(atms)

# %% calculate vertically integrated IWC 
atms["IWC_cumsum"] = calculate_IWC_cumsum(atms)

# %%
IWP_emission = 8e-3  # IWP where high clouds become opaque
atms['hc_temperature'], atms['hc_top_idx'] = calculate_h_cloud_temperature(atms, IWP_emission)

# %% save to file
atms.to_netcdf("/work/bm1183/m301049/nextgems_profiles/profiles_processed_2.nc")

