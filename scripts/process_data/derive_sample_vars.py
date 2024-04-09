# %% import
import xarray as xr
import os
from src.calc_variables import (
    calc_LWP,
    calc_IWP,
    calculate_IWC_cumsum,
    calculate_h_cloud_temperature,
    calc_connected,
)

# %% setting
rename = False
convention='arts'

# %%
path = "/work/bm1183/m301049/nextgems_profiles/monsoon/"
file = "atms_full.nc"
sample = xr.open_dataset(path + file)

# %% rename variables
if rename: 
    sample = sample.rename(
        {
            'qs': 'snow',
            'qr': 'rain',
            'cli': 'IWC', 
            'clw': 'LWC',
            'qg': 'graupel',
            'pfull': 'pressure',
            'ta': 'temperature',
            'zg': 'geometric_height',
            'ts': 'surface_temperature'
        }
    )

# %% calculate variables
sample["LWP"] = calc_LWP(sample,convention=convention)
sample["IWP"] = calc_IWP(sample, convention=convention)
sample["IWC_cumsum"] = calculate_IWC_cumsum(sample, convention=convention)
sample["connected"] = calc_connected(sample, convention=convention, frac_no_cloud=0.15)
sample["hc_temperature"], sample["hc_top_index"] = calculate_h_cloud_temperature(sample, convention=convention)

# %% mask for valid high clouds
if convention == 'icon':
    sample["mask_height"] = sample.sel(level_full=sample["hc_top_index"])["pressure"] < 35000
elif convention == 'arts':
    sample["mask_height"] = sample.sel(pressure=sample["hc_top_index"])["pressure"] < 35000

# %% save
os.remove(path + "atms_full.nc")
sample.to_netcdf(path + "atms_full.nc")

# %%
