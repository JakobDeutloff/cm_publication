# %% import
import xarray as xr
from src.calc_variables import (
    calc_LWP,
    calc_IWP,
    calculate_IWC_cumsum,
    calculate_h_cloud_temperature,
    calc_connected,
)
import os
from dask.diagnostics import ProgressBar

# %% setting
rename = True

# %%
path = "/work/bm1183/m301049/nextgems_profiles/cycle4/"
file = "representative_sample_c4.nc"
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
sample["LWP"] = calc_LWP(sample)
sample["IWP"] = calc_IWP(sample)
sample["IWC_cumsum"] = calculate_IWC_cumsum(sample)
sample["connected"] = calc_connected(sample)
sample["hc_temperature"], sample["hc_top_index"] = calculate_h_cloud_temperature(sample)

# %% mask for valid high clouds
sample["mask_height"] = sample.sel(level_full=sample["hc_top_index"])["pressure"] < 35000

# %% save
sample.to_netcdf(path + "representative_sample_c4_conn3.nc")

# %%
