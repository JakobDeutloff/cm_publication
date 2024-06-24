# %% import
import numpy as np
from src.calc_variables import calc_cf
import xarray as xr
from scipy import interpolate
from tqdm import tqdm

# %% Load icon sample data
ds = xr.open_dataset(
    "/work/bm1183/m301049/nextgems_profiles/cycle3/representative_sample_c3_conn3.nc"
)
# %% calculate cloud fraction
ds["cf"] = calc_cf(ds)

# %% create a new coordinate for pressure
pressure_levels = np.linspace(7000, ds["pressure"].max(), num=80)

# create a new dataset with the same dimensions as ds, but with level_full replaced by pressure
ds_interp = xr.Dataset(
    coords={"time": ds["time"], "pressure_lev": pressure_levels, "cell": ds["cell"]}
)
ds_interp["IWP"] = ds["IWP"]


temp = np.zeros((len(ds["time"]), len(pressure_levels), len(ds["cell"])))
cf = temp.copy()
values = {"temperature": temp, "cf": cf}

# interpolate every profile
for i in tqdm(range(len(ds["time"]))):
    for j in range(len(ds["cell"])):
        for var in ["cf", "temperature"]:
            values[var][i, :, j] = interpolate.interp1d(
                ds["pressure"].isel(time=i, cell=j),
                ds[var].isel(time=i, cell=j),
                fill_value="extrapolate",
            )(pressure_levels)


ds_interp["temperature"] = xr.DataArray(
    values["temperature"], dims=["time", "pressure_lev", "cell"]
)
ds_interp["cf"] = xr.DataArray(values["cf"], dims=["time", "pressure_lev", "cell"])

# %% save data
ds_interp.to_netcdf("/work/bm1183/m301049/nextgems_profiles/cycle3/interp_representative_sample.nc")
# %%
