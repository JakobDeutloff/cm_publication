# %% import
import numpy as np
from src.calc_variables import calc_cf
import xarray as xr
from scipy import interpolate
from tqdm import tqdm
import matplotlib.pyplot as plt

# %% Load icon sample data
ds = xr.open_dataset(
    "/work/bm1183/m301049/nextgems_profiles/monsoon/raw_data_converted.nc"
).sel(lat=slice(-30, 30))

#%% subsample data to 1e6 profiles 
ds_flat = ds.stack(idx=("lat", "lon"))
ds_flat = ds_flat.reset_index("idx")

#%%
random_idx = np.random.choice(ds_flat["idx"], size=int(2e6), replace=False)
ds_rand = ds_flat.sel(idx=random_idx)

# %% calculate cloud fraction
ds_rand["cf"] = calc_cf(ds_rand)
IWP_bins = np.logspace(-5, 1, 70)
IWP_points = (IWP_bins[:-1] + np.diff(IWP_bins)) / 2
ds_binned = ds_rand.groupby_bins("IWP", IWP_bins).mean('stacked_time_idx')
# %% test plot 
fig, ax = plt.subplots()
# plot cloud fraction
cf = ax.contourf(
    IWP_points,
    ds.height,
    ds_binned["cf"].T,
    cmap="Blues",
    levels=np.arange(0.1, 1.1, 0.1),
)

ax.invert_yaxis()
ax.set_xscale("log")

# %% create a new coordinate for pressure
pressure_levels = np.linspace(7000, ds["pressure"].max(), num=80)

#%% create a new dataset with the same dimensions as ds, but with level_full replaced by pressure
ds_interp = xr.Dataset(
    coords={"time": ds_rand["time"], "pressure_lev": pressure_levels, "idx": ds_rand["idx"].values}
)
ds_interp["IWP"] = ds_rand["IWP"]


temp = np.zeros((len(ds["time"]), len(pressure_levels), len(ds_rand["idx"])))
cf = temp.copy()
values = {"temperature": temp, "cf": cf}

#%% interpolate every profile
for i in range(len(ds_rand["time"])):
    for j in tqdm(range(len(ds_rand["idx"]))):
        for var in ["cf", "temperature"]:
            values[var][i, :, j] = interpolate.interp1d(
                ds_rand["pressure"].isel(time=i, idx=j),
                ds_rand[var].isel(time=i, idx=j),
                fill_value="extrapolate",
            )(pressure_levels)


ds_interp["temperature"] = xr.DataArray(
    values["temperature"], dims=["time", "pressure_lev", "idx"]
)
ds_interp["cf"] = xr.DataArray(values["cf"], dims=["time", "pressure_lev", "idx"])

# %% save data
ds_interp.to_netcdf("/work/bm1183/m301049/nextgems_profiles/monsoon/interp_cf.nc")
# %%
