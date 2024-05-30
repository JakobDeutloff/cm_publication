import xarray as xr
from src.read_data import load_atms_and_fluxes
import matplotlib.pyplot as plt
import tqdm

# %% read data
path = "/work/um0878/users/mbrath/StarARTS/results/processed_fluxes/flux_iconc3_Nf10000"
runs = ["", "_noliquid", "_nofrozen"]
file = "fluxes_flux_iconc3_Nf10000"

fluxes = {}
for run in runs:
    fluxes[run] = xr.open_dataset(f"{path}{run}/{file}{run}.nc")

path = "/work/um0878/user_data/jdeutloff/icon_c3_sample/"
atms_unstructured = xr.open_dataset(path + "atms_unstructured_ozone.nc")
atms = xr.open_dataset(path + "atms.nc")

# %% define mappings and keys
flux_keys = [
    "allsky_sw_up",
    "allsky_sw_down",
    "clearsky_sw_up",
    "clearsky_sw_down",
    "allsky_lw_up",
    "allsky_lw_down",
    "clearsky_lw_up",
    "clearsky_lw_down",
]


def mapping(flux, key):
    if key == "allsky_sw_up":
        return flux["allsky_solar"].sel(direction="upward")
    elif key == "allsky_sw_down":
        return flux["allsky_solar"].sel(direction="downward")
    elif key == "clearsky_sw_up":
        return flux["clearsky_solar"].sel(direction="upward")
    elif key == "clearsky_sw_down":
        return flux["clearsky_solar"].sel(direction="downward")
    elif key == "allsky_lw_up":
        return flux["allsky_thermal"].sel(direction="upward")
    elif key == "allsky_lw_down":
        return flux["allsky_thermal"].sel(direction="downward")
    elif key == "clearsky_lw_up":
        return flux["clearsky_thermal"].sel(direction="upward")
    elif key == "clearsky_lw_down":
        return flux["clearsky_thermal"].sel(direction="downward")
    
# %% restructure fluxes dataset to match atms dimensions and coordinates

# %%
for run in runs:
    # create structured dataset and select run
    flux = fluxes[run]
    flux_structured = xr.Dataset(coords=atms.coords)
    flux_structured = flux_structured.rename({"level_full": "pressure"})
    flux_structured["pressure"] = fluxes[""]["pressure"]
    for key in flux_keys:
        flux_structured[key] = xr.DataArray(
            data=0,
            coords=flux_structured.coords,
        )
    # fill structured dataset with values from unstructured dataset
    for i in tqdm.tqdm(flux.index):
        atms_point = atms_unstructured.isel(idx=i)
        iwp_point, local_time_point, profile = (
            atms_point["iwp_points"],
            atms_point["local_time_points"],
            atms_point["profile"],
        )
        for key in flux_keys:
            flux_structured[key].loc[
                dict(iwp_points=iwp_point, local_time_points=local_time_point, profile=profile)
            ] = mapping(flux, key).sel(index=i)

    flux_structured.to_netcdf(f"{path}fluxes{run}_structured.nc")