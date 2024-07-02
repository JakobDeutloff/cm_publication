# %% import
import xarray as xr
from src.calc_variables import (
    calc_LWP,
    calc_IWP,
    calculate_IWC_cumsum,
    calculate_h_cloud_temperature,
    calc_connected,
)

# %% setting
rename = False
convention = "icon"

# %%
path = "/work/bm1183/m301049/iwp_framework/ngc3/raw_data/"
sample = xr.open_dataset(path + "atms.nc")
fluxes_allsky = xr.open_dataset(path + "../data/fluxes.nc")

# %% rename variables
if rename:
    sample = sample.rename(
        {
            "qs": "snow",
            "qr": "rain",
            "cli": "IWC",
            "clw": "LWC",
            "qg": "graupel",
            "pfull": "pressure",
            "ta": "temperature",
            "zg": "geometric_height",
            "ts": "surface_temperature",
        }
    )

# %% calculate variables
sample["LWP"] = calc_LWP(sample, convention=convention)
sample["IWP"] = calc_IWP(sample, convention=convention)
sample["IWC_cumsum"] = calculate_IWC_cumsum(sample, convention=convention)
sample["connected"] = calc_connected(sample, convention='icon_binned', frac_no_cloud=0.05)
# %%
sample["hc_top_temperature"], sample["hc_top_pressure"] = calculate_h_cloud_temperature(
    sample, fluxes_allsky, convention=convention, IWP_emission=6e-3, option="emission"
)

# %% create masks
if convention == "icon":
    sample["mask_height"] = sample["hc_top_pressure"] < 35000
elif convention == "arts":
    sample["mask_height"] = sample["hc_top_pressure"] < 35000
sample["mask_hc_no_lc"] = (sample["IWP"] > 1e-5) & (sample["LWP"] < 1e-4)
sample["mask_low_cloud"] = ((sample["connected"] == 0) & (sample["LWP"] > 1e-4)) * 1

# %% save
path = "/work/bm1183/m301049/iwp_framework/ngc3/data/"
sample.to_netcdf(path + "atms_proc.nc")

# %%
