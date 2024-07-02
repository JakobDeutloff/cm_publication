# %% import
import xarray as xr
from src.calc_variables import (
    calc_LWP,
    calc_IWP,
    calculate_IWC_cumsum,
    calculate_h_cloud_temperature,
    calc_connected,
    change_convention,
)

# %% setting
rename = False
convention = "arts"

# %%
# path = "/work/bm1183/m301049/nextgems_profiles/cycle3/sample_3/"
path = "/work/bm1183/m301049/iwp_framework/mons/raw_data/"
run_allksy = "fullrange_flux_mid1deg/"
run_noice = "fullrange_flux_mid1deg_noice/"
sample = xr.open_dataset(path + run_allksy + "atms.nc")
fluxes_allsky = xr.open_dataset(path + run_allksy + "fluxes_3d.nc")
fluxes_noice = xr.open_dataset(path + run_noice + "fluxes_3d.nc")

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
sample["connected"] = calc_connected(sample, convention=convention, frac_no_cloud=0.1)
sample["hc_top_temperature"], sample["hc_top_pressure"] = calculate_h_cloud_temperature(
    sample, fluxes_allsky, convention=convention, IWP_emission=7.8e-3, option="emission"
)

# %% create masks
if convention == "icon":
    sample["mask_height"] = sample["hc_top_pressure"] < 35000
elif convention == "arts":
    sample["mask_height"] = sample["hc_top_pressure"] < 35000
sample["mask_hc_no_lc"] = (sample["IWP"] > 1e-5) & (sample["LWP"] < 1e-4)
sample["mask_low_cloud"] = ((sample["connected"] == 0) & (sample["LWP"] > 1e-4)) * 1

# %% cange flux convention (positive down)
fluxes_allsky = change_convention(fluxes_allsky)
fluxes_noice = change_convention(fluxes_noice)

# %% save
path = "/work/bm1183/m301049/iwp_framework/mons/data/"
sample.to_netcdf(path + "atms_proc.nc")
fluxes_allsky.to_netcdf(path + "fluxes_allsky_proc.nc")
fluxes_noice.to_netcdf(path + "fluxes_noice_proc.nc")

# %%
