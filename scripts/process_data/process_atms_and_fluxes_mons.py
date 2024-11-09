"""
Processing of the ARTS radiative fluxes and the corresponding ICON profiles
"""
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
import os 
from src.read_data import get_data_path
# %% setting
convention = "arts"

# %%
path = get_data_path()
run_allksy = "flux_monsun_Nf10000_1deg/"
run_noice = "flux_monsun_Nf10000_1deg_nofrozen/"
sample = xr.open_dataset(path + "raw_data/atms.nc")
fluxes_allsky = xr.open_dataset(path + "raw_data/" + run_allksy + "fluxes_3d.nc")
fluxes_noice = xr.open_dataset(path + "raw_data/" + run_noice + "fluxes_3d.nc")

# %% calculate variables
sample["LWP"] = calc_LWP(sample, convention=convention)
sample["IWP"] = calc_IWP(sample, convention=convention)
sample["IWC_cumsum"] = calculate_IWC_cumsum(sample, convention=convention)
sample["connected"] = calc_connected(sample, convention=convention, frac_no_cloud=0.1)
sample["hc_top_temperature"], sample["hc_top_pressure"] = calculate_h_cloud_temperature(
    sample, fluxes_allsky, convention=convention, IWP_emission=6.7e-3, option="emission"
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
os.remove(path + "data/atms_proc.nc")
sample.to_netcdf(path + "data/atms_proc.nc")
os.remove(path + "data/fluxes_allsky_proc.nc")
fluxes_allsky.to_netcdf(path + "data/fluxes_allsky_proc.nc")
os.remove(path + "data/fluxes_noice_proc.nc")
fluxes_noice.to_netcdf(path + "data/fluxes_noice_proc.nc")

# %%
