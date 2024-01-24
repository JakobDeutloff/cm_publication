# %% import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.hc_model import run_model
from src.read_data import (
    load_atms_and_fluxes,
    load_derived_vars,
    load_binned_derived_variables,
    load_parameters,
    load_cre,
    load_average_lc_parameters,
)
from src.plot_functions import plot_model_output
from src.icon_arts_analysis import cut_data
import pickle


# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()
lw_vars_avg, sw_vars_avg, lc_vars_avg = load_binned_derived_variables()
parameters = load_parameters()
cre_binned, cre_interpolated, cre_average = load_cre()
const_lc_quantities = load_average_lc_parameters()
model_results={}

# %% calculate constants used in the model
albedo_cs = cut_data(fluxes_3d["albedo_clearsky"]).mean()
R_t_cs = cut_data(fluxes_3d['clearsky_lw_up']).isel(pressure=-1).mean()
SW_in = cut_data(fluxes_3d["clearsky_sw_down"]).isel(pressure=-1).mean()

# %% Set additional parameters
parameters['threshold_lc_fraction'] = 1e-6

# %% set mask ans bins 
mask = lw_vars["mask_height"]
IWP_bins = np.logspace(-5, 1, num=50)

# %% calculate presctibed lc quantities
prescribed_lc_quantitites = {
    "a_t": cut_data(lc_vars["albedo_allsky"], mask)
    .groupby_bins(cut_data(atms["IWP"], mask), bins=IWP_bins)
    .mean(),
    "R_t": cut_data(lc_vars["R_t"], mask)
    .groupby_bins(cut_data(atms["IWP"], mask), bins=IWP_bins)
    .mean(),
}

# %% run model for all profiles with cloud tops above 350 hPa 
result = run_model(
    IWP_bins,
    albedo_cs = albedo_cs, 
    R_t_cs = R_t_cs,
    SW_in = SW_in,
    T_hc = cut_data(lw_vars["h_cloud_temperature"], mask),
    LWP = cut_data(atms['LWP'], mask),
    IWP = cut_data(atms['IWP'], mask),
    parameters = parameters,
    const_lc_quantities=const_lc_quantities,
    prescribed_lc_quantities=None
)
model_results['all'] = result
# %% plot model results
fig, axes = plot_model_output(
    result,
    IWP_bins,
    mask,
    atms,
    fluxes_3d_noice,
    lw_vars,
    lw_vars_avg,
    sw_vars,
    sw_vars_avg,
    lc_vars,
    cre_average,
    mode='all'
)
per_gridcells = (mask*1).sel(lat=slice(-30, 30)).mean().values * 100
fig.suptitle(f"All Valid High Clouds ({per_gridcells:.0f}% of gridcells)", fontsize=12)
fig.tight_layout()
fig.savefig("plots/model_tests/8_cli_all.png", dpi=300)

 # %% run model for all profiles with cloud tops above 350 hPa and no low clouds below the high clouds 
mask = lw_vars["mask_height"] & lw_vars['mask_hc_no_lc']

result = run_model(
    IWP_bins,
    albedo_cs = albedo_cs, 
    R_t_cs = R_t_cs,
    SW_in = SW_in,
    T_hc = cut_data(lw_vars["h_cloud_temperature"], mask),
    LWP = cut_data(atms['LWP'], mask),
    IWP = cut_data(atms['IWP'], mask),
    parameters = parameters,
    const_lc_quantities=const_lc_quantities,
    prescribed_lc_quantities=None
)
model_results['ice_only'] = result

# %% plot results
fig, axes = plot_model_output(
    result,
    IWP_bins,
    mask,
    atms,
    fluxes_3d_noice,
    lw_vars,
    lw_vars_avg,
    sw_vars,
    sw_vars_avg,
    lc_vars,
    cre_average,
    mode='ice_only'
)
per_gridcells = (mask*1).sel(lat=slice(-30, 30)).mean().values * 100
fig.suptitle(f"High Clouds without Low Clouds ({per_gridcells:.0f}% of gridcells)", fontsize=12)
fig.tight_layout()
fig.savefig("plots/model_tests/8_cli_hc_no_lc.png", dpi=300)

# %% run model for all profiles with cloud tops above 350 hPa and low clouds below the high clouds
mask = lw_vars["mask_height"] & ~lw_vars['mask_hc_no_lc']

result = run_model(
    IWP_bins,
    albedo_cs = albedo_cs, 
    R_t_cs = R_t_cs,
    SW_in = SW_in,
    T_hc = cut_data(lw_vars["h_cloud_temperature"], mask),
    LWP = cut_data(atms['LWP'], mask),
    IWP = cut_data(atms['IWP'], mask),
    parameters = parameters,
    const_lc_quantities=const_lc_quantities,
    prescribed_lc_quantities=None
)
model_results['ice_over_lc'] = result

# %% plot results
fig, axes = plot_model_output(
    result,
    IWP_bins,
    mask,
    atms,
    fluxes_3d_noice,
    lw_vars,
    lw_vars_avg,
    sw_vars,
    sw_vars_avg,
    lc_vars,
    cre_average,
    mode='ice_over_lc'
)
per_gridcells = (mask*1).sel(lat=slice(-30, 30)).mean().values * 100
fig.suptitle(f"High Clouds with Low Clouds ({per_gridcells:.0f}% of gridcells)", fontsize=12)
fig.tight_layout()
fig.savefig("plots/model_tests/8_cli_hc_and_lc.png", dpi=300)

# %% save results
path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"
with open(path + "model_results.pkl", "wb") as f:
    pickle.dump(model_results, f)
    

# %%
