# %%
import numpy as np
from src.hc_model import run_model
from src.read_data import (
    load_atms_and_fluxes,
    load_parameters,
    load_average_lc_parameters,
    load_derived_vars,
)
from src.helper_functions import cut_data
import pickle
import xarray as xr
from tqdm import tqdm


# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()
parameters = load_parameters()
const_lc_quantities = load_average_lc_parameters()
model_results={}

# %% calculate constants used in the model
albedo_cs = cut_data(fluxes_3d["albedo_clearsky"]).mean().values
R_t_cs = cut_data(fluxes_3d['clearsky_lw_up']).isel(pressure=-1).mean().values
SW_in = cut_data(fluxes_3d["clearsky_sw_down"]).isel(pressure=-1).mean().values

# %% set mask ans bins 
mask = atms["mask_height"]
IWP_bins = np.logspace(-5, 1, num=50)

# %% calculate constants used in the model
albedo_cs = cut_data(fluxes_3d["albedo_clearsky"]).mean().values
R_t_cs = cut_data(fluxes_3d['clearsky_lw_up']).isel(pressure=-1).mean().values
SW_in = cut_data(fluxes_3d["clearsky_sw_down"]).isel(pressure=-1).mean().values

# %% setup ensemble 
lc_fractions=np.arange(0, 0.6, 0.1)
results={}

for lc_fraction in tqdm(lc_fractions):
    # Set additional parameters
    parameters['lc_fraction'] = lc_fraction.astype(float)

    # run model
    result = run_model(
        IWP_bins = np.logspace(-5, np.log10(50), 60),
        albedo_cs = albedo_cs, 
        R_t_cs = R_t_cs,
        SW_in = SW_in,
        T_hc = cut_data(atms["hc_top_temperature"], mask),
        LWP = cut_data(atms['LWP'], mask),
        IWP = cut_data(atms['IWP'], mask),
        connectedness=False,
        parameters = parameters,
        const_lc_quantities=const_lc_quantities,
        prescribed_lc_quantities=None
    )

    # save result 
    results[str(lc_fraction)] = result

# %% save result 
path = '/work/bm1183/m301049/cm_results/'
with open(path + 'lc_ensemble.pkl', 'wb') as f:
    pickle.dump(results, f)
# %%
