#%%
import numpy as np
from src.hc_model import run_model
from src.read_data import (
    load_atms_and_fluxes,
    load_derived_vars,
    load_binned_derived_variables,
    load_parameters,
    load_cre,
    load_average_lc_parameters,
)
from src.helper_functions import cut_data
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
albedo_cs = cut_data(fluxes_3d["albedo_clearsky"]).mean().values
R_t_cs = cut_data(fluxes_3d['clearsky_lw_up']).isel(pressure=-1).mean().values
SW_in = cut_data(fluxes_3d["clearsky_sw_down"]).isel(pressure=-1).mean().values

# %% set mask ans bins 
mask = atms["mask_height"]
IWP_bins = np.logspace(-5, 1, num=50)

# %% set additional parameters 
parameters['lc_fraction'] = float(const_lc_quantities['f'])

# %% run model for all profiles with cloud tops above 350 hPa 
result = run_model(
    IWP_bins,
    albedo_cs = albedo_cs, 
    R_t_cs = R_t_cs,
    SW_in = SW_in,
    T_hc = cut_data(atms["hc_top_temperature"], mask),
    LWP = cut_data(atms['LWP'], mask),
    IWP = cut_data(atms['IWP'], mask),
    connectedness=atms['connected'],
    parameters = parameters,
    const_lc_quantities=const_lc_quantities,
    prescribed_lc_quantities=None
)
# %% save result 
path = '/work/bm1183/m301049/cm_results/'
with open(path + 'icon_mons_const_lc.pkl', 'wb') as f:
    pickle.dump(result, f)
# %%
