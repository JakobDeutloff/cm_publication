#%%
import numpy as np
from src.hc_model import run_model
from src.read_data import (
    load_atms_and_fluxes,
    load_parameters,
)
from src.helper_functions import cut_data
import pickle

# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
parameters = load_parameters()
model_results={}

# %% calculate constants used in the model
SW_in = cut_data(fluxes_3d["clearsky_sw_down"]).isel(pressure=-1).mean().values
parameters["SW_in"] = SW_in

# %% set mask ans bins 
mask = atms["mask_height"]
IWP_bins = np.logspace(-5, 1, num=50)


# %% run model for all profiles with cloud tops above 350 hPa 
result = run_model(
    IWP_bins,
    T_hc = cut_data(atms["hc_top_temperature"], mask),
    LWP = cut_data(atms['LWP'], mask),
    IWP = cut_data(atms['IWP'], mask),
    connectedness=atms['connected'],
    parameters = parameters,
    prescribed_lc_quantities=None
)
# %% save result 
path = '/work/bm1183/m301049/cm_results/'
with open(path + 'icon_mons_const_lc.pkl', 'wb') as f:
    pickle.dump(result, f)
# %%
