# %%
import numpy as np
from src.hc_model import run_model
from src.read_data import (
    load_atms_and_fluxes,
    load_parameters,
)
from src.helper_functions import cut_data
import pickle
from tqdm import tqdm


# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
parameters = load_parameters()
model_results={}

# %% set mask ans bins 
mask = atms["mask_height"]
IWP_bins = np.logspace(-5, 1, num=50)

# %% calculate constants used in the model
SW_in = cut_data(fluxes_3d["clearsky_sw_down"]).isel(pressure=-1).mean().values
parameters["SW_in"] = SW_in

# %% setup ensemble 
lc_fractions=np.array([0., 0.1, 0.16, 0.2, 0.32, 0.4, 0.5])
results={}

for lc_fraction in tqdm(lc_fractions):
    # Set additional parameters
    parameters['f'] = lc_fraction.astype(float)

    # run model
    result = run_model(
        IWP_bins,
        T_hc = cut_data(atms["hc_top_temperature"], mask),
        LWP = cut_data(atms['LWP'], mask),
        IWP = cut_data(atms['IWP'], mask),
        connectedness=atms['connected'],
        parameters = parameters,
        prescribed_lc_quantities=None
    )

    # save result 
    results[str(lc_fraction)] = result

# %% save result 
path = '/work/bm1183/m301049/iwp_framework/mons/model_output/'
with open(path + 'lc_ensemble.pkl', 'wb') as f:
    pickle.dump(results, f)
# %%
