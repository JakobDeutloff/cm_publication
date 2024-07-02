# %%
import numpy as np
from src.hc_model import run_model
from src.read_data import (
    load_parameters)
from src.helper_functions import cut_data
import pickle
import xarray as xr

# %% load data
atms = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/atms_proc.nc")
fluxes_allsky = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/fluxes.nc")
parameters = load_parameters(experiment='ngc3')

# %% set mask ans bins 
mask = atms["mask_height"]
IWP_bins = np.logspace(-6, np.log10(30), num=51)
# %% calculate constants used in the model
SW_in = fluxes_allsky["clearsky_sw_down"].isel(pressure=-1).mean().values
parameters["SW_in"] = SW_in

# %% run model
result = run_model(
    IWP_bins,
    T_hc = atms["hc_top_temperature"].where(mask),
    LWP = atms['LWP'].where(mask),
    IWP = atms['IWP'].where(mask),
    connectedness=atms['connected'],
    parameters = parameters,
    prescribed_lc_quantities=None
)

# %% save result 
path = '/work/bm1183/m301049/iwp_framework/ngc3/model_output/'
with open(path + 'first_try.pkl', 'wb') as f:
    pickle.dump(result, f)

# %% plot cre
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(result['SW_cre'])
ax.plot(result['LW_cre'])
ax.plot(result['LW_cre'] + result['SW_cre'])
ax.set_xscale('log')

# %%
