# %%
import numpy as np
from src.read_data import (
    load_atms_and_fluxes,
    load_derived_vars,
)
from src.plot_functions import plot_connectedness
import pickle


# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()

path = '/work/bm1183/m301049/cm_results/'
run = 'icon_mons_const_lc'

with open(path + run + '.pkl', 'rb') as f:
    result = pickle.load(f)

# %%  plot connectedness 
liq_cld_cond = atms["LWC"] + atms["rain"]
ice_cld_cond = atms["IWC"] + atms["snow"] + atms["graupel"]
mask = atms["mask_height"] & (atms["IWP"] > 1e-6) & (atms["LWP"] > 1e-6)
iwp_bins = np.logspace(-5, 1, 7)
fig, axes = plot_connectedness(atms, mask, liq_cld_cond, ice_cld_cond, mode='arts')
fig.savefig(f"plots/paper/connectedness.png", dpi=500, bbox_inches="tight")

# %%
