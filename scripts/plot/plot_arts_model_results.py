# %%
import numpy as np
from src.read_data import (
    load_atms_and_fluxes,
    load_derived_vars,
    load_binned_derived_variables,
    load_parameters,
    load_cre,
    load_average_lc_parameters,
)
from src.plot_functions import plot_model_output_arts, plot_connectedness, plot_sum_cre
import pickle
import os


# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()
lw_vars_avg, sw_vars_avg, lc_vars_avg = load_binned_derived_variables()
parameters = load_parameters()
cre_binned, cre_interpolated, cre_average = load_cre()
const_lc_quantities = load_average_lc_parameters()


path = '/work/bm1183/m301049/cm_results/'
run = 'monsoon'

with open(path + "res_" + run + '.pkl', 'rb') as f:
    result = pickle.load(f)

# %% create folder
if not os.path.exists("plots/model_results/" + run):
    os.mkdir("plots/model_results/" + run)

# %% plot results
mask = lw_vars["mask_height"]
IWP_bins = np.logspace(-5, 1, 50)
fig, axes = plot_model_output_arts(
    result,
    IWP_bins,
    mask,
    atms,
    fluxes_3d_noice,
    lw_vars,
    sw_vars,
    cre_average,
    mode='connected'
)
per_gridcells = (mask*1).sel(lat=slice(-30, 30)).mean().values * 100
fig.suptitle(f"All Valid High Clouds ({per_gridcells:.0f}% of gridcells)", fontsize=12)
fig.tight_layout()
fig.savefig(f"plots/model_results/{run}/results.png", dpi=300)

# %%  plot connectedness 
liq_cld_cond = atms["LWC"] + atms["rain"]
ice_cld_cond = atms["IWC"] + atms["snow"] + atms["graupel"]
mask = lw_vars["mask_height"] & (atms["IWP"] > 1e-6) & (atms["LWP"] > 1e-6)
iwp_bins = np.logspace(-5, 1, 7)
fig, axes = plot_connectedness(atms, mask, liq_cld_cond, ice_cld_cond, mode='arts')
fig.savefig(f"plots/model_results/{run}/connectedness.png", dpi=300, bbox_inches="tight")

# %% Fold Distribution with CRE
fig, axes = plot_sum_cre(result, atms, np.logspace(-5, 1, 50), mode='arts')
fig.savefig(f'plots/model_results/{run}/cre_integrated.png', dpi=300, bbox_inches='tight')

