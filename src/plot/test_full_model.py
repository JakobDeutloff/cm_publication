# %% import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.hc_model import run_model
from src.read_data import (
    load_atms_and_fluxes,
    load_derived_vars,
    load_averaged_derived_variables,
    load_parameters,
    load_cre,
)
from src.plot_functions import plot_model_output
from src.icon_arts_analysis import cut_data


# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()
lw_vars_avg, sw_vars_avg, lc_vars_avg = load_averaged_derived_variables()
parameters = load_parameters()
cre_binned, cre_interpolated, cre_average = load_cre()

# %% run model
mask = lw_vars["mask_height"]
IWP_bins = np.logspace(-5, 0, num=70)
result = run_model(
    IWP_bins,
    cut_data(fluxes_3d_noice, mask),
    cut_data(atms, mask),
    cut_data(lw_vars, mask),
    parameters,
)

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
)

fig.suptitle("LWP dependent model, lc_fraction from 1e-6", fontsize=12)
fig.tight_layout()
fig.savefig("plots/model_tests/1e-6lcf_LWP_dependent.png", dpi=300)

# %%
