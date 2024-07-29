# %%
import numpy as np
from src.read_data import (
    load_atms_and_fluxes,
    load_derived_vars,
    load_parameters,
    load_cre,
    load_mean_derived_vars,
    load_icon_snapshot,
    load_model_output
)
from src.plot_functions import plot_model_output_arts_with_cre
import pickle
import xarray as xr
from src.hc_model import calc_lc_fraction
from src.helper_functions import cut_data
import matplotlib.pyplot as plt

# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lower_trop_vars = load_derived_vars()
mean_lw_vars, mean_sw_vars, mean_lc_vars = load_mean_derived_vars()
parameters = load_parameters()
cre_binned, cre_average = load_cre()
atms_raw = load_icon_snapshot()
result = load_model_output("prefinal")

# %% calculate cloud fractions
iwp_bins = np.logspace(-5, 1, 50)
iwp_points = (iwp_bins[1:] + iwp_bins[:-1]) / 2
f_raw = calc_lc_fraction(cut_data(atms["LWP"], atms["mask_height"]), connected=False)
f_raw_binned = f_raw.groupby_bins(cut_data(atms["IWP"], atms["mask_height"]), iwp_bins).mean()
f_unconnected = calc_lc_fraction(
    cut_data(atms["LWP"], atms["mask_height"]),
    connected=cut_data(atms["connected"], atms["mask_height"]),
)
f_unconnected_binned = f_unconnected.groupby_bins(
    cut_data(atms["IWP"], atms["mask_height"]), iwp_bins
).mean()
f_vals = {"raw": f_raw_binned, "unconnected": f_unconnected_binned}


# %% plot fancy results with cre
fig, axes = plot_model_output_arts_with_cre(
    result,
    iwp_bins,
    atms,
    fluxes_3d_noice,
    lw_vars,
    mean_lw_vars,
    sw_vars,
    mean_sw_vars,
    f_vals,
    parameters,
    cre_average,
)
fig.savefig("plots/fancy_results_with_cre.png", dpi=500, bbox_inches="tight")

# %%
