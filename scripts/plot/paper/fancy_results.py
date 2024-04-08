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
from src.plot_functions import plot_model_output_arts_fancy
import pickle
import xarray as xr
from src.hc_model import calc_lc_fraction
from src.helper_functions import cut_data

# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()
lw_vars_avg, sw_vars_avg, lc_vars_avg = load_binned_derived_variables()
parameters = load_parameters()
cre_binned, cre_interpolated, cre_average = load_cre()
const_lc_quantities = load_average_lc_parameters()
atms_raw = xr.open_dataset("/work/bm1183/m301049/nextgems_profiles/monsoon/raw_data_converted.nc")

path = "/work/bm1183/m301049/cm_results/"
run = "icon_mons_const_lc"
with open(path + run + ".pkl", "rb") as f:
    result = pickle.load(f)

# %% calculate cloud fractions
iwp_bins = np.logspace(-5, 1, 50)
iwp_points = (iwp_bins[1:] + iwp_bins[:-1]) / 2
f_raw = calc_lc_fraction(cut_data(atms["LWP"], lw_vars["mask_height"]), connected=False)
f_raw_binned = f_raw.groupby_bins(cut_data(atms["IWP"], lw_vars["mask_height"]), iwp_bins).mean()
f_unconnected = calc_lc_fraction(
    cut_data(atms["LWP"], lw_vars["mask_height"]),
    connected=cut_data(atms["connected"], lw_vars["mask_height"]),
)
f_unconnected_binned = f_unconnected.groupby_bins(cut_data(atms['IWP'], lw_vars['mask_height']), iwp_bins).mean()
f_mean = f_unconnected_binned.where(iwp_points < 1).mean()
f_vals = {'raw': f_raw_binned, 'unconnected': f_unconnected_binned, 'mean': f_mean}

# %% calculate clearsky constants
albedo_cs = cut_data(fluxes_3d["albedo_clearsky"]).mean().values
R_t_cs = cut_data(fluxes_3d['clearsky_lw_up']).isel(pressure=-1).mean().values
cs_constants = {'a_t': albedo_cs, 'R_t': R_t_cs}

# %% plot fancy results
fig, axes = plot_model_output_arts_fancy(
    result,
    iwp_bins,
    atms,
    fluxes_3d_noice,
    lw_vars,
    sw_vars,
    cre_average,
    lw_vars_avg,
    sw_vars_avg,
    f_vals,
    const_lc_quantities,
    cs_constants,
    atms_raw
)

fig.savefig("plots/paper/fancy_results.png", dpi=500, bbox_inches="tight")
# %%
