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
from src.plot_functions import plot_model_output_arts_reduced, plot_model_output_arts_fancy, plot_model_output_arts_with_cre
import pickle
import xarray as xr
from src.hc_model import calc_lc_fraction
from src.helper_functions import cut_data
import matplotlib.pyplot as plt

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

# %% plot reduced results
fig, axes = plot_model_output_arts_reduced(
    result,
    iwp_bins,
    atms,
    fluxes_3d_noice,
    lw_vars,
    sw_vars,
    lw_vars_avg,
    sw_vars_avg,
    f_vals,
    const_lc_quantities,
    cs_constants,
)
fig.tight_layout()
fig.savefig("plots/paper/reduced_results.png", dpi=500, bbox_inches="tight")
# %% plor fancy results
fig, axes = plot_model_output_arts_fancy(
        result,
    iwp_bins,
    atms,
    fluxes_3d_noice,
    lw_vars,
    sw_vars,
    lw_vars_avg,
    sw_vars_avg,
    f_vals,
    const_lc_quantities,
    cs_constants,
)

fig.savefig("plots/paper/fancy_results.png", dpi=500, bbox_inches="tight")


# %% plot fancy results with cre 
fig, axes = plot_model_output_arts_with_cre(
    result,
    iwp_bins,
    atms,
    fluxes_3d_noice,
    lw_vars,
    sw_vars,
    lw_vars_avg,
    sw_vars_avg,
    f_vals,
    const_lc_quantities,
    cs_constants,
    cre_average,
)
fig.savefig("plots/paper/fancy_results_with_cre.png", dpi=500, bbox_inches="tight")

# %% plot CRE Comparison 
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(cre_average['IWP'], cre_average['connected_sw'], color='blue', linestyle='--')
ax.plot(cre_average['IWP'], cre_average['connected_lw'], color='red', linestyle='--')
ax.plot(cre_average['IWP'], cre_average['connected_sw'] + cre_average['connected_lw'], color='black', linestyle='--')
ax.plot(result.index, result['SW_cre'], color='blue')
ax.plot(result.index, result['LW_cre'], color='red')
ax.plot(result.index, result['SW_cre'] + result['LW_cre'], color='black')
ax.set_xscale('log')
ax.set_xlim(1e-5, 1)
ax.set_xlabel('IWP / kg m$^{-2}$')
ax.set_ylabel('HCRE / W m$^{-2}$')
ax.spines[['top', 'right']].set_visible(False)
# make legend with fake handles and labels 
handles = [plt.Line2D([0], [0], color='grey', linestyle='--'), plt.Line2D([0], [0], color='grey'), plt.Line2D([0], [0], color='red', linestyle='-'), plt.Line2D([0], [0], color='blue', linestyle='-'), plt.Line2D([0], [0], color='black', linestyle='-')]
labels = ['ARTS', 'Conceptual Model', 'LW', 'SW', 'Net']
fig.legend(handles, labels, bbox_to_anchor=(0.95, -0.04), ncol=5)
fig.savefig("plots/paper/cre_comparison.png", dpi=500, bbox_inches="tight")

# %%
