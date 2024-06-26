# %%
import numpy as np
from src.read_data import (
    load_atms_and_fluxes,
    load_derived_vars,
    load_parameters,
    load_cre,
    load_mean_derived_vars,
)
from src.plot_functions import (
    plot_model_output_arts_reduced,
    plot_model_output_arts_fancy,
    plot_model_output_arts_with_cre,
)
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
atms_raw = xr.open_dataset("/work/bm1183/m301049/iwp_framework/mons/data/full_snapshot_proc.nc")
path = "/work/bm1183/m301049/iwp_framework/mons/model_output/"
run = "frozen_only"
with open(path + run + ".pkl", "rb") as f:
    result = pickle.load(f)

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
#fig.savefig("plots/paper/fancy_results_with_cre.png", dpi=500, bbox_inches="tight")
# %% plot reduced results
fig, axes = plot_model_output_arts_reduced(
    result,
    iwp_bins,
    atms,
    fluxes_3d_noice,
    lw_vars,
    sw_vars,
    f_vals,
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
    f_vals,
)

fig.savefig("plots/paper/fancy_results.png", dpi=500, bbox_inches="tight")

# %% plot CRE Comparison
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(cre_average["IWP"], cre_average["connected_sw"], color="blue", linestyle="--")
ax.plot(cre_average["IWP"], cre_average["connected_lw"], color="red", linestyle="--")
ax.plot(
    cre_average["IWP"],
    cre_average["connected_sw"] + cre_average["connected_lw"],
    color="black",
    linestyle="--",
)
ax.plot(result.index, result["SW_cre"], color="blue")
ax.plot(result.index, result["LW_cre"], color="red")
ax.plot(result.index, result["SW_cre"] + result["LW_cre"], color="black")
ax.set_xscale("log")
ax.set_xlim(1e-5, 1)
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("HCRE / W m$^{-2}$")
ax.spines[["top", "right"]].set_visible(False)
# make legend with fake handles and labels
handles = [
    plt.Line2D([0], [0], color="grey", linestyle="--"),
    plt.Line2D([0], [0], color="grey"),
    plt.Line2D([0], [0], color="red", linestyle="-"),
    plt.Line2D([0], [0], color="blue", linestyle="-"),
    plt.Line2D([0], [0], color="black", linestyle="-"),
]
labels = ["ARTS", "Conceptual Model", "LW", "SW", "Net"]
fig.legend(handles, labels, bbox_to_anchor=(0.95, -0.04), ncol=5)
fig.savefig("plots/paper/cre_comparison.png", dpi=500, bbox_inches="tight")

# %%
