# %% import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.read_data import load_atms_and_fluxes, load_derived_vars, load_parameters, load_binned_derived_variables
from src.helper_functions import logistic, cut_data

# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()
lw_vars_binned, sw_vars_binned, lc_vars_binned = load_binned_derived_variables()
params = load_parameters()


# %% make one dataset for all variables - this is the long term plan for the routines
atms["hc_temperature"] = lw_vars["h_cloud_temperature"]
atms["hc_emissivity"] = lw_vars["high_cloud_emissivity"]
atms["hc_albedo"] = sw_vars["high_cloud_albedo"]
atms["mask_height"] = lw_vars["mask_height"]
atms["mask_hc_no_lc"] = lw_vars["mask_hc_no_lc"]
atms = cut_data(atms, mask=atms["mask_height"])
fluxes_3d = cut_data(fluxes_3d, mask=atms["mask_height"])

# %% plot
iwp_bins = np.logspace(-5, 1, num=50)
iwp_points = (iwp_bins[1:] + iwp_bins[:-1]) / 2
fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex='col')

# plot temperature
mean_t = atms["hc_temperature"].groupby_bins(atms["IWP"], iwp_bins).mean()
axes[0].scatter(atms["IWP"], atms["hc_temperature"], s=1, color="grey")
axes[0].plot(iwp_points, mean_t, color="black")
axes[0].set_ylabel("Temperature / K")

# emissivity
fitted_emissivity = logistic(np.log10(iwp_points), *params["em_hc"])
axes[1].scatter(
    atms["IWP"].where(atms["mask_hc_no_lc"]),
    atms["hc_emissivity"].where(atms["mask_hc_no_lc"]),
    s=1,
    color="grey",
)
axes[1].plot(lw_vars_binned['binned_emissivity'], color="black")
axes[1].plot(iwp_points, fitted_emissivity, color="red", linestyle="--")
axes[1].set_ylabel("Emissivity")

# plot albedo
fitted_albedo = logistic(np.log10(iwp_points), *params["alpha_hc"])
sc = axes[2].scatter(
    atms["IWP"].where(atms["mask_hc_no_lc"]),
    atms["hc_albedo"].where(atms["mask_hc_no_lc"]),
    s=1,
    c=fluxes_3d['allsky_sw_down'].isel(pressure=-1).where(atms["mask_hc_no_lc"]),
    cmap="viridis"
)
axes[2].plot(sw_vars_binned['binned_albedo'], color="black", label='Mean')
axes[2].plot(iwp_points, fitted_albedo, color="red", linestyle="--", label='Fited Logistic')
axes[2].set_ylabel("Albedo")
axes[2].set_xlabel("IWP / kg m$^{-2}$")

# colorbar at bottom of plots
fig.colorbar(sc, ax=axes, label="SW down / W m$^{-2}$", orientation="horizontal", pad=0.07, aspect=40)

# legend below colorbar
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.2, -0.032, 0.5, 0.2), ncol=2)

for ax in axes:
    ax.set_xscale("log")
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xlim(1e-5, 1e1)

fig.savefig("plots/paper/hc_vars.png", dpi=400, bbox_inches="tight")
# %%
