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
    fluxes_3d_noice.where(mask),
    atms.where(mask),
    lw_vars.where(mask),
    parameters,
)

# %% plot model results
fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex="col")

axes[0, 0].scatter(
    cut_data(atms["IWP"], mask),
    cut_data(lw_vars["h_cloud_temperature"], mask),
    s=0.1,
    color="k",
)
axes[0, 0].plot(result["T_hc"], color="r")
axes[0, 0].set_ylabel("T_hc [K]")

axes[0, 1].scatter(
    cut_data(atms["IWP"], mask), cut_data(atms["LWP"], mask), s=0.1, color="k"
)
axes[0, 1].plot(result["LWP"], color="red")
axes[0, 1].set_ylim(1e-5, 1e1)
axes[0, 1].set_yscale("log")
axes[0, 1].set_ylabel("LWP [kg/m^2]")

axes[1, 0].plot(result["lc_fraction"], color="red")
axes[1, 0].set_ylabel("lc_fraction")


axes[1, 1].scatter(
    cut_data(atms["IWP"], mask),
    cut_data(fluxes_3d_noice["albedo_allsky"], mask),
    s=0.1,
    color="k",
)
cut_data(fluxes_3d_noice["albedo_allsky"], mask).groupby_bins(
    cut_data(atms["IWP"], mask), bins=IWP_bins
).mean().plot(ax=axes[1, 1], color="b")
axes[1, 1].plot(result["alpha_t"], color="r")
axes[1, 1].set_ylabel("alpha_t")

axes[2, 0].scatter(
    cut_data(atms["IWP"], mask), cut_data(lc_vars["R_t"], mask), s=0.1, color="k"
)
lc_vars["R_t"].sel(lat=slice(-30, 30)).groupby_bins(
    atms["IWP"].sel(lat=slice(-30, 30)), bins=IWP_bins
).mean().plot(ax=axes[2, 0], color="b")
axes[2, 0].plot(result["R_t"], color="red")
axes[2, 0].set_ylabel("R_t")

axes[2, 1].scatter(
    cut_data(atms["IWP"], mask & lw_vars["mask_hc_no_lc"]),
    cut_data(sw_vars["high_cloud_albedo"], mask & lw_vars["mask_hc_no_lc"]),
    s=0.1,
    color="k",
)
axes[2, 1].plot(result["alpha_hc"])
axes[2, 1].set_ylabel("alpha_hc")

axes[3, 0].scatter(
    cut_data(atms["IWP"], mask & lw_vars["mask_hc_no_lc"]),
    cut_data(lw_vars["high_cloud_emissivity"], mask & lw_vars["mask_hc_no_lc"]),
    s=0.1,
    color="k",
)
axes[3, 0].plot(result["em_hc"])
axes[3, 0].set_ylabel("em_hc")

axes[3, 1].plot(result["SW_cre"], color="blue", label="SW")
axes[3, 1].plot(result["LW_cre"], color="red", label="LW")
axes[3, 1].plot(result["SW_cre"] + result["LW_cre"], color="k", label="Net")
axes[3, 1].plot(cre_average["IWP"], cre_average["all_sw"], color="blue", linestyle="--")
axes[3, 1].plot(cre_average["IWP"], cre_average["all_lw"], color="red", linestyle="--")
axes[3, 1].plot(cre_average["IWP"], cre_average["all_net"], color="k", linestyle="--")
axes[3, 1].set_ylabel("CRE / W m^-2")
axes[3, 1].legend()

for ax in axes.flatten():
    ax.set_xscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# %%
