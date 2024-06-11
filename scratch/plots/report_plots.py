# %% import
import matplotlib.pyplot as plt
import numpy as np
import pickle
from src.read_data import load_cre, load_derived_vars, load_atms_and_fluxes
from calc_variables import cut_data

# %% load data
cre_binned, cre_interpolated, cre_average = load_cre()
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()
path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"
model_results = pickle.load(open(path + "model_results.pkl", "rb"))

# %% plot model results together with cre_average

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex="col", sharey=True)

# all clouds
axes[0].plot(cre_average["IWP"], cre_average["all_sw"], color="blue", linestyle="--", label="SW")
axes[0].plot(cre_average["IWP"], cre_average["all_lw"], color="red", linestyle="--", label="LW")
axes[0].plot(
    cre_average["IWP"], cre_average["all_net"], color="black", linestyle="--", label="Net"
)
axes[0].plot(model_results["all"]["SW_cre"], color="blue", label="SW")
axes[0].plot(model_results["all"]["LW_cre"], color="red", label="LW")
axes[0].plot(
    model_results["all"]["SW_cre"] + model_results["all"]["LW_cre"], color="black", label="Net"
)
mask = lw_vars["mask_height"]
per_grid = (mask * 1).sel(lat=slice(-30, 30)).mean().values * 100
axes[0].set_title(f"All Valid Clouds ({per_grid:.0f}% of gridcells)")

# high clouds without low clouds
mask = cre_average["IWP"] < 2
axes[1].plot(
    cre_average["IWP"][mask], cre_average["ice_only_sw"][mask], color="blue", linestyle="--", label="SW"
)
axes[1].plot(
    cre_average["IWP"][mask], cre_average["ice_only_lw"][mask], color="red", linestyle="--", label="LW"
)
axes[1].plot(
    cre_average["IWP"][mask], cre_average["ice_only_net"][mask], color="black", linestyle="--", label="Net"
)
mask_model = model_results["ice_only"].index < 2
axes[1].plot(model_results["ice_only"]["SW_cre"][mask_model], color="blue", label="SW")
axes[1].plot(model_results["ice_only"]["LW_cre"][mask_model], color="red", label="LW")
axes[1].plot(
    model_results["ice_only"]["SW_cre"][mask_model] + model_results["ice_only"]["LW_cre"][mask_model],
    color="black",
    label="Net",
)
mask_grid = lw_vars["mask_height"] & lw_vars["mask_hc_no_lc"]
per_grid = (mask_grid * 1).sel(lat=slice(-30, 30)).mean().values * 100
axes[1].set_title(f"High Clouds without Low Clouds ({per_grid:.0f}% of gridcells)")
axes[0].set_ylabel("Cloud Radiative Effect / W m$^{-2}$")



# make fake labels, remove axes at (1, 1) and put legend there
fake_ax = fig.add_subplot(111, frameon=False)
fake_ax.plot([], [], color="grey", label="Model")
fake_ax.plot([], [], color="grey", linestyle="--", label="ICON-ARTS")
fake_ax.plot([], [], color="blue", label="SW")
fake_ax.plot([], [], color="red", label="LW")
fake_ax.plot([], [], color="black", label="Net")
handles, labels = fake_ax.get_legend_handles_labels()
fake_ax.remove()
fig.legend(handles, labels, bbox_to_anchor=(0.77, -0.03), ncol=5)


for ax in axes.flatten():
    ax.set_xscale("log")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlim([1e-5, 1e1])
    ax.axhline(0, color="grey", linestyle=":")
    ax.set_xlabel("IWP / kg m$^{-2}$")

plt.savefig("plots/cre_report.png", dpi=300, bbox_inches="tight")



# %% plot T_top
mask = lw_vars["mask_height"]
IWP_bins = np.logspace(-5, 1, 50)
T_top_average = (
    cut_data(lw_vars["h_cloud_temperature"], mask)
    .groupby_bins(cut_data(atms["IWP"], mask), IWP_bins)
    .mean()
)
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.scatter(cut_data(atms["IWP"], mask), cut_data(lw_vars["h_cloud_temperature"], mask), s=1, color='k')
T_top_average.plot(ax=ax, color='red')

ax.set_xscale("log")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_ylabel("T$_{h}$ / K")
ax.set_xlabel("IWP / kg m$^{-2}$")

# %% plot R_t, f and alpha_t 
