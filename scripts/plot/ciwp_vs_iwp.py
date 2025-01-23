# %% import
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from src.read_data import get_data_path

# %% load data
path = get_data_path()
cre_allsky = xr.open_dataset(path + "data/cre_mean.nc")
cre_allsky_std = xr.open_dataset(path + "data/cre_std.nc")
cre_ice = xr.open_dataset(path + "data/cre_ciwp_mean.nc")
cre_ice_std = xr.open_dataset(path + "data/cre_ciwp_std.nc")
ds_monsoon = xr.open_dataset(path + "data/full_snapshot_proc.nc")

# %% calculate IWP  without precipitate
iwp_nosnow = (ds_monsoon["IWC"] * ds_monsoon["dzghalf"]).sum("height")
precip_path = (
    (ds_monsoon["snow"] + ds_monsoon["graupel"]) * ds_monsoon["dzghalf"]
).sum("height")

# %% calculate IWP Hists
n_cells = len(ds_monsoon.lat) * len(ds_monsoon.lon)
hist_allsky, edges_allsky = np.histogram(
    ds_monsoon["IWP"].where(ds_monsoon["mask_height"]), bins=cre_allsky.IWP_bins
)
hist_allsky = hist_allsky / n_cells
hist_nosnow, edges_nosnow = np.histogram(
    iwp_nosnow.where(ds_monsoon["mask_height"]), bins=cre_ice.IWP_bins
)
hist_nosnow = hist_nosnow / n_cells

# %% plot CRE vs IWP
fig, axes = plt.subplots(
    2, 2, figsize=(10, 6), sharey="row", sharex="col", height_ratios=[1, 0.5]
)

axes[0, 0].axhline(0, color="grey", linestyle="-", linewidth=0.5)
cre_allsky["connected_sw"].plot(ax=axes[0, 0], color="blue")
cre_allsky["connected_lw"].plot(ax=axes[0, 0], color="red")
cre_allsky["connected_net"].plot(ax=axes[0, 0], color="k")
axes[0, 0].fill_between(
    cre_allsky_std.IWP,
    cre_allsky["connected_net"] - cre_allsky_std["connected_net"],
    cre_allsky["connected_net"] + cre_allsky_std["connected_net"],
    color="k",
    alpha=0.5,
    edgecolor="none",
)
axes[0, 0].fill_between(
    cre_allsky_std.IWP,
    cre_allsky["connected_sw"] - cre_allsky_std["connected_sw"],
    cre_allsky["connected_sw"] + cre_allsky_std["connected_sw"],
    color="blue",
    alpha=0.5,
    edgecolor="none",
)
axes[0, 0].fill_between(
    cre_allsky_std.IWP,
    cre_allsky["connected_lw"] - cre_allsky_std["connected_lw"],
    cre_allsky["connected_lw"] + cre_allsky_std["connected_lw"],
    color="red",
    alpha=0.5,
    edgecolor="none",
)


axes[0, 1].axhline(0, color="grey", linestyle="-", linewidth=0.5)
cre_ice["connected_sw"].plot(ax=axes[0, 1], color="blue")
cre_ice["connected_lw"].plot(ax=axes[0, 1], color="red")
cre_ice["connected_net"].plot(ax=axes[0, 1], color="k")
axes[0, 1].fill_between(
    cre_ice_std.IWP,
    cre_ice["connected_net"] - cre_ice_std["connected_net"],
    cre_ice["connected_net"] + cre_ice_std["connected_net"],
    color="k",
    alpha=0.5,
    edgecolor="none",
)
axes[0, 1].fill_between(
    cre_ice_std.IWP,
    cre_ice["connected_sw"] - cre_ice_std["connected_sw"],
    cre_ice["connected_sw"] + cre_ice_std["connected_sw"],
    color="blue",
    alpha=0.5,
    edgecolor="none",
)
axes[0, 1].fill_between(
    cre_ice_std.IWP,
    cre_ice["connected_lw"] - cre_ice_std["connected_lw"],
    cre_ice["connected_lw"] + cre_ice_std["connected_lw"],
    color="red",
    alpha=0.5,
    edgecolor="none",
)


axes[1, 0].stairs(hist_allsky, edges_allsky, color="black")

axes[1, 1].stairs(hist_nosnow, edges_nosnow, color="black")

for ax in axes.flatten():
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")

axes[0, 0].set_xlim(1e-5, 10)
axes[0, 0].set_ylim(-300, 200)
axes[0, 0].set_yticks([-200, 0, 200])
axes[0, 1].set_xlim(1e-5, 0.5)
axes[1, 0].set_yticks([0, 0.02])
axes[1, 0].set_xlabel("$I$ / kg m$^{-2}$")
axes[1, 1].set_xlabel("$I_{\mathrm{ci}}$ / kg m$^{-2}$")
axes[0, 0].set_ylabel("$C(I)$ / W m$^{-2}$")
axes[0, 1].set_ylabel("$C_{\mathrm{ci}}(I_{\mathrm{ci}})$ / W m$^{-2}$")
axes[1, 0].set_ylabel("$P(I)$")
axes[1, 1].set_ylabel("$P(I_{\mathrm{ci}})$")


# make legend for cre
labels = ["LW", "SW", "net"]
handles = [
    plt.Line2D([0], [0], color="red"),
    plt.Line2D([0], [0], color="blue"),
    plt.Line2D([0], [0], color="k"),
]
axes[0, 0].legend(handles, labels, frameon=False)

# label the subplots
labels = ["a", "b", "c", "d"]
for i, ax in enumerate(axes.flatten()):
    ax.text(
        0.05,
        0.95,
        labels[i],
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        color="k",
    )

fig.tight_layout()
fig.savefig("plots/cre_vs_ciwp.png", dpi=300)

# %% calculate mean std
print(f"Mean std allsky lw: {cre_allsky_std['connected_lw'].mean().values:.2f} w/m2")
print(f"Mean std allsky sw: {cre_allsky_std['connected_sw'].mean().values:.2f} w/m2")
print(f"Mean std allsky net: {cre_allsky_std['connected_net'].mean().values:.2f} w/m2")
print(f"Mean std nosnow lw: {cre_ice_std['connected_lw'].mean().values:.2f} w/m2")
print(f"Mean std nosnow sw: {cre_ice_std['connected_sw'].mean().values:.2f} w/m2")
print(f"Mean std nosnow net: {cre_ice_std['connected_net'].mean().values:.2f} w/m2")

# %% calculate mean cre
mean_cre_net_allsky = (cre_allsky["connected_net"] * hist_allsky).sum().values
mean_cre_sw_allsky = (cre_allsky["connected_sw"] * hist_allsky).sum().values
mean_cre_lw_allsky = (cre_allsky["connected_lw"] * hist_allsky).sum().values
mean_cre_net_nosnow = (cre_ice["connected_net"] * hist_nosnow).sum().values
mean_cre_sw_nosnow = (cre_ice["connected_sw"] * hist_nosnow).sum().values
mean_cre_lw_nosnow = (cre_ice["connected_lw"] * hist_nosnow).sum().values

print(
    f"Net CRE allsky: {mean_cre_net_allsky:.2f} W/m2, Noprecip: {mean_cre_net_nosnow:.2f} W/m2, change: { (100 * (mean_cre_net_nosnow - mean_cre_net_allsky) / mean_cre_net_allsky):.2f} %"
)
print(
    f"SW CRE allsky: {mean_cre_sw_allsky:.2f} W/m2, Noprecip: {mean_cre_sw_nosnow:.2f} W/m2, change: { (100 * (mean_cre_sw_nosnow - mean_cre_sw_allsky) / mean_cre_sw_allsky):.2f} %"
)
print(
    f"LW CRE allsky: {mean_cre_lw_allsky:.2f} W/m2, Noprecip: {mean_cre_lw_nosnow:.2f} W/m2, change: { (100 * (mean_cre_lw_nosnow - mean_cre_lw_allsky) / mean_cre_lw_allsky):.2f} %"
)

# %% calculate fraction of total ice which is cloud ice
print(
    f"fraction of ice being cloud ice: {iwp_nosnow.sum().values /  ds_monsoon['IWP'].sum().values}"
)

# %% correlate iwp and ciwp
bins = np.logspace(-6, 1, 100)
bin_centers = (bins[:-1] + bins[1:]) / 2
binned_iwp_nosnow = iwp_nosnow.groupby_bins(
    ds_monsoon["IWP"], bins, labels=bin_centers
).mean()
std_binned_iwp_nosnow = iwp_nosnow.groupby_bins(ds_monsoon["IWP"], bins).std()
flat_iwp = ds_monsoon["IWP"].values.flatten()
flat_iwp_nosnow = iwp_nosnow.values.flatten()

# %% plot iwp vs ciwp
fig, ax = plt.subplots()
rand_idx = np.random.randint(0, len(flat_iwp), 10000)
ax.scatter(flat_iwp[rand_idx], flat_iwp_nosnow[rand_idx], s=0.5, marker="o", color="k")
ax.plot(bin_centers, binned_iwp_nosnow, color="red", label="mean")
ax.fill_between(
    bin_centers,
    binned_iwp_nosnow.values - std_binned_iwp_nosnow.values,
    binned_iwp_nosnow.values + std_binned_iwp_nosnow.values,
    alpha=0.5,
    color="red",
    label="$\pm$ $\sigma$",
    edgecolor="none",
)
ax.plot(np.logspace(-8, 1, 100), np.logspace(-8, 1, 100), color="grey", linestyle="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("$I_{\mathrm{ci}}$ / kg m$^{-2}$")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(1e-6, 10)
ax.set_ylim(1e-6, 10)
ax.legend(frameon=False)
fig.savefig("plots/iwp_vs_ciwp.png", dpi=300)

# %%
