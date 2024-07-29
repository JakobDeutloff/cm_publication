# %% import
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pickle

# %% load data
ds_monsoon = xr.open_dataset("/work/bm1183/m301049/iwp_framework/mons/data/full_snapshot_proc.nc")
path = "/work/bm1183/m301049/iwp_framework/mons/model_output/"
run = "frozen_only"
with open(path + run + ".pkl", "rb") as f:
    result = pickle.load(f)

# %% multiply hist with cre result
IWP_bins = np.logspace(-5, 1, num=50)

n_profiles = ds_monsoon['IWP'].count().values
hist, edges = np.histogram(ds_monsoon['IWP'].where(ds_monsoon['mask_height']), bins=IWP_bins)
hist = hist / n_profiles

cre_sw_weighted = hist * result['SW_cre']
cre_lw_weighted = hist * result['LW_cre']
cre_net_weighted = cre_sw_weighted + cre_lw_weighted

# %% plot all three in one figure
fig, axes = plt.subplots(3, 1, figsize=(10, 6), height_ratios=[1, 1, 2], sharex=True)

# plot cre 
axes[0].axhline(0, color='grey', linestyle='--')
axes[0].plot(result.index, result['SW_cre'], label='SW', color='blue')
axes[0].plot(result.index, result['LW_cre'], label='LW', color='red')
axes[0].plot(result.index, result['SW_cre'] + result['LW_cre'], label='Net', color='k')
axes[0].set_yticks([-200, 0, 200])
axes[0].set_ylabel('$C(I)$ / W m$^{-2}$')

# plot IWP hist
axes[1].stairs(hist, edges, color='k')
axes[1].set_yticks([0, 0.02])
axes[1].set_ylabel('$P(I)$')


# plot cre weighted IWP hist
axes[2].stairs(cre_sw_weighted, IWP_bins, color='blue', label='SW')
axes[2].stairs(cre_lw_weighted, IWP_bins, color='red', label='LW')
axes[2].stairs(cre_net_weighted, IWP_bins, color='k', label='Net', fill=True, alpha=0.5)
axes[2].stairs(cre_net_weighted, IWP_bins, color='k', label='Net')
axes[2].set_xscale('log')
axes[2].set_xlabel('$I$ / kg m$^{-2}$')
axes[2].set_ylabel(r"$C(I) \cdot P(I) ~ / ~ \mathrm{W ~ m^{-2}}$")
axes[2].set_yticks([-1, 0, 1])
axes[2].set_xlim(1e-5, 10)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# legend at bottom with labels from ax 0
fig.subplots_adjust(bottom=0.1)
fig.legend(
    handles=axes[0].lines[1:],
    labels=['SW', 'LW', 'net'],
    bbox_to_anchor=(0.5, -0.05),
    loc='center',
    frameon=True,
    ncols=3,
)


fig.savefig("plots/paper/concept_cre_all.png", dpi=500, bbox_inches='tight')

# %% plot just P * C 
fig, ax = plt.subplots(figsize=(10, 4))

# plot cre weighted IWP hist
ax.stairs(cre_sw_weighted, IWP_bins, color='blue', label='SW')
ax.stairs(cre_lw_weighted, IWP_bins, color='red', label='LW')
ax.stairs(cre_net_weighted, IWP_bins, color='k', fill=True, alpha=0.5)
ax.stairs(cre_net_weighted, IWP_bins, color='k', label='Net')
ax.set_xscale('log')
ax.set_xlabel('$I$ / kg m$^{-2}$')
ax.set_ylabel(r"$C(I) \cdot P(I) ~ / ~ \mathrm{W ~ m^{-2}}$")
ax.set_yticks([-1, 0, 1])
ax.set_xlim(1e-5, 10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
fig.savefig("plots/paper/concept_cre.png", dpi=500, bbox_inches='tight')

# %%
