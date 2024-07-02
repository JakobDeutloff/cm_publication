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

# %% plot
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.stairs(cre_sw_weighted, IWP_bins, color='blue', label='SW')
ax.stairs(cre_lw_weighted, IWP_bins, color='red', label='LW')
ax.stairs(cre_net_weighted, IWP_bins, color='k', label='Net', fill=True, alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('$I$ / kg m$^{-2}$')
ax.set_ylabel(r"$C(I) \cdot P(I) ~ / ~ \mathrm{W ~ m^{-2}}$")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
fig.savefig("plots/paper/concept_cre.png", dpi=500, bbox_inches='tight')

# %%
