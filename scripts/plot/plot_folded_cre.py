# %% import
import matplotlib.pyplot as plt
import numpy as np
from src.read_data import load_model_output, load_icon_snapshot

# %% load data
ds_monsoon = load_icon_snapshot()
result = load_model_output("prefinal")


# %% multiply hist with cre result
IWP_bins = np.logspace(-5, 1, num=50)

n_profiles = ds_monsoon['IWP'].count().values
hist, edges = np.histogram(ds_monsoon['IWP'].where(ds_monsoon['mask_height']), bins=IWP_bins)
hist = hist / n_profiles

cre_sw_weighted = hist * result['SW_cre']
cre_lw_weighted = hist * result['LW_cre']
cre_net_weighted = cre_sw_weighted + cre_lw_weighted

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
fig.savefig("plots/concept_cre.png", dpi=500, bbox_inches='tight')

# %%
