# %% import 
import matplotlib.pyplot as plt
import numpy as np
import pickle 
from src.read_data import load_cre, load_derived_vars

# %% load data
cre_binned, cre_interpolated, cre_average = load_cre()
lw_vars, sw_vars, lc_vars = load_derived_vars()
path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"
model_results = pickle.load(open(path + "model_results.pkl", "rb"))

# %% plot model results together with cre_average 

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex='col', sharey=True)

# all clouds 
axes[0, 0].plot(cre_average['IWP'], cre_average['all_sw'], color='blue', linestyle='--', label='SW')
axes[0, 0].plot(cre_average['IWP'], cre_average['all_lw'], color='red', linestyle='--', label='LW')
axes[0, 0].plot(cre_average['IWP'], cre_average['all_net'], color='black', linestyle='--', label='Net')
axes[0, 0].plot(model_results['all']['SW_cre'], color='blue', label='SW')
axes[0, 0].plot(model_results['all']['LW_cre'], color='red', label='LW')
axes[0, 0].plot(model_results['all']['SW_cre'] + model_results['all']['LW_cre'], color='black', label='Net')
mask = lw_vars['mask_height']
per_grid = (mask*1).sel(lat=slice(-30, 30)).mean().values * 100
axes[0, 0].set_title(f'All Valid Clouds ({per_grid:.0f}% of gridcells)')

# high clouds without low clouds
axes[1, 0].plot(cre_average['IWP'], cre_average['ice_only_sw'], color='blue', linestyle='--', label='SW')
axes[1, 0].plot(cre_average['IWP'], cre_average['ice_only_lw'], color='red', linestyle='--', label='LW')
axes[1, 0].plot(cre_average['IWP'], cre_average['ice_only_net'], color='black', linestyle='--', label='Net')
axes[1, 0].plot(model_results['ice_only']['SW_cre'], color='blue', label='SW')
axes[1, 0].plot(model_results['ice_only']['LW_cre'], color='red', label='LW')
axes[1, 0].plot(model_results['ice_only']['SW_cre'] + model_results['ice_only']['LW_cre'], color='black', label='Net')
mask = lw_vars['mask_height'] & lw_vars['mask_hc_no_lc']
per_grid = (mask*1).sel(lat=slice(-30, 30)).mean().values * 100
axes[1, 0].set_title(f'High Clouds without Low Clouds ({per_grid:.0f}% of gridcells)')

# high clouds over low clouds
axes[0, 1].plot(cre_average['IWP'], cre_average['ice_over_lc_sw'], color='blue', linestyle='--', label='SW')
axes[0, 1].plot(cre_average['IWP'], cre_average['ice_over_lc_lw'], color='red', linestyle='--', label='LW')
axes[0, 1].plot(cre_average['IWP'], cre_average['ice_over_lc_net'], color='black', linestyle='--', label='Net')
axes[0, 1].plot(model_results['ice_over_lc']['SW_cre'], color='blue', label='SW')
axes[0, 1].plot(model_results['ice_over_lc']['LW_cre'], color='red', label='LW')
axes[0, 1].plot(model_results['ice_over_lc']['SW_cre'] + model_results['ice_over_lc']['LW_cre'], color='black', label='Net')
mask = lw_vars['mask_height'] & ~lw_vars['mask_hc_no_lc']
per_grid = (mask*1).sel(lat=slice(-30, 30)).mean().values * 100
axes[0, 1].set_title(f'High Clouds with Low Clouds ({per_grid:.0f}% of gridcells)')

# make fake labels, remove axes at (1, 1) and put legend there 

axes[1, 1].plot([], [], color='grey', label='Model')
axes[1, 1].plot([], [], color='grey', linestyle='--', label='ICON-ARTS')
axes[1, 1].plot([], [], color='blue', label='SW')
axes[1, 1].plot([], [], color='red', label='LW')
axes[1, 1].plot([], [], color='black', label='Net')
handles, labels = axes[1, 1].get_legend_handles_labels()
axes[1, 1].remove()
fig.legend(handles, labels, bbox_to_anchor=(0.87, 0.4), ncol=3)


for ax in axes.flatten():
    ax.set_xscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([1e-5, 1e1])



# %%
