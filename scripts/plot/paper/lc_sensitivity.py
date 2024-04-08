# %% import 
import numpy as np
import pickle
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
# %% load data
path = '/work/bm1183/m301049/cm_results/'
ensemble = pickle.load(open(path + 'lc_ensemble.pkl', 'rb'))
sample = xr.open_dataset("/work/bm1183/m301049/nextgems_profiles/monsoon/raw_data_converted.nc")

# %%

def sum_cre(result, sample, iwp_bins, mode='icon'):

    if mode == 'icon':
        n_cells = (len(sample.cell) * len(sample.time))
    else:
        n_cells = len(sample.lat) * len(sample.lon)

    hist, edges = np.histogram(sample["IWP"].where(sample['mask_height']), bins=iwp_bins)
    hist = hist / n_cells
    sum_sw = (result["SW_cre"] * hist).sum()
    sum_lw = (result["LW_cre"] * hist).sum()
    sum_net = sum_sw + sum_lw

    return {
        "SW": sum_sw,
        "LW": sum_lw,
        "net": sum_net,
    }

# %% calculate sum_cre for each ensemble member
IWP_bins = np.logspace(-5, np.log10(50), 60)
idx = [float(i) for i in list(ensemble.keys())]
sum_cre_ensemble = pd.DataFrame(index=idx, columns=['SW', 'LW', 'net'])
for key, result in ensemble.items():
    sum_cre_ensemble.loc[float(key)]['SW'] = sum_cre(result, sample, IWP_bins, mode='arts')['SW']
    sum_cre_ensemble.loc[float(key)]['LW'] = sum_cre(result, sample, IWP_bins, mode='arts')['LW']
    sum_cre_ensemble.loc[float(key)]['net'] = sum_cre(result, sample, IWP_bins, mode='arts')['net']

# %% plot sum_cre
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(sum_cre_ensemble.index, sum_cre_ensemble['SW'], label='SW', color='blue')
ax.plot(sum_cre_ensemble.index, sum_cre_ensemble['LW'], label='LW', color='red')
ax.plot(sum_cre_ensemble.index, sum_cre_ensemble['net'], label='Net', color='black')
ax.axhline(0, color='black', linestyle='--')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('Low Cloud Fraction')
ax.set_ylabel('HCRE / W m$^{-2}$')

fig.legend(handles=ax.lines, labels=['SW', 'LW', 'Net'], loc='upper right', bbox_to_anchor=(0.75, -0.01), ncols=3)
fig.savefig('plots/paper/lc_sensitivity.png', dpi=500, bbox_inches='tight')


# %%
