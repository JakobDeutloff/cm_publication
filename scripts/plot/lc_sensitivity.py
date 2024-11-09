# %% import 
import numpy as np
import pickle
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from src.read_data import load_parameters
# %% load data
path = '/work/bm1183/m301049/iwp_framework/mons/model_output/'
ensemble = pickle.load(open(path + 'lc_ensemble.pkl', 'rb'))
sample = xr.open_dataset("/work/bm1183/m301049/iwp_framework/mons/data/full_snapshot_proc.nc")
parameters = load_parameters()

# %% define function to calculate sum_cre
def sum_cre(result, sample, iwp_bins):

    n_cells = sample['IWP'].count().values
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
IWP_bins = np.logspace(-5, 1, num=50)
idx = [float(i) for i in list(ensemble.keys())]
sum_cre_ensemble = pd.DataFrame(index=idx, columns=['SW', 'LW', 'net'])
for key, result in ensemble.items():
    sum_cre_ensemble.loc[float(key)]['SW'] = sum_cre(result, sample, IWP_bins)['SW']
    sum_cre_ensemble.loc[float(key)]['LW'] = sum_cre(result, sample, IWP_bins)['LW']
    sum_cre_ensemble.loc[float(key)]['net'] = sum_cre(result, sample, IWP_bins)['net']

# %% calculate slopes
res_net = linregress(sum_cre_ensemble.index, list(sum_cre_ensemble['net'].values))
res_sw = linregress(sum_cre_ensemble.index, list(sum_cre_ensemble['SW'].values))
res_lw = linregress(sum_cre_ensemble.index, list(sum_cre_ensemble['LW'].values))

# %% plot sum_cre
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(sum_cre_ensemble.index, sum_cre_ensemble['SW'], label='SW', color='blue')
ax.plot(sum_cre_ensemble.index, sum_cre_ensemble['LW'], label='LW', color='red')
ax.plot(sum_cre_ensemble.index, sum_cre_ensemble['net'], label='Net', color='black')
ax.axhline(0, color='grey', linestyle='--')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('$f_{\mathrm{lc}}$')
ax.set_ylabel(r'$\overline{C}$ / W m$^{-2}$')
ax.set_xticks([0, parameters['f'], 0.5])
tick_labels = ax.get_xticklabels()
tick_labels[1].set_fontweight('bold')
ax.set_yticks([-20, 0, 20])
fig.legend(handles=ax.lines, labels=['SW', 'LW', 'Net'], loc='upper right', bbox_to_anchor=(0.75, -0.01), ncols=3)
ax.text(0.8, 0.6, f'{res_net.slope:.2f} W m$^{{-2}}$', transform=ax.transAxes, color='black')
ax.text(0.8, 0.18, f'{res_sw.slope:.2f} W m$^{{-2}}$', transform=ax.transAxes, color='blue')
ax.text(0.8, 0.95, f'{res_lw.slope:.2f} W m$^{{-2}}$', transform=ax.transAxes, color='red')
fig.savefig('plots/lc_sensitivity.png', dpi=500, bbox_inches='tight')


# %% numbers for lc fraction increase 
increase = (sum_cre_ensemble.loc[0.32]['net'] - sum_cre_ensemble.loc[0.16]['net']) / sum_cre_ensemble.loc[0.16]['net'] * 100
print(f'Net HCRE increase for doubling of f: {increase:.2f}%')

# %%
