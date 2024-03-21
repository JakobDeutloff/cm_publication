# %% import 
import numpy as np
import pickle
import xarray as xr
import matplotlib.pyplot as plt
from src.plot_functions import plot_sum_cre
import pandas as pd

# %% load data
path = '/work/bm1183/m301049/cm_results/'
ensemble = pickle.load(open(path + 'lc_ensemble.pkl', 'rb'))
sample = xr.open_dataset("/work/bm1183/m301049/nextgems_profiles/cycle3/representative_sample_c3_conn3.nc")

# %%

def sum_cre(result, sample, iwp_bins, mode='icon'):

    if mode == 'icon':
        n_cells = (len(sample.cell) * len(sample.time))
    else:
        n_cells = len(sample.lat) * len(sample.lon)

    hist, edges = np.histogram(sample["IWP"], bins=iwp_bins)
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
    sum_cre_ensemble.loc[float(key)]['SW'] = sum_cre(result, sample, IWP_bins)['SW']
    sum_cre_ensemble.loc[float(key)]['LW'] = sum_cre(result, sample, IWP_bins)['LW']
    sum_cre_ensemble.loc[float(key)]['net'] = sum_cre(result, sample, IWP_bins)['net']

# %% plot sum_cre
fig, ax = plt.subplots()
ax.plot(sum_cre_ensemble.index, sum_cre_ensemble['SW'], label='SW', color='blue')
ax.plot(sum_cre_ensemble.index, sum_cre_ensemble['LW'], label='LW', color='red')
ax.plot(sum_cre_ensemble.index, sum_cre_ensemble['net'], label='net', color='black')
ax.axhline(0, color='black', linestyle='--')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('LC fraction')
ax.set_ylabel('CRE / W m$^{-2}$')
ax.legend()


# %%
