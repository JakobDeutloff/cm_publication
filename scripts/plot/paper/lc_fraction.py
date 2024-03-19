# %% import 
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from src.hc_model import calc_lc_fraction
from src.plot_functions import plot_connectedness

# %% load data
atms = xr.open_dataset("/work/um0878/user_data/jdeutloff/icon_c3_sample/atms_full.nc")

# %% calculate cloud fractions 
iwp_bins = np.logspace(-6, np.log10(30), 51)
iwp_points = (iwp_bins[1:] + iwp_bins[:-1]) / 2
f_raw = calc_lc_fraction(atms['LWP'].where(atms['mask_height']), connected=False)
f_raw_binned = f_raw.groupby_bins(atms['IWP'].where(atms['mask_height']), iwp_bins).mean()
f_connected = calc_lc_fraction(atms['LWP'].where(atms['mask_height']), connected=atms['connected'].where(atms['mask_height'])) 
f_connected_binned = f_connected.groupby_bins(atms['IWP'].where(atms['mask_height']), iwp_bins).mean()
f_mean = f_connected_binned.where(iwp_points < 1e-1).mean()

# %% plot 
fig, ax = plt.subplots()
ax.plot(iwp_points, f_raw_binned, label='Raw', color='purple')
ax.plot(iwp_points, f_connected_binned, color='green', linestyle='--', label='Connected')
ax.axhline(f_mean, color='k', linestyle='-', label='Constant')
ax.set_xscale('log')
ax.set_xlabel('IWP / kg m$^{-2}$')
ax.set_ylabel('Low Cloud Fraction')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
# %%
plot_connectedness(atms, f_connected, iwp_bins, iwp_points, f_connected_binned, f_mean, 'lc_fraction')

# %%
