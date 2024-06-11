"""
make series of control plots of binned atms data and derived variables
"""
# %% import
import matplotlib.pyplot as plt
from src.read_data import load_binned_atms
import numpy as np
import xarray as xr

# %% load data
atms = load_binned_atms()
atms_c3 = xr.open_dataset('/work/bm1183/m301049/nextgems_profiles/cycle3/profiles_processed_formatted.nc')

# %% find profiles with high cloud tops above 350 hPa
mask_hc_no_lc = (atms["LWP"] < 1e-6)
mask_height = atms.sel(level_full=atms["hc_top_idx"])['pressure'] < 35000
mask_height_c3 = atms_c3.sel(level_full=atms_c3["hc_top_idx"])['pressure'] < 35000
unvalid_profiles = (mask_height * 1).mean() * 100
unvalid_profiles_c3 = (mask_height_c3 * 1).mean() * 100
print(f'Valid profiles: {unvalid_profiles.values:.2f}%')
print(f'Valid profiles c3: {unvalid_profiles_c3.values:.2f}%')
iwp_bins = np.logspace(-6, 1, 51)

# %% plot hc temperature against IWP 
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True)

temp = atms['hc_temperature'].groupby('iwp').mean(['sw', 'profile'])
temp_c3 = atms_c3['hc_temperature'].groupby('iwp').mean(['sw', 'profile'])
axes[0, 0].scatter(atms['IWP'], atms["hc_temperature"], s=0.1, color='k')
axes[0, 0].plot(atms['iwp'], temp, color='r')
axes[0, 0].plot(atms['iwp'], temp_c3, color='b')
axes[0, 0].invert_yaxis()
axes[0, 0].set_ylabel('hc_temperature / K')
axes[0, 0].set_xscale('log')

axes[0, 1].scatter(atms['IWP'], atms["LWP"], s=0.1, color='k')
axes[0, 1].set_ylabel('LWP / kg m$^{-2}$')
axes[0, 1].set_ylim(1e-10, 1e1)
axes[0, 1].set_yscale('log')

lc_fraction = ((atms['LWP'] > 1e-6 )*1).groupby('iwp').mean(['sw', 'profile'])
lc_fraction_c3 = ((atms_c3['LWP'] > 1e-6 )*1).groupby('iwp').mean(['sw', 'profile'])
axes[1, 0].plot(atms['iwp'], lc_fraction, color='r')
axes[1, 0].plot(atms['iwp'], lc_fraction_c3, color='b')

# %%
