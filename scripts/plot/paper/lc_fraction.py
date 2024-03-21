# %% import 
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from src.hc_model import calc_lc_fraction
from src.plot_functions import plot_connectedness
from src.read_data import load_atms_and_fluxes, load_derived_vars

# %% load data
atms, fluxes, fluxes_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vards = load_derived_vars()
atms['mask_height'] = lw_vars['mask_height']

# %% calculate cloud fractions 
iwp_bins = np.logspace(-5, 1, 50)
iwp_points = (iwp_bins[1:] + iwp_bins[:-1]) / 2
f_raw = calc_lc_fraction(atms['LWP'].where(atms['mask_height']), connected=False)
f_raw_binned = f_raw.groupby_bins(atms['IWP'].where(atms['mask_height']), iwp_bins).mean()
f_connected = calc_lc_fraction(atms['LWP'].where(atms['mask_height']), connected=atms['connected'].where(atms['mask_height'])) 
f_connected_binned = f_connected.groupby_bins(atms['IWP'].where(atms['mask_height']), iwp_bins).mean()
f_mean = f_connected_binned.where(iwp_points < 3e-1).mean()

# %% plot lc fraction
fig, ax = plt.subplots()
ax.plot(iwp_points, f_raw_binned, label='Raw', color='purple')
ax.plot(iwp_points, f_connected_binned, color='blue', linestyle='--', label='Connected')
ax.axhline(f_mean, color='r', linestyle='--', label='Constant')
ax.set_xscale('log')
ax.set_xlabel('IWP / kg m$^{-2}$')
ax.set_ylabel(r'$f$')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
fig.savefig('plots/paper/lc_fraction.png', dpi=400, bbox_inches='tight')

# %% plot connectedness
liq_cld_cond = atms["LWC"] + atms["rain"]
ice_cld_cond = atms["IWC"] + atms["snow"] + atms["graupel"]
mask = lw_vars["mask_height"] & (atms["IWP"] > 1e-6) & (atms["LWP"] > 1e-6)
iwp_bins = np.logspace(-5, 1, 7)
fig, axes = plot_connectedness(atms, mask, liq_cld_cond, ice_cld_cond, mode='arts')
fig.savefig('plots/paper/connectedness.png', dpi=400, bbox_inches='tight')
# %%
