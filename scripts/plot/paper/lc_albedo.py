# %% import 
import numpy as np
import matplotlib.pyplot as plt
from src.read_data import load_atms_and_fluxes, load_derived_vars, load_average_lc_parameters
from src.helper_functions import cut_data, cut_data_mixed
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

# %% load data 
atms, fluxes, fluxes_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()
const_lc_quantities = load_average_lc_parameters()

# %% plot alpha_t colored by LWP and mean values
albedo_cs = cut_data(fluxes["albedo_clearsky"]).mean().values
albedo_lc = const_lc_quantities["a_t"]
lc_frac = 0.24
albedo_mixed = lc_frac * albedo_lc + (1 - lc_frac) * albedo_cs

# Create colormap
colors = ["black", "grey", "blue"]
cmap = LinearSegmentedColormap.from_list("my_cmap", colors)

fig, ax = plt.subplots()
sc = ax.scatter(
    cut_data(atms["IWP"], lw_vars["mask_height"]),
    cut_data_mixed(fluxes_noice['albedo_clearsky'], fluxes_noice["albedo_allsky"], lw_vars["mask_height"], atms["connected"]),
    c=cut_data_mixed((atms['LWP'] * 0) + 1e-12, atms["LWP"], lw_vars["mask_height"], atms["connected"]),
    cmap=cmap,
    norm=LogNorm(vmin=1e-6, vmax=1e0),
    s=1
)
ax.axhline(albedo_cs, color='k', linestyle='--', label='Clearsky')
ax.axhline(albedo_lc, color='b', linestyle='--', label='Low Cloud')
ax.axhline(albedo_mixed, color='r', linestyle='--', label='Mixed')

ax.set_xscale('log')
ax.set_xlim(1e-6, 1e1)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('IWP / kg m$^{-2}$')
ax.set_ylabel(r'$\alpha_{\mathrm{t}}$')

ax.legend(bbox_to_anchor=(1, -0.15), ncols=3)
fig.colorbar(sc, label='LWP / kg m$^{-2}$', extend='both')
fig.savefig('plots/paper/lc_albedo.png', dpi=400, bbox_inches='tight')

# %%
