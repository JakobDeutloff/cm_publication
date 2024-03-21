# %% import
import numpy as np
import matplotlib.pyplot as plt
from src.read_data import (
    load_atms_and_fluxes,
    load_derived_vars,
    load_average_lc_parameters,
    load_parameters,
)
from src.helper_functions import cut_data, cut_data_mixed
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

# %% load data
atms, fluxes, fluxes_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()
const_lc_quantities = load_average_lc_parameters()
parameters = load_parameters()

# %% plot R_t vs IWP with LWP colors

colors = ["black", "grey", "blue"]
cmap = LinearSegmentedColormap.from_list("my_cmap", colors)

fig, ax = plt.subplots()
sc = ax.scatter(
    cut_data(atms["IWP"], lw_vars["mask_height"]),
    cut_data_mixed(
        fluxes_noice["clearsky_lw_up"].isel(pressure=-1),
        fluxes_noice["allsky_lw_up"].isel(pressure=-1),
        lw_vars["mask_height"],
        atms["connected"],
    ),
    c=cut_data_mixed((atms['LWP'] * 0) + 1e-12, atms["LWP"], lw_vars["mask_height"], atms["connected"]),
    cmap=cmap,
    norm=LogNorm(vmin=1e-6, vmax=1e0),
    s=1,
)
ax.set_xscale("log")

# %%
