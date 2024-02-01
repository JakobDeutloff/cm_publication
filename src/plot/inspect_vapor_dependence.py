# %% import 
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from src.read_data import load_atms_and_fluxes, load_derived_vars
from src.icon_arts_analysis import cut_data
from scipy.stats import linregress
import pickle 

# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()
aux = xr.open_dataset("/work/bm1183/m301049/icon_arts_processed/fullrange_flux_mid1deg/aux.nc")

# %%  calculate column water vapor 
cell_height = atms["geometric height"].diff("pressure")
cell_height_bottom = xr.DataArray(
    data=cell_height.isel(pressure=0).values,
    dims=["lat", "lon"],
    coords={
        "pressure": fluxes_3d["pressure"][0],
        "lat": fluxes_3d["lat"],
        "lon": fluxes_3d["lon"],
    },
)
cell_height = xr.concat([cell_height_bottom, cell_height], dim="pressure")

col_h2o = (atms['H2O'] * cell_height).sum('pressure')
# %% plot R_t at locations without low clouds against IWP and color with column water vapor
mask = lw_vars['mask_hc_no_lc']

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sc = ax.scatter(
    cut_data(atms['IWP'], mask=mask),
    cut_data(lc_vars['R_t'], mask=mask),
    c=cut_data(col_h2o, mask=mask),
    s=1,
    cmap='viridis',
)

ax.set_xscale('log')
ax.set_xlabel('IWP [kg/m$^2$]')
ax.set_ylabel('R$_t$ [W/m$^2$]')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Column water vapor [kg/m$^2$]')

# %% plot R_t vs column water vapor and column water vapor vs IWP
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].scatter(
    cut_data(col_h2o, mask=mask),
    cut_data(lc_vars['R_t'], mask=mask),
    s=0.5,
    color='k'
)

axes[0].set_xlabel('Column water vapor / kg/m$^2$')
axes[0].set_ylabel('R$_t$ / W/m$^2$')

sc = axes[1].scatter(
    cut_data(atms['IWP'], mask=mask),
    cut_data(col_h2o, mask=mask),
    s=0.5,
    color='k'
)
axes[1].set_xscale('log')
axes[1].set_xlabel('IWP / kg/m$^2$')
axes[1].set_ylabel('Column water vapor / kg/m$^2$')

for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


# %% linear regression of R_t vs IWP 
x = cut_data(atms['IWP'], mask=mask).values.flatten()
y = cut_data(lc_vars['R_t'], mask=mask).values.flatten()
mask_nan_x = ~np.isnan(x)
mask_nan_y = ~np.isnan(y)
mask_nan = mask_nan_x & mask_nan_y
x = np.log10(x[mask_nan])
y = y[mask_nan] - np.mean(y[mask_nan])
res = linregress(x, y)

# %% plot R_t vs IWP and add linear regression
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(
    cut_data(atms['IWP'], mask=mask),
    cut_data(lc_vars['R_t'], mask=mask) - np.mean(cut_data(lc_vars['R_t'], mask=mask)),
    s=0.5,
    color='k'
)
x = np.logspace(-6, 1, 100)
ax.plot(
    x,
    res.intercept + res.slope * np.log10(x),
    color='r'
)
ax.set_xscale('log')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('IWP / kg/m$^2$')
ax.set_ylabel('R$_t$ / W/m$^2$')

# %% export parameters
with open('/work/bm1183/m301049/icon_arts_processed/derived_quantities/water_vapor_dependence.pkl', 'wb') as f:
    pickle.dump(res, f)

# %%
