#%%
import xarray as xr 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# %% load data
path = "/work/bm1183/m301049/freddi_runs/"

atms = xr.open_dataset(path + "atms.nc")
fluxes_3d = xr.open_dataset(path + "fluxes_3d.nc")
fluxes_2d = xr.open_dataset(path + "fluxes_2d.nc")
aux = xr.open_dataset(path + "aux.nc")

# %% plot ice clouds 
IWC_vi = (atms['IWC'] * atms['geometric height']).sum('pressure')
IWC_vi.plot()
# %%
fig, ax = plt.subplots()
bins = np.logspace(-5, np.log10(IWC_vi.max()), num=70)
ax.hist(IWC_vi.values.flatten(), bins=bins)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0, 20])
ax.set_ylim([0, 2000])


# %%
