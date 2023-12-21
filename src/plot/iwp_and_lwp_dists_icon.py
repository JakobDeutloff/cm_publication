# %% import 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# %% load freddis data
path = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg/"
atms = xr.open_dataset(path + run + "atms_full.nc")

# %% plot distribution of IWP and LWP
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
IWP_bins = np.logspace(-6, 2, 100)
atms["IWP"].plot.hist(ax=axes[0], bins=IWP_bins)
axes[0].set_xscale("log")
axes[0].set_xlabel("IWP [kg/m^2]")
axes[0].set_ylabel("Relative Frequency")

# %%
