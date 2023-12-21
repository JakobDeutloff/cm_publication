# %% import 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.read_data import load_atms_and_fluxes
from src.plot_functions import scatterplot
from scipy.stats import linregress
from src.icon_arts_analysis import cut_data
# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars = xr.open_dataset("/work/bm1183/m301049/icon_arts_processed/derived_quantities/lw_vars.nc")

# %% plot IWP vs LWP 
fig, ax = scatterplot(
    atms["IWP"].sel(lat=slice(-30, 30)),
    atms["LWP"].sel(lat=slice(-30, 30)),
    xlabel="IWP / kg m$^{-2}$",
    ylabel="LWP / kg m$^{-2}$",
    xlim=[1e-7, 1e2],
    ylim=[1e-6, 1e2],
    logx=True,
    logy=True,
)

# plot binned LWP vs IWP
IWP_bins = np.logspace(-7, 1, 50)
IWP_points= (IWP_bins[1:] + IWP_bins[:-1]) / 2
binned_LWP = atms["LWP"].where(atms['LWP'] > 1e-6).sel(lat=slice(-30, 30)).groupby_bins(atms["IWP"].sel(lat=slice(-30, 30)), IWP_bins).median()
ax.plot(IWP_points, binned_LWP, color="k", linewidth=2, label="mean LWP")

#  plot linear regression of LWP vs IWP
x_data = cut_data(atms["IWP"], mask=(atms["IWP"] > 0) & (atms['LWP'] > 1e-6)).values.flatten()
nan_x = np.isnan(x_data)
y_data = cut_data(atms["LWP"], (atms["IWP"] > 0) & (atms['LWP'] > 1e-6)).values.flatten()
nan_y = np.isnan(y_data)
mask = (~nan_x) & (~nan_y)
x_data = np.log10(x_data[mask])
y_data = np.log10(y_data[mask])

result = linregress(x_data, y_data)
lin_reg = result.intercept + result.slope * np.log10(IWP_points)
ax.plot(IWP_points, 10**lin_reg, color="k", linestyle="--", label="linear regression")


# %%
