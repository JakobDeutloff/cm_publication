# %%
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# %% load cloudsat data for 2015
path_cloudsat = "/work/bm1183/m301049/cloudsat/"
cloudsat = xr.open_dataset(path_cloudsat + "2015-07-01_2016-07-01_fwp.nc")
cloudsat = cloudsat.to_pandas()

# %% select tropics
lat_mask = (cloudsat["lat"] <= 30) & (cloudsat["lat"] >= -30)
cloudsat_trop = cloudsat[lat_mask]

# %% calculate share of zeros
zeros_cloudsat = (
    cloudsat_trop["ice_water_path"].where(cloudsat_trop["ice_water_path"] == 0).count()
    / cloudsat_trop["ice_water_path"].count()
)

# %% IWP histogramms 
fig, axes = plt.subplots(2, 1, figsize=(8, 6))


# Don't divide by binwidth 
bins = np.logspace(-2, 5, num=100)
powerlaw = (bins ** -1.3) * 10 ** -1.1
hist, edges = np.histogram(
    cloudsat_trop["ice_water_path"], bins=bins, density=False
)
hist_norm = hist / len(cloudsat_trop["ice_water_path"])
axes[0].stairs(hist_norm, edges, color="blue")
axes[0].plot(bins[1:], np.diff(bins) * powerlaw[1:], color="red")
axes[0].set_ylabel("Relative Frequency")
axes[0].set_title('Not Divided by Binwidth')
axes[0].set_xscale("log")
axes[0].set_ylim(0, 0.01)


# divide by binwidth
hist, edges = np.histogram(
    cloudsat_trop["ice_water_path"], bins=bins, density=False
)
hist_norm = hist / (len(cloudsat_trop["ice_water_path"]) * np.diff(edges))
axes[1].stairs(hist_norm, edges, color="blue")
axes[1].plot(bins, powerlaw, color="red")
axes[1].set_ylabel('P(IWP) / (g m$^{-2})^{-1}$')
axes[1].set_yscale("log")
axes[1].set_ylim(1e-8, 1e0)
axes[1].set_title("Divided by Binwidth")
axes[1].set_xscale("log")
axes[1].set_xlabel("IWP / g m$^{-2}$")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(1e-2, 1e5)


fig.tight_layout()

# %% fit lines to IWP histograms

bins = np.logspace(-2, 6, num=100)
hist, edges = np.histogram(
    cloudsat_trop["ice_water_path"], bins=bins, density=False
)
x_vals = edges[1:]
x_vals_log = np.log10(x_vals)
y_vals_log = np.log10(hist_norm)
x_vals_log = x_vals_log[~np.isinf(y_vals_log)]
y_vals_log = y_vals_log[~np.isinf(y_vals_log)]
mask_x = x_vals_log > -1
mask_y = y_vals_log > -7


slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals_log[mask_x & mask_y], y_vals_log[mask_x & mask_y])

# plot fit and histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.stairs(hist_norm, edges, color="blue", label="2C-ICE")
ax.plot(x_vals, x_vals ** slope * 10 ** intercept , color="red", label="fit")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-2, 1e5)
ax.set_ylim(1e-8, 1e0)

ax.legend()


# %%
