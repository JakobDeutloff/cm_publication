# %% import
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from src.read_data import load_atms_and_fluxes
from src.plot_functions import scatterplot
from scipy.stats import linregress
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import pandas as pd
import pickle

# %% read data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars = xr.open_dataset("/work/bm1183/m301049/icon_arts_processed/derived_quantities/lw_vars.nc")

 # %% initialize dataset
lc_vars = xr.Dataset()
mean_lc_vars = pd.DataFrame()

# %% mask out night values 
mask_night = fluxes_3d['allsky_sw_down'].isel(pressure=-1) == 0 
(mask_night*1).plot.pcolormesh()

# %% calculate albedo
albedo_allsky = np.abs(
    fluxes_3d_noice.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_3d_noice.isel(pressure=-1)["allsky_sw_down"]
)
albedo_clearsky = np.abs(
    fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_up"]
    / fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_down"]
)
albedo_lc = (albedo_allsky - albedo_clearsky) / (
    albedo_clearsky * (albedo_allsky - 2) + 1
)

lc_vars['albedo_allsky'] = albedo_allsky
lc_vars['albedo_clearsky'] = albedo_clearsky

# %% average and interpolate albedo 
LWP_bins = np.logspace(-6, 2, num=70)
LWP_points = (LWP_bins[1:] + LWP_bins[:-1]) / 2
SW_down_bins = np.linspace(0, 1360, 30)
binned_lc_albedo = np.zeros([len(LWP_bins) - 1, len(SW_down_bins) - 1]) * np.nan

for i in range(len(LWP_bins) - 1):
    LWP_mask = (atms["LWP"] > LWP_bins[i]) & (atms["LWP"] <= LWP_bins[i + 1])
    for j in range(len(SW_down_bins) - 1):
        SW_mask = (fluxes_3d["allsky_sw_down"].isel(pressure=-1) > SW_down_bins[j]) & (
            fluxes_3d["allsky_sw_down"].isel(pressure=-1) <= SW_down_bins[j + 1]
        )
        binned_lc_albedo[i, j] = float(
            (
                lc_vars["albedo_allsky"]
                .where(LWP_mask & SW_mask)
                .sel(lat=slice(-30, 30))
            )
            .mean()
            .values
        )

# %% interpolate albedo bins
non_nan_indices = np.array(np.where(~np.isnan(binned_lc_albedo)))
non_nan_values = binned_lc_albedo[~np.isnan(binned_lc_albedo)]
nan_indices = np.array(np.where(np.isnan(binned_lc_albedo)))
interpolated_values = griddata(
    non_nan_indices.T, non_nan_values, nan_indices.T, method="linear"
)
binned_lc_albedo_interp = binned_lc_albedo.copy()
binned_lc_albedo_interp[np.isnan(binned_lc_albedo)] = interpolated_values

# %% plot albedo in SW bins
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

pcol = axes[0].pcolormesh(
    LWP_bins, SW_down_bins, binned_lc_albedo.T , cmap="viridis"
)
axes[1].pcolormesh(
    LWP_bins, SW_down_bins, binned_lc_albedo_interp.T , cmap="viridis"
)

axes[0].set_ylabel("SWin at TOA / W m$^{-2}$")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("LWP / kg m$^{-2}$")
    ax.set_xlim([1e-4, 5e1])

fig.colorbar(pcol, label="High Cloud Albedo", location="bottom", ax=axes[:], shrink=0.8)

# %% average over SW albedo bins
mean_lc_albedo_SW = np.zeros(len(LWP_points))
mean_lc_albedo_SW_interp = np.zeros(len(LWP_points))
SW_down = (SW_down_bins[1:] + SW_down_bins[:-1]) / 2  # center of SW bins
for i in range(len(LWP_bins) - 1):
    nan_mask = ~np.isnan(binned_lc_albedo[i, :])
    mean_lc_albedo_SW[i] = np.sum(
        binned_lc_albedo[i, :][nan_mask] * SW_down[nan_mask]
    ) / np.sum(SW_down[nan_mask])
    nan_mask_interp = ~np.isnan(binned_lc_albedo_interp[i, :])
    mean_lc_albedo_SW_interp[i] = np.sum(
        binned_lc_albedo_interp[i, :][nan_mask_interp] * SW_down[nan_mask_interp]
    ) / np.sum(SW_down[nan_mask_interp])

mean_lc_vars.index = LWP_points
mean_lc_vars['binned_albedo'] = mean_lc_albedo_SW
mean_lc_vars['interpolated_albedo'] = mean_lc_albedo_SW_interp

# %% fit polynom to weighted and interpolated albedo
lwp_mask = (LWP_points <=10) & (LWP_points >= 1e-5)
p = np.polyfit(np.log10(LWP_points[lwp_mask ]), mean_lc_vars['interpolated_albedo'][lwp_mask] , 9)
poly = np.poly1d(p)
fitted_curve = poly(np.log10(LWP_points[lwp_mask]))

# %% fit logistic function to weighted and interpolated albedo
def logistic(x, L, x0, k):
    return L / (1 + np.exp(-k*(x-x0)))

x = np.log10(LWP_points)
y = mean_lc_vars['interpolated_albedo']
nan_mask = ~np.isnan(y)
x = x[nan_mask]
y = y[nan_mask]

popt, pcov = curve_fit(logistic, x, y)
logistic_curve = logistic(np.log10(LWP_points), *popt)

# %% plot albedo vs LWP
def cut_data(data, mask=True):
    return data.sel(lat=slice(-30, 30)).where(mask)


fig, ax = scatterplot(
    cut_data(atms["LWP"]),
    cut_data(lc_vars["albedo_lc"]),
    cut_data(fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_down"]),
    xlabel="LWP / kg m$^{-2}$",
    ylabel="Albedo",
    cbar_label="SW Down / W m$^{-2}$",
    xlim=[1e-14, 1e2],
)

ax.axhline(albedo_clearsky.sel(lat=slice(-30, 30)).mean(), color="k", linestyle="-")
ax.axhline(albedo_allsky.where(atms["LWP"] < 1e-7).sel(lat=slice(-30, 30)).mean(), color="k", linestyle="-")
ax.plot(mean_lc_vars["interpolated_albedo"], color="k", linestyle="-")
ax.plot(LWP_points[lwp_mask], fitted_curve, color="r", linestyle="--")
ax.plot(LWP_points, logistic_curve, color="b", linestyle="--")


# %% calculate R_t
R_t = fluxes_3d_noice.isel(pressure=-1)["allsky_lw_up"]
clearsky_flux = (
    fluxes_3d_noice.isel(pressure=-1)["clearsky_lw_up"].sel(lat=slice(-30, 30)).mean()
)
lc_vars["R_t"] = R_t

# %% linear regression of R_t vs LWP for LWP > 1e-6
mask_lwp = atms["LWP"] > 1e-6
x_data = np.log10(cut_data(atms["LWP"], mask_lwp)).values.flatten()
x_data = x_data[~np.isnan(x_data)]
y_data = cut_data(R_t, mask_lwp).values.flatten()
y_data = y_data[~np.isnan(y_data)]

result = linregress(x_data, y_data)

# %% average R_t over LWP bins
R_t_binned = cut_data(R_t).groupby_bins(cut_data(atms["LWP"]), LWP_bins).mean()
mean_lc_vars['binned_R_t'] = R_t_binned

# %% plot R_t vs LWP
fig, ax = scatterplot(
    cut_data(atms["LWP"]),
    cut_data(lc_vars["R_t"]),
    cut_data(fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_down"]),
    xlabel="LWP / kg m$^{-2}$",
    ylabel="R$_t$",
    cbar_label="SW Down / W m$^{-2}$",
    xlim=[1e-6, 1e2],
)
mean_lc_vars["binned_R_t"].plot(ax=ax, color="red", linestyle="-")
ax.axhline(clearsky_flux, color="k", linestyle="--")
lwp_points = np.logspace(-6, 2, 100)
ax.plot(lwp_points, result.intercept + result.slope * np.log10(lwp_points), color="k", linestyle="--")

# %% save variables 
path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"

lc_vars.to_netcdf(path + "lc_vars.nc")

with open(path + "mean_lc_vars.pkl", "wb") as f:
    pickle.dump(mean_lc_vars, f)

with open(path + "Rt_params.pkl", "wb") as f:
    pickle.dump(result, f)

with open(path + "lc_albedo_params.pkl", "wb") as f:
    pickle.dump(popt, f)


# %%
