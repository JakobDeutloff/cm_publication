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
lw_vars = xr.open_dataset(
    "/work/bm1183/m301049/icon_arts_processed/derived_quantities/lw_vars.nc"
)
aux = xr.open_dataset(
    "/work/bm1183/m301049/icon_arts_processed/fullrange_flux_mid1deg_noice/aux.nc"
)

# %% initialize dataset
lc_vars = xr.Dataset()
mean_lc_vars = pd.DataFrame()

# %% mask of night values
mask_night = fluxes_3d["allsky_sw_down"].isel(pressure=-1) == 0

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

lc_vars["albedo_allsky"] = albedo_allsky
lc_vars["albedo_clearsky"] = albedo_clearsky

# %% average and interpolate albedo
LWP_bins = np.logspace(-14, 2, num=150)
LWP_points = (LWP_bins[1:] + LWP_bins[:-1]) / 2
SW_down_bins = np.linspace(0, 1360, 30)
binned_allsky_albedo = np.zeros([len(LWP_bins) - 1, len(SW_down_bins) - 1]) * np.nan

# allsky albedo depends on SW and LWP
for i in range(len(LWP_bins) - 1):
    LWP_mask = (atms["LWP"] > LWP_bins[i]) & (atms["LWP"] <= LWP_bins[i + 1])
    for j in range(len(SW_down_bins) - 1):
        SW_mask = (fluxes_3d["allsky_sw_down"].isel(pressure=-1) > SW_down_bins[j]) & (
            fluxes_3d["allsky_sw_down"].isel(pressure=-1) <= SW_down_bins[j + 1]
        )
        binned_allsky_albedo[i, j] = float(
            (lc_vars["albedo_allsky"].where(LWP_mask & SW_mask).sel(lat=slice(-30, 30)))
            .mean()
            .values
        )

# %% interpolate albedo bins
non_nan_indices = np.array(np.where(~np.isnan(binned_allsky_albedo)))
non_nan_values = binned_allsky_albedo[~np.isnan(binned_allsky_albedo)]
nan_indices = np.array(np.where(np.isnan(binned_allsky_albedo)))
interpolated_values = griddata(
    non_nan_indices.T, non_nan_values, nan_indices.T, method="linear"
)
binned_allsky_albedo_interp = binned_allsky_albedo.copy()
binned_allsky_albedo_interp[np.isnan(binned_allsky_albedo)] = interpolated_values

# %% plot albedo in SW bins
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

pcol = axes[0].pcolormesh(
    LWP_bins, SW_down_bins, binned_allsky_albedo.T, cmap="viridis"
)
axes[1].pcolormesh(
    LWP_bins, SW_down_bins, binned_allsky_albedo_interp.T, cmap="viridis"
)

axes[0].set_ylabel("SWin at TOA / W m$^{-2}$")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("LWP / kg m$^{-2}$")
    ax.set_xlim([1e-4, 5e1])

fig.colorbar(pcol, label="Allsky Albedo", location="bottom", ax=axes[:], shrink=0.8)

# %% average over SW albedo bins
mean_allsky_albedo = np.zeros(len(LWP_points))
mean_allsky_albedo_interp = np.zeros(len(LWP_points))
SW_down = (SW_down_bins[1:] + SW_down_bins[:-1]) / 2  # center of SW bins
for i in range(len(LWP_bins) - 1):
    nan_mask = ~np.isnan(binned_allsky_albedo[i, :])
    mean_allsky_albedo[i] = np.sum(
        binned_allsky_albedo[i, :][nan_mask] * SW_down[nan_mask]
    ) / np.sum(SW_down[nan_mask])
    nan_mask_interp = ~np.isnan(binned_allsky_albedo_interp[i, :])
    mean_allsky_albedo_interp[i] = np.sum(
        binned_allsky_albedo_interp[i, :][nan_mask_interp] * SW_down[nan_mask_interp]
    ) / np.sum(SW_down[nan_mask_interp])

mean_lc_vars.index = LWP_points
mean_lc_vars.index.name = "LWP"
mean_lc_vars["binned_albedo"] = mean_allsky_albedo
mean_lc_vars["interpolated_albedo"] = mean_allsky_albedo_interp
mean_clearsky_albedo = lc_vars["albedo_clearsky"].sel(lat=slice(-30, 30)).mean().values

# calculate average albedo for LWP > 1e-6
number_of_points = np.zeros(len(LWP_points))
for i in range(len(LWP_bins) - 1):
    LWP_mask = (atms["LWP"] > LWP_bins[i]) & (atms["LWP"] <= LWP_bins[i + 1])
    number_of_points[i] = (
        lc_vars["albedo_allsky"].where(LWP_mask & ~mask_night).count().values
    )

mask_lwp_bins = LWP_points > 1e-6
average_albedo = np.sum(
    mean_lc_vars["interpolated_albedo"][mask_lwp_bins] * number_of_points[mask_lwp_bins]
) / np.sum(number_of_points[mask_lwp_bins][mean_lc_vars["interpolated_albedo"][mask_lwp_bins] > 0])


# %% fit logistic function to weighted and interpolated albedo
def logistic(x, L, x0, k, j):
    return L / (1 + np.exp(-k * (x - x0))) + j


x = np.log10(LWP_points)
y = mean_lc_vars["interpolated_albedo"]
y[x < -4] = mean_clearsky_albedo  # define albedo as clearsky albedo below LWP of 10^-4
nan_mask = ~np.isnan(y)
x = x[nan_mask]
y = y[nan_mask]

popt, pcov = curve_fit(logistic, x, y)
logistic_curve = logistic(np.log10(LWP_points), *popt)


# %% plot albedo vs LWP
def cut_data(data, mask=True):
    return data.sel(lat=slice(-30, 30)).where(mask)


fig, ax = scatterplot(
    cut_data(atms["LWP"], ~mask_night),
    cut_data(lc_vars["albedo_allsky"], ~mask_night),
    cut_data(fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_down"], ~mask_night),
    xlabel="LWP / kg m$^{-2}$",
    ylabel="Albedo",
    cbar_label="SW Down / W m$^{-2}$",
    xlim=[1e-14, 1e2],
)

ax.axhline(mean_clearsky_albedo, color="k", linestyle="--", label="Mean clearsky")
ax.axhline(average_albedo, color="grey", linestyle="--", label='Mean low cloud')
ax.plot(mean_lc_vars["interpolated_albedo"], color="k", linestyle="-", label="Mean")
ax.plot(LWP_points, logistic_curve, color="r", linestyle="--", label="Fit")
ax.legend()
fig.tight_layout()
fig.savefig("plots/albedo_vs_LWP.png", dpi=300)

# %% plot clearsky albedo

fig, ax = scatterplot(
    cut_data(fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_down"], ~mask_night),
    cut_data(lc_vars["albedo_clearsky"], ~mask_night),
    cut_data(aux["land sea mask"], ~mask_night),
    logx=False,
    xlabel="SW Down / W m$^{-2}$",
    ylabel="Clearsky Albedo",
    cbar_label="Land Sea Mask",
)
#  mean ocean albedo
mean_ocean_albedo = (
    lc_vars["albedo_clearsky"]
    .where((aux["land sea mask"] < 0) & ~mask_night)
    .sel(lat=slice(-30, 30))
    .mean()
)
print(f"Ocean albedo: {mean_ocean_albedo.values.round(4)}")


# %% calculate R_t
R_t = fluxes_3d_noice.isel(pressure=-1)["allsky_lw_up"]
clearsky_R_t = (
    fluxes_3d_noice.isel(pressure=-1)["clearsky_lw_up"].sel(lat=slice(-30, 30)).mean()
).values
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
mean_lc_vars["binned_R_t"] = R_t_binned
lc_R_t = cut_data(R_t).where(mask_lwp).mean().values

# %% plot R_t vs LWP
fig, ax = plt.subplots()

ax.scatter(cut_data(atms["LWP"]),cut_data(lc_vars["R_t"]), marker=".", s=1, color="blue")

mean_lc_vars["binned_R_t"].plot(ax=ax, color="k", linestyle="-", label='Mean')
ax.axhline(clearsky_R_t, color="k", linestyle="--", label='Mean clearsky')
ax.axhline(lc_R_t, color="grey", linestyle="--", label='Mean low cloud')
lwp_points = np.logspace(-14, 2, 100)

# regression line
y = result.intercept + result.slope * np.log10(lwp_points)
mask = y > clearsky_R_t
y[~mask] = clearsky_R_t
ax.plot(
    lwp_points,
    y,
    color="red",
    linestyle="--",
    label='Fit'
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xscale("log")
ax.set_xlabel("LWP / kg m$^{-2}$")
ax.set_ylabel("R$_t$ / W m$^{-2}$")
ax.legend()
fig.tight_layout()
fig.savefig("plots/R_t_vs_LWP.png", dpi=300)


# %% save variables
path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"

lc_vars.to_netcdf(path + "lc_vars.nc")

with open(path + "mean_lc_vars.pkl", "wb") as f:
    pickle.dump(mean_lc_vars, f)

with open(path + "R_t_params.pkl", "wb") as f:
    pickle.dump(result, f)

with open(path + "alpha_t_params.pkl", "wb") as f:
    pickle.dump(popt, f)

with open(path + "average_a_t_r_t.pkl", "wb") as f:
    pickle.dump({'a_t': average_albedo, 'R_t': lc_R_t}, f)

# %%
