# %% import
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from src.read_data import load_atms_and_fluxes
from src.plot_functions import scatterplot
from scipy.stats import linregress
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

# %% read data
atms, fluxes_3d, fluxes_3d_noice, lw_vars = load_atms_and_fluxes()

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
                albedo_lc
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
    ax.set_xlabel("IWP / kg m$^{-2}$")
    ax.set_xlim([1e-4, 5e1])

fig.colorbar(pcol, label="High Cloud Albedo", location="bottom", ax=axes[:], shrink=0.8)

# %% fit polynom to weighted and interpolated albedo
lwp_mask = (LWP_points <=10) & (LWP_points >= 1e-5)
p = np.polyfit(np.log10(LWP_points[lwp_mask ]), mean_lc_albedo_SW_interp[lwp_mask] , 9)
poly = np.poly1d(p)
fitted_curve = poly(np.log10(LWP_points[lwp_mask]))

# %% fit logistic function to weighted and interpolated albedo
def logistic(x, L, x0, k):
    return L / (1 + np.exp(-k*(x-x0)))

x = np.log10(LWP_points)
y = mean_lc_albedo_SW_interp
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
    cut_data(albedo_lc),
    cut_data(fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_down"]),
    xlabel="LWP / kg m$^{-2}$",
    ylabel="Albedo",
    cbar_label="SW Down / W m$^{-2}$",
    xlim=[1e-7, 1e2],
)

ax.plot(LWP_points, mean_lc_albedo_SW_interp, color="k", linestyle="-")
ax.plot(LWP_points[lwp_mask], fitted_curve, color="r", linestyle="--")
ax.plot(LWP_points, logistic_curve, color="b", linestyle="--")


# %% calculate R_t
R_t = fluxes_3d_noice.isel(pressure=-1)["allsky_lw_up"]
clearsky_flux = (
    fluxes_3d_noice.isel(pressure=-1)["clearsky_lw_up"].sel(lat=slice(-30, 30)).mean()
)

# %% linear regression of R_t vs LWP for LWP > 1e-6
mask_lwp = atms["LWP"] > 1e-6
x_data = np.log10(cut_data(atms["LWP"], mask_lwp)).values.flatten()
x_data = x_data[~np.isnan(x_data)]
y_data = cut_data(R_t, mask_lwp).values.flatten()
y_data = y_data[~np.isnan(y_data)]

result = linregress(x_data, y_data)

# %% plot R_t vs LWP
fig, ax = scatterplot(
    cut_data(atms["LWP"]),
    cut_data(R_t),
    cut_data(fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_down"]),
    xlabel="LWP / kg m$^{-2}$",
    ylabel="R$_t$",
    cbar_label="SW Down / W m$^{-2}$",
    xlim=[1e-6, 1e2],
)
R_t_binned = cut_data(R_t).groupby_bins(cut_data(atms["LWP"]), np.logspace(-6, 2, 100)).mean()
R_t_binned.plot(ax=ax, color="red", linestyle="-")
ax.axhline(clearsky_flux, color="k", linestyle="--")
lwp_points = np.logspace(-6, 2, 100)
ax.plot(lwp_points, result.intercept + result.slope * np.log10(lwp_points), color="k", linestyle="--")

# %%
