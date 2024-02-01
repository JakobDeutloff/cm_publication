# %% imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.interpolate import griddata
from src.read_data import load_atms_and_fluxes
from src.plot_functions import scatterplot
from scipy.optimize import curve_fit

# %% load  data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars = xr.open_dataset("/work/bm1183/m301049/icon_arts_processed/derived_quantities/lw_vars.nc")


# %% calculate high cloud albedo
sw_vars = xr.Dataset()
mean_sw_vars = pd.DataFrame()

def calc_hc_albedo(a_cs, a_as):
    return (a_as - a_cs) / (a_cs * (a_as-2) + 1)

sw_vars["noice_albedo"] = np.abs(
    fluxes_3d_noice.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_3d_noice.isel(pressure=-1)["allsky_sw_down"]
)
sw_vars["allsky_albedo"] = np.abs(
    fluxes_3d.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_3d.isel(pressure=-1)["allsky_sw_down"]
)
sw_vars["high_cloud_albedo"] = calc_hc_albedo(sw_vars["noice_albedo"], sw_vars["allsky_albedo"])

# %% calculate mean albedos by weighting with the incoming SW radiation in IWP bins
IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
SW_down_bins = np.linspace(0, 1360, 30)
binned_hc_albedo = np.zeros([len(IWP_bins) - 1, len(SW_down_bins) - 1]) * np.nan

for i in range(len(IWP_bins) - 1):
    IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] <= IWP_bins[i + 1])
    for j in range(len(SW_down_bins) - 1):
        SW_mask = (fluxes_3d["allsky_sw_down"].isel(pressure=-1) > SW_down_bins[j]) & (
            fluxes_3d["allsky_sw_down"].isel(pressure=-1) <= SW_down_bins[j + 1]
        )
        binned_hc_albedo[i, j] = float(
            (
                sw_vars["high_cloud_albedo"]
                .where(IWP_mask & SW_mask & lw_vars['mask_height'] & lw_vars['mask_hc_no_lc'])
                .sel(lat=slice(-30, 30))
            )
            .mean()
            .values
        )

# %% interpolate albedo bins
non_nan_indices = np.array(np.where(~np.isnan(binned_hc_albedo)))
non_nan_values = binned_hc_albedo[~np.isnan(binned_hc_albedo)]
nan_indices = np.array(np.where(np.isnan(binned_hc_albedo)))
interpolated_values = griddata(
    non_nan_indices.T, non_nan_values, nan_indices.T, method="linear"
)
binned_hc_albedo_interp = binned_hc_albedo.copy()
binned_hc_albedo_interp[np.isnan(binned_hc_albedo)] = interpolated_values

# %% plot albedo in SW bins
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

pcol = axes[0].pcolormesh(
    IWP_bins, SW_down_bins, binned_hc_albedo.T , cmap="viridis"
)
axes[1].pcolormesh(
    IWP_bins, SW_down_bins, binned_hc_albedo_interp.T , cmap="viridis"
)

axes[0].set_ylabel("SWin at TOA / W m$^{-2}$")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("IWP / kg m$^{-2}$")
    ax.set_xlim([1e-4, 5e1])

fig.colorbar(pcol, label="High Cloud Albedo", location="bottom", ax=axes[:], shrink=0.8)

# %% average over SW albedo bins
mean_hc_albedo_SW = np.zeros(len(IWP_points))
mean_hc_albedo_SW_interp = np.zeros(len(IWP_points))
SW_down = (SW_down_bins[1:] + SW_down_bins[:-1]) / 2  # center of SW bins
for i in range(len(IWP_bins) - 1):
    nan_mask = ~np.isnan(binned_hc_albedo[i, :])
    mean_hc_albedo_SW[i] = np.sum(
        binned_hc_albedo[i, :][nan_mask] * SW_down[nan_mask]
    ) / np.sum(SW_down[nan_mask])
    nan_mask_interp = ~np.isnan(binned_hc_albedo_interp[i, :])
    mean_hc_albedo_SW_interp[i] = np.sum(
        binned_hc_albedo_interp[i, :][nan_mask_interp] * SW_down[nan_mask_interp]
    ) / np.sum(SW_down[nan_mask_interp])


mean_sw_vars.index = IWP_points
mean_sw_vars.index.name = "IWP"
mean_sw_vars["binned_albedo"] = mean_hc_albedo_SW
mean_sw_vars["interpolated_albedo"] = mean_hc_albedo_SW_interp

# %% fit logistic function to mean albedo
def logistic(x, L, x0, k):
    return L / (1 + np.exp(-k*(x-x0)))

x = np.log10(IWP_points)
y = mean_sw_vars['interpolated_albedo']
nan_mask = ~np.isnan(y)
x = x[nan_mask]
y = y[nan_mask]

popt, pcov = curve_fit(logistic, x, y)
logistic_curve = logistic(np.log10(IWP_points), *popt)

# %% plot fitted albedo in scatterplot with IWP

def cut_data(data, mask):
    return data.where(mask).sel(lat=slice(-30, 30))

fig, ax = scatterplot(
    cut_data(atms["IWP"], lw_vars["mask_height"] & lw_vars["mask_hc_no_lc"]),
    cut_data(sw_vars["high_cloud_albedo"], lw_vars["mask_height"] & lw_vars["mask_hc_no_lc"]),
    cut_data(fluxes_3d["allsky_sw_down"].isel(pressure=-1), lw_vars["mask_height"] & lw_vars["mask_hc_no_lc"]),
    "IWP / kg m$^{-2}$",
    "High Cloud Albedo",
    cbar_label="SWin at TOA / W m$^{-2}$",
    xlim=[1e-5, 1e1],
    ylim=[0, 1],
)


ax.plot(mean_sw_vars['interpolated_albedo'], label="Mean Albedo", color="k")
ax.plot(IWP_points, logistic_curve, label="Fitted Logistic", color="red", linestyle='--')
ax.legend()

fig.savefig("plots/albedo.png", dpi=300)
plt.show()

# %% save coefficients as pkl file
path = '/work/bm1183/m301049/icon_arts_processed/derived_quantities/'
sw_vars.to_netcdf(path + "sw_vars.nc")

with open(path + "hc_albedo_params.pkl", "wb") as f:
    pickle.dump(popt, f)

with open(path + 'mean_sw_vars.pkl', 'wb') as f:
    pickle.dump(mean_sw_vars, f)    

# %%
