# %% import
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from src.read_data import load_atms_and_fluxes
from src.plot_functions import scatterplot
from src.helper_functions import cut_data, cut_data_mixed
from scipy.stats import linregress
from scipy.interpolate import griddata
import pandas as pd
import pickle
from src.hc_model import calc_lc_fraction
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# %% read data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars = xr.open_dataset("/work/bm1183/m301049/iwp_framework/mons/data/lw_vars.nc")
aux = xr.open_dataset(
    "/work/bm1183/m301049/iwp_framework/mons/raw_data/fullrange_flux_mid1deg_noice/aux.nc"
)

# %% initialize dataset
lower_trop_vars = xr.Dataset()
binned_lower_trop_vars = pd.DataFrame()

# %% set masks
mask_low_cloud = atms["LWP"] > 1e-4
mask_connected = atms["connected"] == 1

# %% calculate mean lc fraction
iwp_bins = np.logspace(-5, 1, num=50)
f_mean = atms['mask_low_cloud'].where(atms['mask_height']).sel(lat=slice(-30, 30)).mean().values.round(2)

# %% calculate albedos
albedo_allsky = np.abs(
    fluxes_3d_noice.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_3d_noice.isel(pressure=-1)["allsky_sw_down"]
)
albedo_clearsky = np.abs(
    fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_up"]
    / fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_down"]
)
lower_trop_vars["albedo_allsky"] = albedo_allsky
lower_trop_vars["albedo_clearsky"] = albedo_clearsky
alpha_t = xr.where(atms['mask_low_cloud'], albedo_allsky, albedo_clearsky)
lower_trop_vars["alpha_t"] = alpha_t

# %% average and interpolate albedo
LWP_bins = np.logspace(-14, 2, num=150)
LWP_points = (LWP_bins[1:] + LWP_bins[:-1]) / 2
lon_bins = np.linspace(-180, 180, num=36)
lon_points = (lon_bins[1:] + lon_bins[:-1]) / 2
binned_lt_albedo = np.zeros([len(LWP_bins) - 1, len(lon_bins) - 1]) * np.nan

# allsky albedo depends on SW and LWP
for i in range(len(LWP_bins) - 1):
    LWP_mask = (atms["LWP"] > LWP_bins[i]) & (atms["LWP"] <= LWP_bins[i + 1])
    for j in range(len(lon_bins) - 1):
        binned_lt_albedo[i, j] = float(
            (
                lower_trop_vars["albedo_allsky"]
                .where(LWP_mask)
                .sel(lat=slice(-30, 30), lon=slice(lon_bins[j], lon_bins[j + 1]))
            )
            .mean()
            .values
        )

# %% interpolate albedo bins
non_nan_indices = np.array(np.where(~np.isnan(binned_lt_albedo)))
non_nan_values = binned_lt_albedo[~np.isnan(binned_lt_albedo)]
nan_indices = np.array(np.where(np.isnan(binned_lt_albedo)))
interpolated_values = griddata(non_nan_indices.T, non_nan_values, nan_indices.T, method="linear")
binned_lt_albedo_interp = binned_lt_albedo.copy()
binned_lt_albedo_interp[np.isnan(binned_lt_albedo)] = interpolated_values

# %% plot albedo in SW bins
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

pcol = axes[0].pcolormesh(LWP_bins, lon_bins, binned_lt_albedo.T, cmap="viridis")
axes[1].pcolormesh(LWP_bins, lon_bins, binned_lt_albedo_interp.T, cmap="viridis")

axes[0].set_ylabel("SWin at TOA / W m$^{-2}$")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("LWP / kg m$^{-2}$")
    ax.set_xlim([1e-4, 5e1])

fig.colorbar(pcol, label="Allsky Albedo", location="bottom", ax=axes[:], shrink=0.8)

# %% average over SW  bins
SW_weights = np.zeros(len(lon_points))
for i in range(len(lon_bins) - 1):
    SW_weights[i] = float(
        fluxes_3d.isel(pressure=-1)["allsky_sw_down"]
        .sel(lat=slice(-30, 30), lon=slice(lon_bins[i], lon_bins[i + 1]))
        .mean()
        .values
    )

mean_lt_albedo = np.zeros(len(LWP_points))
mean_lt_albedo_interp = np.zeros(len(LWP_points))
for i in range(len(LWP_bins) - 1):
    nan_mask = ~np.isnan(binned_lt_albedo[i, :])
    mean_lt_albedo[i] = np.sum(binned_lt_albedo[i, :][nan_mask] * SW_weights[nan_mask]) / np.sum(
        SW_weights[nan_mask]
    )
    nan_mask_interp = ~np.isnan(binned_lt_albedo_interp[i, :])
    mean_lt_albedo_interp[i] = np.sum(
        binned_lt_albedo_interp[i, :][nan_mask] * SW_weights[nan_mask]
    ) / np.sum(SW_weights[nan_mask])
binned_albedos = pd.DataFrame(np.array([mean_lt_albedo_interp, mean_lt_albedo]).T, index=LWP_points, columns=["interpolated", "raw"])


#  %% calculate mean albedos
mask_lwp_bins = LWP_points > 1e-4
number_of_points = np.zeros(len(LWP_points))
for i in range(len(LWP_bins) - 1):
    LWP_mask = (atms["LWP"] > LWP_bins[i]) & (atms["LWP"] <= LWP_bins[i + 1])
    number_of_points[i] = (
        cut_data(lower_trop_vars["albedo_allsky"])
        .where(LWP_mask & ~mask_connected & atms["mask_height"])
        .count()
        .values
    )

mean_cloud_albedo = float(
    np.sum(
        binned_albedos['interpolated'][mask_lwp_bins]
        * number_of_points[mask_lwp_bins]
    )
    / np.sum(
        number_of_points[mask_lwp_bins]
    )
)

mean_clearsky_albedo = float(
    (
        (cut_data(lower_trop_vars["albedo_clearsky"], ~atms["mask_low_cloud"] & atms["mask_height"]))
        .mean()
        .values
    )
)

# %% plot albedo vs LWP
fig, ax = scatterplot(
    cut_data(atms["LWP"], atms["mask_height"]),
    cut_data(lower_trop_vars["albedo_allsky"], atms["mask_height"]),
    cut_data(fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_down"], atms["mask_height"]),
    xlabel="LWP / kg m$^{-2}$",
    ylabel="Albedo",
    cbar_label="SW Down / W m$^{-2}$",
    xlim=[1e-14, 1e2],
)

ax.axhline(mean_clearsky_albedo, color="k", linestyle="--", label="Mean clearsky")
ax.axhline(mean_cloud_albedo, color="grey", linestyle="--", label="Mean low cloud")
ax.plot(binned_albedos['interpolated'], color="k", linestyle="-", label="Mean")
ax.legend()
fig.tight_layout()

# %% plot clearsky albedo
fig, ax = scatterplot(
    cut_data(fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_down"], atms["mask_height"]),
    cut_data(lower_trop_vars["albedo_clearsky"], atms["mask_height"]),
    cut_data(aux["land sea mask"], atms["mask_height"]),
    logx=False,
    xlabel="SW Down / W m$^{-2}$",
    ylabel="Clearsky Albedo",
    cbar_label="Land Sea Mask",
)
#  mean ocean albedo
mean_ocean_albedo = (
    lower_trop_vars["albedo_clearsky"]
    .where((aux["land sea mask"] < 0) & atms["mask_height"])
    .sel(lat=slice(-30, 30))
    .mean()
)
print(f"Ocean albedo: {mean_ocean_albedo.values.round(4)}")


# %% calculate R_t
lower_trop_vars["R_t"] = xr.where(
    (mask_connected | ~mask_low_cloud),
    fluxes_3d_noice.isel(pressure=-1)["clearsky_lw_up"],
    fluxes_3d_noice.isel(pressure=-1)["allsky_lw_up"],
)

# %% calculate mean R_t
mean_R_l = float(
    cut_data(lower_trop_vars["R_t"], atms["mask_height"] & atms['mask_low_cloud']).mean().values
)
mean_R_cs = float(
    cut_data(lower_trop_vars["R_t"], atms["mask_height"] & ~atms['mask_low_cloud']).mean().values
)

# %% linear regression of R_t vs IWP
x_data = np.log10(cut_data(atms["IWP"], atms["mask_height"])).values.flatten()
y_data = cut_data(lower_trop_vars["R_t"], atms["mask_height"]).values.flatten()
nan_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
x_data = x_data[nan_mask]
y_data = y_data[nan_mask]
y_data = y_data - np.mean(y_data)
c_h20_coeffs = linregress(x_data, y_data)

# %% plot R_t vs IWP and regression
fig, ax = plt.subplots()
ax.scatter(cut_data(atms["LWP"]), cut_data(lower_trop_vars["R_t"]), marker=".", s=1, color="blue")
colors = ["black", "grey", "blue"]
cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
sc_rt = ax.scatter(
    cut_data(atms["IWP"], atms["mask_height"]),
    cut_data(
        lower_trop_vars["R_t"],
        atms["mask_height"],
    ),
    c=cut_data_mixed(
        (atms["LWP"] * 0) + 1e-12, atms["LWP"], atms["mask_height"], atms["connected"]
    ),
    cmap=cmap,
    norm=LogNorm(vmin=1e-6, vmax=1e0),
    s=0.1,
)
iwp_bins = np.logspace(-5, 1, num=50)
iwp_points = (iwp_bins[1:] + iwp_bins[:-1]) / 2
binned_r_t = (
    cut_data(lower_trop_vars["R_t"], atms["mask_height"])
    .groupby_bins(cut_data(atms["IWP"], atms["mask_height"]), iwp_bins)
    .mean()
)
ax.plot(iwp_points, binned_r_t, color="orange", label="Mean")
ax.axhline(mean_R_cs, color="grey", linestyle="--", label="Clearsky")
ax.axhline(mean_R_l, color="navy", linestyle="--", label="Low Cloud")
fit = (
    f_mean * mean_R_l
    + (1 - f_mean) * mean_R_cs
    + c_h20_coeffs.slope * np.log10(iwp_points)
    + c_h20_coeffs.intercept
)
ax.plot(iwp_points, fit, color="red", label="Fit")

ax.set_ylabel(r"LT LW Emissions ($\mathrm{R_t}$) / $\mathrm{W ~ m^{-2}}$")
ax.legend()
ax.set_ylim(-350, -200)
ax.set_xlim(1e-5, 1e1)
ax.set_xscale("log")

# %% calculate lower tropospheric variables binned by iwp
binned_lower_trop_vars["alpha_t"] = (
    cut_data(lower_trop_vars["alpha_t"], atms["mask_height"])
    .groupby_bins(cut_data(atms["IWP"], atms["mask_height"]), iwp_bins)
    .mean()
)
binned_lower_trop_vars["R_t"] = (
    cut_data(lower_trop_vars["R_t"], atms["mask_height"])
    .groupby_bins(cut_data(atms["IWP"], atms["mask_height"]), iwp_bins)
    .mean()
)

# %% save variables
path = "/work/bm1183/m301049/iwp_framework/mons/"

lower_trop_vars.to_netcdf(path + "data/lower_trop_vars.nc")

with open(path + "data/lower_trop_vars_mean.pkl", "wb") as f:
    pickle.dump(binned_lower_trop_vars, f)

with open(path + "parameters/C_h2o_params.pkl", "wb") as f:
    pickle.dump(c_h20_coeffs, f)

with open(path + "parameters/lower_trop_params.pkl", "wb") as f:
    pickle.dump(
        {
            "a_l": mean_cloud_albedo,
            "a_cs": mean_clearsky_albedo,
            "R_l": mean_R_l,
            "R_cs": mean_R_cs,
            "f": f_mean,
        },
        f,
    )

# %%
