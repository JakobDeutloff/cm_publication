# %% import
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import griddata
import pandas as pd
import pickle
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# %% read data
path = "/work/bm1183/m301049/iwp_framework/ngc3/"
atms = xr.open_dataset(path + "data/atms_proc.nc")
fluxes_allsky = xr.open_dataset(path + "data/fluxes.nc")
fluxes_noice = xr.open_dataset(path + "data/fluxes_nofrozen.nc")
fluxes_noliquid = xr.open_dataset(path + "data/fluxes_noliquid.nc")

# %% initialize dataset
lower_trop_vars = xr.Dataset()

# %% set masks
mask_low_cloud = atms["LWP"] > 1e-4
mask_connected = atms["connected"] == 1

# %% calculate mean lc fraction
f_mean = (
    atms["mask_low_cloud"].where(atms["mask_height"] & (atms["IWP"] < 1)).mean().values.round(2)
)

# %% calculate albedos
albedo_allsky = np.abs(
    fluxes_noice.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_noice.isel(pressure=-1)["allsky_sw_down"]
)
albedo_clearsky = np.abs(
    fluxes_allsky.isel(pressure=-1)["clearsky_sw_up"]
    / fluxes_allsky.isel(pressure=-1)["clearsky_sw_down"]
)
lower_trop_vars["albedo_allsky"] = albedo_allsky
lower_trop_vars["albedo_clearsky"] = albedo_clearsky
alpha_t = xr.where(atms["mask_low_cloud"], albedo_allsky, albedo_clearsky)
lower_trop_vars["alpha_t"] = alpha_t

# %% average and interpolate albedo
LWP_bins = np.logspace(-14, 2, num=150)
LWP_points = (LWP_bins[1:] + LWP_bins[:-1]) / 2
local_time_bins = np.linspace(0, 24, 25)
local_time_points = (local_time_bins[1:] + local_time_bins[:-1]) / 2
binned_lt_albedo = np.zeros([len(LWP_bins) - 1, len(local_time_bins) - 1]) * np.nan

# allsky albedo depends on SW and LWP
for i in range(len(LWP_bins) - 1):
    LWP_mask = (atms["LWP"] > LWP_bins[i]) & (atms["LWP"] <= LWP_bins[i + 1])
    for j in range(len(local_time_bins) - 1):
        binned_lt_albedo[i, j] = float(
            (
                lower_trop_vars["albedo_allsky"]
                .where(LWP_mask)
                .sel(local_time_points=slice(local_time_bins[j], local_time_bins[j + 1]))
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

pcol = axes[0].pcolormesh(LWP_bins, local_time_bins, binned_lt_albedo.T, cmap="viridis")
axes[1].pcolormesh(LWP_bins, local_time_bins, binned_lt_albedo_interp.T, cmap="viridis")

axes[0].set_ylabel("SWin at TOA / W m$^{-2}$")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("LWP / kg m$^{-2}$")
    ax.set_xlim([1e-4, 5e1])

fig.colorbar(pcol, label="Allsky Albedo", location="bottom", ax=axes[:], shrink=0.8)

# %% average over SW  bins
SW_weights = np.zeros(len(local_time_bins) - 1)
for i in range(len(local_time_bins) - 1):
    SW_weights[i] = float(
        fluxes_allsky.isel(pressure=-1)["allsky_sw_down"]
        .sel(local_time_points=slice(local_time_bins[i], local_time_bins[i + 1]))
        .mean()
        .values
    )

mean_lt_albedo = np.zeros(len(LWP_points))
for i in range(len(LWP_bins) - 1):
    nan_mask = ~np.isnan(binned_lt_albedo[i, :])
    mean_lt_albedo[i] = np.sum(binned_lt_albedo[i, :][nan_mask] * SW_weights[nan_mask]) / np.sum(
        SW_weights[nan_mask]
    )
binned_albedos = pd.DataFrame(np.array(mean_lt_albedo).T, index=LWP_points, columns=["raw"])


#  %% calculate mean albedos


mean_cloud_albedo = float(np.sum(
    lower_trop_vars["albedo_allsky"].where(atms["mask_low_cloud"] & atms["mask_height"])
    * fluxes_allsky["allsky_sw_down"].isel(pressure=-1).where(atms["mask_low_cloud"] & atms['mask_height'])
) / np.sum(fluxes_allsky["allsky_sw_down"].isel(pressure=-1).where(atms["mask_low_cloud"] & atms['mask_height'])))

mean_clearsky_albedo = float(
    lower_trop_vars["albedo_clearsky"]
    .where(~atms["mask_low_cloud"] & atms["mask_height"])
    .mean()
    .values
)

mean_alpha_t = (lower_trop_vars["alpha_t"] * fluxes_allsky.isel(pressure=-1)["allsky_sw_down"]).sum(
    ["local_time_points", "profile"]
) / fluxes_allsky.isel(pressure=-1)["allsky_sw_down"].sum(["local_time_points", "profile"])


# %% plot albedo vs LWP
fig, ax = plt.subplots()

ax.scatter(
    atms["LWP"].where(atms["mask_height"]),
    lower_trop_vars["albedo_allsky"].where(atms["mask_height"]),
    c=fluxes_noice.isel(pressure=-1)["clearsky_sw_down"].where(atms["mask_height"]),
    s=0.1,
)

ax.axhline(mean_clearsky_albedo, color="k", linestyle="--", label="Mean clearsky")
ax.axhline(mean_cloud_albedo, color="grey", linestyle="--", label="Mean low cloud")
ax.plot(binned_albedos["raw"], color="k", linestyle="-", label="Mean")
ax.set_xscale("log")
ax.legend()
fig.tight_layout()

# %% plot albedo vs IWP
fig, ax = plt.subplots()
ax.scatter(
    atms["IWP"].where(atms["mask_height"]),
    lower_trop_vars["alpha_t"].where(atms["mask_height"]),
    c=fluxes_noice.isel(pressure=-1)["clearsky_sw_down"].where(atms["mask_height"]),
    s=0.1,
)
ax.plot(mean_alpha_t["iwp_points"], mean_alpha_t, color="red", label="Mean")
ax.set_xscale("log")

# %% plot clearsky albedo
fig, ax = plt.subplots()
aux = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/raw_data/aux.nc")

ax.scatter(
    fluxes_noice.isel(pressure=-1)["clearsky_sw_down"].where(atms["mask_height"]),
    lower_trop_vars["albedo_clearsky"].where(atms["mask_height"]),
    c=aux["ocean_fraction_surface"].where(atms["mask_height"]),
    s=0.1,
)
#  mean ocean albedo
mean_ocean_albedo = (
    lower_trop_vars["albedo_clearsky"]
    .where((aux["ocean_fraction_surface"] > 0) & atms["mask_height"] & (atms["LWP"] < 1e-4))
    .mean()
)
print(f"Ocean albedo: {mean_ocean_albedo.values.round(4)}")

# %% plot distribution of clearsky albedo
fig, axes = plt.subplots(1, 2)
bins = np.linspace(0, 1, 51)
axes[0].hist(
    lower_trop_vars["albedo_clearsky"]
    .where(atms["mask_height"] & (aux["ocean_fraction_surface"] == 0))
    .values.flatten(),
    bins=bins,
)
axes[0].set_title("Land")
axes[0].set_xlabel("Clearsky Albedo")
axes[1].hist(
    lower_trop_vars["albedo_clearsky"]
    .where(atms["mask_height"] & (aux["ocean_fraction_surface"] == 1))
    .values.flatten(),
    bins=bins,
)
axes[1].set_title("Ocean")
axes[1].set_xlabel("Clearsky Albedo")
fig.tight_layout()


# %% calculate R_t
lower_trop_vars["R_t"] = xr.where(
    atms["mask_low_cloud"],
    fluxes_noice.isel(pressure=-1)["allsky_lw_up"],
    fluxes_noice.isel(pressure=-1)["clearsky_lw_up"],
)

# %% calculate mean R_t
mean_R_l = float(
    lower_trop_vars["R_t"].where(atms["mask_height"] & atms["mask_low_cloud"]).mean().values
)
mean_R_cs = float(
    lower_trop_vars["R_t"].where(atms["mask_height"] & ~atms["mask_low_cloud"]).mean().values
)

# %% regress R_t on IWP
IWP_bins = np.logspace(-6, np.log10(30), 51)
mean_R_t = lower_trop_vars["R_t"].where(atms["mask_height"]).mean(["local_time_points", "profile"])
x = np.log10(atms["iwp_points"])
y = mean_R_t.values - np.mean(mean_R_t.values)
c_h20_coeffs = linregress(x, y)

# %% plot R_t vs IWP and regression
fig, ax = plt.subplots()
colors = ["black", "grey", "blue"]
cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
sc_rt = ax.scatter(
    atms["IWP"].where(atms["mask_height"]),
    lower_trop_vars["R_t"].where(atms["mask_height"]),
    c=xr.where(atms["mask_low_cloud"], atms["LWP"], atms["LWP"] * 0 + 1e-12).where(
        atms["mask_height"]
    ),
    cmap=cmap,
    norm=LogNorm(vmin=1e-6, vmax=1e0),
    s=0.1,
)
fit = (
    f_mean * mean_R_l
    + (1 - f_mean) * mean_R_cs
    + c_h20_coeffs.slope * np.log10(atms["iwp_points"])
    + c_h20_coeffs.intercept
)
ax.plot(atms["iwp_points"], fit, color="red", label="Fit")
ax.axhline(mean_R_cs, color="grey", linestyle="--", label="Clearsky")
ax.axhline(mean_R_l, color="navy", linestyle="--", label="Low Cloud")
ax.plot(mean_R_t["iwp_points"], mean_R_t, color="orange", label="Mean")

ax.set_ylabel(r"LT LW Emissions ($\mathrm{R_t}$) / $\mathrm{W ~ m^{-2}}$")
ax.legend()
ax.set_ylim(-350, -200)
ax.set_xlim(1e-5, 1e1)
ax.set_xscale("log")

# %% save variables
path = "/work/bm1183/m301049/iwp_framework/ngc3/"

lower_trop_vars.to_netcdf(path + "data/lower_trop_vars.nc")

mean_alpha_t.to_netcdf(path + "data/mean_alpha_t.nc")

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
