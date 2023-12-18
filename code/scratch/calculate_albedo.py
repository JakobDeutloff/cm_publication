# %% imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata
import pickle

# %% load freddis data
path = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg_noice/"
atms = xr.open_dataset(path + run + "atms_full.nc")
fluxes_3d = xr.open_dataset(path + run + "fluxes_3d_full.nc")
aux = xr.open_dataset(path + run + "aux.nc")

# %% find profiles with high clouds and no low clouds below and above 8 km
idx_height = (atms["IWC"] + atms['snow'] + atms['graupel']).argmax("pressure")
mask_graupel = atms.isel(pressure=idx_height)["pressure"] < 35000
mask_iwc = atms["IWC"].max('pressure') > 1e-12
mask_valid = mask_graupel & mask_iwc
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)

# %% calculate high cloud albedo
atms["clearsky_albedo"] = np.abs(
    fluxes_3d.isel(pressure=-1)["clearsky_sw_up"]
    / fluxes_3d.isel(pressure=-1)["clearsky_sw_down"]
)
atms["allsky_albedo"] = np.abs(
    fluxes_3d.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_3d.isel(pressure=-1)["allsky_sw_down"]
)
atms["high_cloud_albedo"] = (atms["allsky_albedo"] - atms["clearsky_albedo"]) / (
    1 - atms["clearsky_albedo"]
)

# %% calculate mean albedos by weighting with the incoming SW radiation in IWP bins
IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
SW_down_bins = np.linspace(0, 1360, 30)
binned_hc_albedo = np.zeros([len(IWP_bins) - 1, len(SW_down_bins) - 1])

for i in range(len(binned_hc_albedo) - 1):
    IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i + 1])
    for j in range(len(SW_down_bins) - 1):
        SW_mask = (fluxes_3d["allsky_sw_down"].isel(pressure=-1) > SW_down_bins[j]) & (
            fluxes_3d["allsky_sw_down"].isel(pressure=-1) < SW_down_bins[j + 1]
        )
        binned_hc_albedo[i, j] = float(
            (
                atms["high_cloud_albedo"]
                .where(IWP_mask & SW_mask & mask_hc_no_lc & mask_height)
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

# %% average over SW albedo bins
mean_hc_albedo_SW = np.zeros(len(IWP_bins) - 1)
mean_hc_albedo_SW_interp = np.zeros(len(IWP_bins) - 1)
for i in range(len(IWP_bins) - 1):
    nan_mask = ~np.isnan(binned_hc_albedo[i, :])
    mean_hc_albedo_SW[i] = np.sum(
        binned_hc_albedo[i, :][nan_mask] * SW_down_bins[:-1][nan_mask]
    ) / np.sum(SW_down_bins[:-1][nan_mask])
    mean_hc_albedo_SW_interp[i] = np.sum(
        binned_hc_albedo_interp[i, :] * SW_down_bins[:-1]
    ) / np.sum(SW_down_bins[:-1])

# %% Calculate mean albedo by calculating cumulative radiation balance in IWP and SW bins
fluxes_toa = fluxes_3d.isel(pressure=-1)

binned_clearsky_sw_up = np.zeros([len(IWP_bins) - 1, len(SW_down_bins) - 1])
binned_clearsky_sw_down = np.zeros([len(IWP_bins) - 1, len(SW_down_bins) - 1])
binned_allsky_sw_up = np.zeros([len(IWP_bins) - 1, len(SW_down_bins) - 1])
binned_allsky_sw_down = np.zeros([len(IWP_bins) - 1, len(SW_down_bins) - 1])

for i in range(len(binned_hc_albedo) - 1):
    IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i + 1])
    for j in range(len(SW_down_bins) - 1):
        SW_mask = (fluxes_toa["allsky_sw_down"] > SW_down_bins[j]) & (
            fluxes_toa["allsky_sw_down"] < SW_down_bins[j + 1]
        )
        binned_clearsky_sw_up[i, j] = float(
            (
                fluxes_toa["clearsky_sw_up"]
                .where(IWP_mask & SW_mask & mask_hc_no_lc & mask_height)
                .sel(lat=slice(-30, 30))
            )
            .mean()
            .values
        )
        binned_clearsky_sw_down[i, j] = float(
            (
                fluxes_toa["clearsky_sw_down"]
                .where(IWP_mask & SW_mask & mask_hc_no_lc & mask_height)
                .sel(lat=slice(-30, 30))
            )
            .mean()
            .values
        )
        binned_allsky_sw_up[i, j] = float(
            (
                fluxes_toa["allsky_sw_up"]
                .where(IWP_mask & SW_mask & mask_hc_no_lc & mask_height)
                .sel(lat=slice(-30, 30))
            )
            .mean()
            .values
        )
        binned_allsky_sw_down[i, j] = float(
            (
                fluxes_toa["allsky_sw_down"]
                .where(IWP_mask & SW_mask & mask_hc_no_lc & mask_height)
                .sel(lat=slice(-30, 30))
            )
            .mean()
            .values
        )


# %% interpolate SW fluxes
def interpolate(data):
    non_nan_indices = np.array(np.where(~np.isnan(data)))
    non_nan_values = data[~np.isnan(data)]
    nan_indices = np.array(np.where(np.isnan(data)))

    interpolated_values = griddata(
        non_nan_indices.T, non_nan_values, nan_indices.T, method="linear"
    )

    copy = data.copy()
    copy[np.isnan(data)] = interpolated_values
    return copy


SW_fluxes_original = {
    "clearsky_sw_up": binned_clearsky_sw_up,
    "clearsky_sw_down": binned_clearsky_sw_down,
    "allsky_sw_up": binned_allsky_sw_up,
    "allsky_sw_down": binned_allsky_sw_down,
}
SW_fluxes_interpolated = {}

for key in SW_fluxes_original.keys():
    SW_fluxes_interpolated[key] = interpolate(SW_fluxes_original[key])


# %% calculate albedo from binned fluxes
def hc_albedo(fluxes):
    clearsky_albedo = np.nansum(fluxes["clearsky_sw_up"], axis=1) / np.nansum(
        fluxes["clearsky_sw_down"], axis=1
    )
    allsky_albedo = np.nansum(fluxes["allsky_sw_up"], axis=1) / np.nansum(
        fluxes["allsky_sw_down"], axis=1
    )
    binned_hc_albedo_radsum = (allsky_albedo - clearsky_albedo) / (1 - clearsky_albedo)
    return binned_hc_albedo_radsum


hc_albedo_radsum_original = hc_albedo(SW_fluxes_original)
hc_albedo_radsum_interpolated = hc_albedo(SW_fluxes_interpolated)


# %% plot albedo in SW bins
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

pcol = axes[0].pcolormesh(
    IWP_bins, SW_down_bins, binned_hc_albedo.T * -1, cmap="viridis"
)
axes[1].pcolormesh(
    IWP_bins, SW_down_bins, binned_hc_albedo_interp.T * -1, cmap="viridis"
)

axes[0].set_ylabel("SWin at TOA / W m$^{-2}$")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("IWP / kg m$^{-2}$")
    ax.set_xlim([1e-4, 5e1])

fig.colorbar(pcol, label="High Cloud Albedo", location="bottom", ax=axes[:], shrink=0.8)

# %% plot binned SW fluxes
fig, axes = plt.subplots(2, 4, figsize=(15, 10), sharey=True, sharex=True)

# Original data
axes[0, 0].pcolormesh(
    IWP_bins, SW_down_bins, SW_fluxes_original["clearsky_sw_up"].T, cmap="viridis"
)
axes[0, 0].set_xscale("log")
axes[0, 0].set_title("Clearsky SW up")

axes[0, 1].pcolormesh(
    IWP_bins, SW_down_bins, SW_fluxes_original["clearsky_sw_down"].T, cmap="viridis"
)
axes[0, 1].set_xscale("log")
axes[0, 1].set_title("Clearsky SW down")

axes[1, 0].pcolormesh(
    IWP_bins, SW_down_bins, SW_fluxes_original["allsky_sw_up"].T, cmap="viridis"
)
axes[1, 0].set_xscale("log")
axes[1, 0].set_title("Allsky SW up")

axes[1, 1].pcolormesh(
    IWP_bins, SW_down_bins, SW_fluxes_original["allsky_sw_down"].T, cmap="viridis"
)
axes[1, 1].set_xscale("log")
axes[1, 1].set_title("Allsky SW down")

# Interpolated data
axes[0, 2].pcolormesh(
    IWP_bins, SW_down_bins, SW_fluxes_interpolated["clearsky_sw_up"].T, cmap="viridis"
)
axes[0, 2].set_xscale("log")
axes[0, 2].set_title("Clearsky SW up")

axes[0, 3].pcolormesh(
    IWP_bins, SW_down_bins, SW_fluxes_interpolated["clearsky_sw_down"].T, cmap="viridis"
)
axes[0, 3].set_xscale("log")
axes[0, 3].set_title("Clearsky SW down")

axes[1, 2].pcolormesh(
    IWP_bins, SW_down_bins, SW_fluxes_interpolated["allsky_sw_up"].T, cmap="viridis"
)
axes[1, 2].set_xscale("log")
axes[1, 2].set_title("Allsky SW up")

axes[1, 3].pcolormesh(
    IWP_bins, SW_down_bins, SW_fluxes_interpolated["allsky_sw_down"].T, cmap="viridis"
)
axes[1, 3].set_xscale("log")
axes[1, 3].set_title("Allsky SW down")

fig.tight_layout()

# %% plot all albedos
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(IWP_points, mean_hc_albedo_SW, label="Weighted albedo", color="k")
ax.plot(
    IWP_points,
    mean_hc_albedo_SW_interp,
    label="Weightd albedo interpolated",
    color="red",
)
ax.plot(
    IWP_points,
    hc_albedo_radsum_original * -1,
    label="Cumulative Radiation",
    linestyle="--",
    color="k",
)
ax.plot(
    IWP_points,
    hc_albedo_radsum_interpolated * -1,
    label="Cumulative Radiation Interpolated",
    linestyle="--",
    color="red",
)

ax.set_xlim([1e-4, 1])
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("High Cloud Albedo")
ax.legend()

# %% fit polynom to weighted and interpolated albedo
iwp_mask = IWP_points <=1
p = np.polyfit(np.log10(IWP_points[iwp_mask]), mean_hc_albedo_SW_interp[iwp_mask], 5)
poly = np.poly1d(p)
fitted_curve = poly(np.log10(IWP_points[iwp_mask]))


# %% plot weighted albedo in scatterplot with IWP
fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"].where(mask_hc_no_lc & mask_height).sel(lat=slice(-30, 30)),
    atms["high_cloud_albedo"].where(mask_hc_no_lc & mask_height).sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d.isel(pressure=-1)["allsky_sw_down"]
    .where(mask_hc_no_lc)
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)

ax.plot(IWP_points[iwp_mask], mean_hc_albedo_SW_interp[iwp_mask], label="Mean Albedo", color="k")
ax.plot(IWP_points[iwp_mask], fitted_curve, label="Fitted Polynomial", color="red", linestyle='--')

cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("High Cloud Albedo")
ax.set_xscale("log")
ax.set_xlim([1e-5, 1e1])
ax.legend()
fig.tight_layout()

# %% save coefficients as pkl file
with open("data/fitted_albedo.pkl", "wb") as f:
    pickle.dump(p, f)

# %%
