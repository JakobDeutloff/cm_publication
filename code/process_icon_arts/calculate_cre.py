# %% import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle

# %% load freddis data
path_freddi = "/work/bm1183/m301049/freddi_runs/"
atms = xr.open_dataset(path_freddi + "atms_full.nc")
fluxes_3d = xr.open_dataset(path_freddi + "fluxes_3d_full.nc")
fluxes_2d = xr.open_dataset(path_freddi + "fluxes_2d.nc")
aux = xr.open_dataset(path_freddi + "aux.nc")

# %% find profiles with high clouds and no low clouds below and above 8 km
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)
mask_height = ~atms["h_cloud_top_pressure"].isnull()

# %% find longitude of the sun
fluxes_toa = fluxes_3d.isel(pressure=-1)
lon_sun = 0

# %% bin CRE by IWP and SW rad down at TOA
IWP_bins = np.logspace(-5, 2, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
lon_bins = np.linspace(-180, 180, num=36)
dummy = np.zeros([len(IWP_bins) - 1, len(lon_bins) - 1])
cre = {'all': dummy.copy(), 'sw': dummy.copy(), 'lw': dummy.copy()}

for i in range(len(IWP_bins) - 1):
    IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i + 1])
    for j in range(len(lon_bins) - 1):
        lon_mask = (fluxes_toa.lon > lon_bins[j]) & (fluxes_toa.lon <= lon_bins[j + 1])
        cre['all'][i, j] = float(
            (
                atms["cloud_rad_effect"]
                .where(IWP_mask & lon_mask & mask_hc_no_lc & mask_height)
                .sel(lat=slice(-30, 30))
            )
            .mean()
            .values
        )
        cre['sw'][i, j] = float(
            (
                atms["sw_cloud_rad_effect"]
                .where(IWP_mask & lon_mask & mask_hc_no_lc & mask_height)
                .sel(lat=slice(-30, 30))
            )
            .mean()
            .values
        )
        cre['lw'][i, j] = float(
            (
                atms["lw_cloud_rad_effect"]
                .where(IWP_mask & lon_mask & mask_hc_no_lc & mask_height)
                .sel(lat=slice(-30, 30))
            )
            .mean()
            .values
        )
# %% Interpolate CRE bins
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

interp_cre = {'all': cre['all'].copy(), 'sw': cre["sw"].copy(), 'lw': cre["lw"].copy()}
for key in cre.keys():
    interp_cre[key] = interpolate(cre[key])


# %% plot binned total CRE and interpolated CRE
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
pcol = axes[0].pcolormesh(IWP_bins, lon_bins, cre['all'].T, cmap='seismic', vmin=-600, vmax=600)
axes[0].set_ylabel("Longitude [deg]")
axes[0].set_title("CRE binned by IWP and longitude")
axes[1].pcolor(IWP_bins, lon_bins, interp_cre['all'].T, cmap='seismic', vmin=-600, vmax=600)
axes[1].set_title("Interpolated CRE")

for ax in axes:
    ax.set_xscale("log")    
    ax.set_xlabel("IWP [kg m$^{-2}$]")

fig.colorbar(pcol, label="High Cloud Radiative Effect", location="bottom", ax=axes[:], shrink=0.7, extend='min', pad=0.2)
fig.savefig("plots/CRE_binned_by_IWP_and_lon.png", dpi=300, bbox_inches="tight")

# %% Average over longitude bins
mean_cre = {'all': np.nanmean(cre['all'], axis=1), 'sw': np.nanmean(cre['sw'], axis=1), 'lw': np.nanmean(cre['lw'], axis=1)}
mean_cre_interp = {'all': np.nanmean(interp_cre['all'], axis=1), 'sw': np.nanmean(interp_cre['sw'], axis=1), 'lw': np.nanmean(interp_cre['lw'], axis=1)}

# %% plot mean CRE vs IWP
fig, ax = plt.subplots()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
end = -13
ax.plot(IWP_points[:end], mean_cre["all"][:end], label="CRE ", color='k', alpha=0.3)
ax.plot(IWP_points[:end], mean_cre_interp['all'][:end], label="CRE interpolated", color='k')
ax.plot(IWP_points[:end], mean_cre["sw"][:end], label="SW CRE", color='blue', alpha=0.3)
ax.plot(IWP_points[:end], mean_cre_interp['sw'][:end], label="SW CRE interpolated", color='blue')
ax.plot(IWP_points[:end], mean_cre["lw"][:end], label="LW CRE", color='r', alpha=0.3)
ax.plot(IWP_points[:end], mean_cre_interp['lw'][:end], label="LW CRE interpolated", color='r')
ax.set_xscale("log")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Cloud Radiative Effect / W m$^{-2}$")
ax.legend()
ax.axhline(0, color="k", linestyle="--")
fig.tight_layout()
fig.savefig("plots/mean_CRE_vs_IWP.png", dpi=300)

# %% plot mean CRE in Scatterplot with IWP

fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"].where(mask_hc_no_lc & mask_height).sel(lat=slice(-30, 30)),
    (atms["cloud_rad_effect"]).where(mask_hc_no_lc & mask_height).sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d.isel(pressure=-1)["allsky_sw_down"]
    .where(mask_hc_no_lc)
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)

ax.plot(IWP_points[:end], mean_cre_interp["all"][:end], label="Average CRE", color='r')

cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Cloud Radiative Effect / W m$^{-2}$")
ax.set_xscale("log")
ax.set_xlim([1e-4, 1e1])
ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
fig.tight_layout()

# %% save mean CRE
with open("data/hc_cre.pkl", "wb") as f:
    pickle.dump([IWP_points[:end], mean_cre_interp["all"][:end]], f)

with open("data/hc_cre_sw.pkl", "wb") as f:
    pickle.dump([IWP_points[:end], mean_cre_interp['sw'][:end]], f)

with open("data/hc_cre_lw.pkl", "wb") as f:
    pickle.dump([IWP_points[:end], mean_cre_interp['lw'][:end]], f)

# %%
