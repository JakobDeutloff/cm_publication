# %% import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# %% load freddis data
path_freddi = "/work/bm1183/m301049/freddi_runs/"
atms = xr.open_dataset(path_freddi + "atms_full.nc")
fluxes_3d = xr.open_dataset(path_freddi + "fluxes_3d_full.nc")
fluxes_2d = xr.open_dataset(path_freddi + "fluxes_2d.nc")
aux = xr.open_dataset(path_freddi + "aux.nc")

# %% find profiles with high clouds and no low clouds below
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)

# %% find longitude of the sun
fluxes_toa = fluxes_3d.isel(pressure=-1)
lon_sun = 0

# %% bin CRE by IWP and SW rad down at TOA
IWP_bins = np.logspace(-5, 2, num=50)
lon_bins = np.linspace(-180, 180, num=36)
binned_hc_CRE = np.zeros([len(IWP_bins) - 1, len(lon_bins) - 1])


for i in range(len(IWP_bins) - 1):
    IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i + 1])
    for j in range(len(lon_bins) - 1):
        lon_mask = (fluxes_toa.lon > lon_bins[j]) & (fluxes_toa.lon <= lon_bins[j + 1])
        binned_hc_CRE[i, j] = float(
            (
                atms["cloud_rad_effect"]
                .where(IWP_mask & lon_mask & mask_hc_no_lc)
                .sel(lat=slice(-30, 30))
            )
            .mean()
            .values
        )
# %% Interpolate CRE bins

non_nan_indices = np.array(np.where(~np.isnan(binned_hc_CRE)))
non_nan_values = binned_hc_CRE[~np.isnan(binned_hc_CRE)]
nan_indices = np.array(np.where(np.isnan(binned_hc_CRE)))

interpolated_values = griddata(
    non_nan_indices.T, non_nan_values, nan_indices.T, method="linear"
)
binned_hc_CRE_interp = binned_hc_CRE.copy()
binned_hc_CRE_interp[np.isnan(binned_hc_CRE)] = interpolated_values



# %% plot binned CRE and interpolated CRE
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
pcol = axes[0].pcolormesh(IWP_bins, lon_bins, binned_hc_CRE.T, cmap='seismic', vmin=-600, vmax=600)
axes[0].set_ylabel("Longitude [deg]")
axes[0].set_title("CRE binned by IWP and longitude")
axes[1].pcolor(IWP_bins, lon_bins, binned_hc_CRE_interp.T, cmap='seismic', vmin=-600, vmax=600)
axes[1].set_title("Interpolated CRE")

for ax in axes:
    ax.set_xscale("log")    
    ax.set_xlabel("IWP [kg m$^{-2}$]")

fig.colorbar(pcol, label="High Cloud Radiative Effect", location="bottom", ax=axes[:], shrink=0.7, extend='min', pad=0.2)
fig.savefig("plots/CRE_binned_by_IWP_and_lon.png", dpi=300, bbox_inches="tight")

# %% Average over longitude bins
mean_hc_CRE = np.nanmean(binned_hc_CRE, axis=1)
mean_hc_CRE_interp = binned_hc_CRE_interp.mean(axis=1)

# %% simply average over every IWP bin
mean_hc_CRE_iwp_bins_only = np.zeros(len(IWP_bins) - 1)

for i in range(len(IWP_bins) - 1):
    IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i + 1])
    mean_hc_CRE_iwp_bins_only[i] = float(
        (
            atms["cloud_rad_effect"]
            .where(IWP_mask & mask_hc_no_lc)
            .sel(lat=slice(-30, 30))
        )
        .mean()
        .values
    )

# %% plot mean CRE vs IWP
fig, ax = plt.subplots()
ax.plot(IWP_bins[:-1], mean_hc_CRE, label="Lon bins")
ax.plot(IWP_bins[:-1], mean_hc_CRE_interp, label="Lon bins interpolated")
ax.plot(IWP_bins[:-1], mean_hc_CRE_iwp_bins_only, label="IWP bins only")
ax.set_xscale("log")
ax.legend()
ax.axhline(0, color="k", linestyle="--")

# %% plot mean CRE in Scatterplot with IWP

fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"].where(mask_hc_no_lc).sel(lat=slice(-30, 30)),
    (atms["cloud_rad_effect"]).where(mask_hc_no_lc).sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d.isel(pressure=-1)["allsky_sw_down"]
    .where(mask_hc_no_lc)
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)

ax.plot(IWP_bins[:-1], mean_hc_CRE_interp, label="Average CRE", color='r')

cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Cloud Radiative Effect / W m$^{-2}$")
ax.set_xscale("log")
ax.set_xlim([1e-4, 1e1])
ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
fig.tight_layout()

# %%
