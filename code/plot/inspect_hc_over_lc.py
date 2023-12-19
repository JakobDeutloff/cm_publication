# %% import modules
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from src.plot_functions import plot_profiles_noice

# %% load data
path = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg_noice/"
fluxes_3d_noice = xr.open_dataset(path + run + "fluxes_3d_full.nc")
atms = xr.open_dataset(path + run + "atms_full.nc")
run = "fullrange_flux_mid1deg/"
fluxes_3d = xr.open_dataset(path + run + "fluxes_3d_full.nc")
cre_binned = xr.open_dataset("data/cre_binned.nc")
cre_interpolated = xr.open_dataset("data/cre_interpolated.nc")
lw_vars = xr.open_dataset("data/lw_vars.nc")

# %% fractions of gridcells I use
mask_height = lw_vars["mask_height"]
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)

n_gridcells_noice = mask_height.sel(lat=slice(-30, 30)).mean()
n_gridcells_no_lc = mask_hc_no_lc.sel(lat=slice(-30, 30)).mean()
n_gridcells_noice_no_lc = (mask_hc_no_lc & mask_height).sel(lat=slice(-30, 30)).mean()

print(f"Fraction of gridcells with proper high clouds: {n_gridcells_noice.values:.2f}")
print(
    f"Fraction of gridcells with any high clouds and no low clouds: {n_gridcells_no_lc.values:.2f}"
)
print(
    f"Fraction of gridcells with proper high clouds and no low clouds: {n_gridcells_noice_no_lc.values:.2f}"
)

# %% find bins with positive CRE at high SW in
strange_mask = (
    (cre_binned["all_net"] > 0)
    & (cre_binned["all_net"].IWP > 5e-2)
    & (-70 < cre_binned["all_net"].lon)
    & (cre_binned["all_net"].lon < 70)
)

IWP, lon = np.meshgrid(cre_binned["all_net"].IWP, cre_binned["all_net"].lon)
x_coords = IWP[strange_mask.T]
y_coords = lon[strange_mask.T]

# %% plot Net CRE and selected bins
fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey="row")

cre_binned["all_net"].plot.pcolormesh(
    ax=axes[0],
    x="IWP",
    vmin=-300,
    vmax=300,
    cmap="RdBu_r",
    extend="both",
    add_colorbar=False,
)
axes[0].scatter(x_coords, y_coords, color="k", marker="x", alpha=0.5)
axes[0].set_title("All clouds")
axes[0].set_ylabel("Longitude")

cmap = cre_binned["ice_only_net"].plot.pcolormesh(
    ax=axes[1],
    x="IWP",
    vmin=-300,
    vmax=300,
    cmap="RdBu_r",
    extend="both",
    add_colorbar=False,
)
axes[1].scatter(x_coords, y_coords, color="k", marker="x", alpha=0.5)
axes[1].set_title("High clouds no low clouds")
axes[1].set_ylabel("")

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1e2)
    ax.set_xlabel("IWP / kg m$^{-2}$")

fig.tight_layout()
fig.colorbar(
    cmap,
    ax=axes,
    label=" Net HCRE / W m$^{-2}$",
    orientation="horizontal",
    shrink=0.6,
    pad=0.2,
    extend="both",
)
fig.savefig("plots/cre_net_binned.png", dpi=300)

# %% plot SW cre and selected bins
fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey="row")

cre_binned["all_sw"].plot.pcolormesh(
    ax=axes[0],
    x="IWP",
    cmap="RdBu_r",
    vmin=-200,
    vmax=200,
    extend="min",
    add_colorbar=False,
)
axes[0].scatter(x_coords, y_coords, color="k", marker="x", alpha=0.5)
axes[0].set_title("All clouds")
axes[0].set_ylabel("Longitude")

cmap = cre_binned["ice_only_sw"].plot.pcolormesh(
    ax=axes[1],
    x="IWP",
    cmap="RdBu_r",
    vmin=-200,
    vmax=200,
    extend="min",
    add_colorbar=False,
)
axes[1].scatter(x_coords, y_coords, color="k", marker="x", alpha=0.5)
axes[1].set_title("High clouds no low clouds")
axes[1].set_ylabel("")

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1e2)
    ax.set_xlabel("IWP / kg m$^{-2}$")

fig.tight_layout()
fig.colorbar(
    cmap,
    ax=axes,
    label="SW HCRE / W m$^{-2}$",
    orientation="horizontal",
    shrink=0.6,
    pad=0.2,
    extend="min",
)

# %% plot LW cre and selected bins
fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey="row")

cre_binned["all_lw"].plot.pcolormesh(
    ax=axes[0], x="IWP", cmap="Reds", vmin=0, vmax=200, extend="max", add_colorbar=False
)
axes[0].scatter(x_coords, y_coords, color="k", marker="x", alpha=0.5)
axes[0].set_title("All clouds")
axes[0].set_ylabel("Longitude")

cmap = cre_binned["ice_only_lw"].plot.pcolormesh(
    ax=axes[1], x="IWP", cmap="Reds", vmin=0, vmax=200, extend="max", add_colorbar=False
)
axes[1].scatter(x_coords, y_coords, color="k", marker="x", alpha=0.5)
axes[1].set_title("High clouds no low clouds")
axes[1].set_ylabel("")

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1e2)
    ax.set_xlabel("IWP / kg m$^{-2}$")

fig.tight_layout()
fig.colorbar(
    cmap,
    ax=axes,
    label=" LW HCRE / W m$^{-2}$",
    orientation="horizontal",
    shrink=0.6,
    pad=0.2,
    extend="max",
)

# %% look at profiles in positive bins 

# find lon and IWP coordinate of maximum sw cre
max_idx = cre_binned["all_sw"].where(strange_mask).argmax(dim=["lon", "IWP"])
IWP_max = cre_binned["all_sw"].IWP.isel(IWP=max_idx['IWP'].values) # 0.339 
lon_max = cre_binned["all_sw"].lon.isel(lon=max_idx['lon'].values) # -20

# get respective bins
dist = cre_binned.IWP_bins - IWP_max
idx_low = dist.where(dist < 0).argmax(dim="IWP_bins")
idx_high = dist.where(dist > 0).argmin(dim="IWP_bins")
IWP_bin = [cre_binned.IWP_bins.isel(IWP_bins=idx_low), cre_binned.IWP_bins.isel(IWP_bins=idx_high)]

dist = cre_binned.lon_bins - lon_max
idx_low = dist.where(dist < 0).argmax(dim="lon_bins")
idx_high = dist.where(dist > 0).argmin(dim="lon_bins")
lon_bin = [cre_binned.lon_bins.isel(lon_bins=idx_low), cre_binned.lon_bins.isel(lon_bins=idx_high)]

# get lat and lon coordinates of profiles in the bins
IWP_mask = (atms.IWP > IWP_bin[0]) & (atms.IWP < IWP_bin[1])
lon_mask = (atms.lon > lon_bin[0]) & (atms.lon < lon_bin[1])
lons_3d, lats_3d = np.meshgrid(atms.lon, atms.lat)
lons = lons_3d[mask_height & IWP_mask & lon_mask]
lats = lats_3d[mask_height & IWP_mask & lon_mask]
lats_tropic = (lats < 30) & (lats > -30)
lats = lats[lats_tropic]
lons = lons[lats_tropic]

# plot profile 
fig, axes = plot_profiles_noice(lat=lats[0], lon=lons[0], atms=atms, fluxes_3d=fluxes_3d, fluxes_3d_noice=fluxes_3d_noice)
fig.savefig(f'plots/positive_sw_cre/profile_{IWP_max}_{lon_max}.png', dpi=300)

# %% plot low cloud fraction as a function of IWP
low_clouds = (atms["LWP"] > 1e-6).sel(lat=slice(-30, 30))
IWP_bins = np.logspace(-5, 2, 100)
low_clouds_binned = low_clouds.groupby_bins(
    atms["IWP"].sel(lat=slice(-30, 30)), IWP_bins
).mean()
LWP_binned = atms["LWP"].groupby_bins(atms["IWP"], IWP_bins).mean()


fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

low_clouds_binned.plot(ax=axes[0], color="r")
axes[0].set_xscale("log")
axes[0].set_xlabel("IWP / kg m$^{-2}$")
axes[0].set_ylabel("Low cloud fraction (LWP > 1e-6 kg m$^{-2}$)")

axes[1].scatter(
    atms["IWP"].sel(lat=slice(-30, 30)),
    atms["LWP"].sel(lat=slice(-30, 30)),
    color="k",
    marker="o",
    s=0.2,
)
LWP_binned.plot(ax=axes[1], color="r")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("IWP / kg m$^{-2}$")
axes[1].set_ylabel("LWP / kg m$^{-2}$")
axes[1].set_ylim(1e-6, 1e2)
axes[1].set_xlim(1e-6, 1e2)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


fig.tight_layout()
fig.savefig("plots/low_cloud_fraction.png", dpi=300)

# %% plot low cloud albedo as function of LWP

albedo_allsky = np.abs(
    fluxes_3d_noice.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_3d_noice.isel(pressure=-1)["allsky_sw_down"]
)
albedo_clearsky = np.abs(
    fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_up"]
    / fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_down"]
)
albedo_lc = (albedo_allsky - albedo_clearsky) / (albedo_clearsky * (albedo_allsky - 2) + 1)

cell_height = atms["geometric height"].diff("pressure") 
LWC_int = (atms["LWC"] * cell_height).sum("pressure")

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
c = ax.scatter(
    atms["LWP"].sel(lat=slice(-30, 30))
    .sel(lat=slice(-30, 30)),
    albedo_lc
    .sel(lat=slice(-30, 30)),
    marker="o",
    s=0.2,
    c=fluxes_3d_noice.isel(pressure=-1)["allsky_sw_down"]
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)
fig.colorbar(c, ax=ax, label="SW down TOA / W m$^{-2}$")
ax.set_xlabel("LWP/ kg m$^{-2}$")
ax.set_ylabel("Low cloud albedo")

ax.set_xscale("log")
ax.set_xlim(1e-6, 1e2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig("plots/low_cloud_albedo_lwp.png", dpi=300)

# %% plot troposphere albedo vs IWP 
albedo_t = np.abs(
    fluxes_3d_noice.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_3d_noice.isel(pressure=-1)["allsky_sw_down"]
)

IWP_bins = np.logspace(-6, 2, 100)
albedo_t_binned = albedo_t.sel(lat=slice(-30, 30)).groupby_bins(
    atms["IWP"].sel(lat=slice(-30, 30)), IWP_bins
).mean()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
c = ax.scatter(
    atms["IWP"]
    .sel(lat=slice(-30, 30)),
    albedo_t.sel(lat=slice(-30, 30)),
    marker="o",
    s=0.2,
    c=atms["LWP"].sel(lat=slice(-30, 30)),
    cmap="viridis",
    norm=LogNorm(),
)

albedo_t_binned.plot(ax=ax, color="r")
fig.colorbar(c, ax=ax, label="LWP / kg m$^{-2}$")
ax.set_xscale("log")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel(r"$\alpha_t$")
ax.set_title('')
fig.tight_layout()
fig.savefig("plots/albedo_troposphere.png", dpi=300)

#%% plot LW up as a function of IWP
R_t_binned = fluxes_3d_noice.isel(pressure=-1)["allsky_lw_up"].sel(
    lat=slice(-30, 30)
).groupby_bins(atms["IWP"].sel(lat=slice(-30, 30)), IWP_bins).mean()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
c = ax.scatter(
    atms["IWP"]
    .sel(lat=slice(-30, 30)),
    fluxes_3d_noice.isel(pressure=-1)["allsky_lw_up"].sel(lat=slice(-30, 30)),
    marker="o",
    s=0.2,
    c=atms["LWP"].sel(lat=slice(-30, 30)),
    cmap="viridis",
    norm=LogNorm(),
)

R_t_binned.plot(ax=ax, color="r")
fig.colorbar(c, ax=ax, label="LWP / kg m$^{-2}$")
ax.set_xscale("log")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel(r"$R_t$ / W m$^{-2}$")
ax.set_title('')
fig.tight_layout()
fig.savefig("plots/R_troposphere.png", dpi=300)


