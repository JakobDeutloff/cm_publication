# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from src.plot_functions import plot_profiles
from cmocean import cm

# %% load data from freddis runs
path_freddi = "/work/bm1183/m301049/freddi_runs/"
atms = xr.open_dataset(path_freddi + "atms_full.nc")
fluxes_3d = xr.open_dataset(path_freddi + "fluxes_3d_full.nc")
fluxes_2d = xr.open_dataset(path_freddi + "fluxes_2d.nc")
aux = xr.open_dataset(path_freddi + "aux.nc")

# %% find high clouds with no low clouds below
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)
lon_3d, lat_3d = np.meshgrid(atms["lon"], atms["lat"])
lons = lon_3d[mask_hc_no_lc]
lats = lat_3d[mask_hc_no_lc]

# %% plot location of profiles
fig, ax = plt.subplots(figsize=(12, 4), subplot_kw={"projection": ccrs.PlateCarree()})
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

cont = (
    (fluxes_3d["allsky_sw_down"])
    .isel(pressure=-1)
    .sel(lat=slice(-30, 30))
    .plot.contourf(
        ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", add_colorbar=False
    )
)
ax.scatter(lons, lats, s=0.1, color="red", marker="o")

ax.set_ylim(-30, 30)
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
ax.set_title("Profiles with IWP > 1e-6 kg m$^{-2}$ and LWP < 1e-10 kg m$^{-2}$")
fig.colorbar(
    cont, label="SWin at TOA / W m$^{-2}$", orientation="horizontal", shrink=0.5
)
fig.tight_layout()
fig.savefig("plots/profile_locations.png", dpi=300)

# %% select and plot profile with a certain IWP 
IWP = 1e-3  # kg m^-2
diff = abs(atms['IWP'].where(mask_hc_no_lc) - IWP)
min_diff_indices = diff.argmin(dim=["lat", "lon"])
lat = diff["lat"].isel(lat=min_diff_indices["lat"]).values
lon = diff["lon"].isel(lon=min_diff_indices["lon"]).values
#  plot profiles at point
plot_profiles(lat, lon, atms, fluxes_3d)

# %% calculate plot quantities
iwp_cutoff = 1e-3

iwp = atms["IWP"].sel(lat=slice(-30, 30))
iwp = iwp.where(iwp > iwp_cutoff)

pres = atms["h_cloud_top_pressure"].sel(lat=slice(-30, 30))
pres = pres.where(atms["IWP"].sel(lat=slice(-30, 30)) > iwp_cutoff) / 100

temp = atms["h_cloud_temperature"].sel(lat=slice(-30, 30))
temp = temp.where(atms["IWP"].sel(lat=slice(-30, 30)) > iwp_cutoff)

cloud_rad_effect = atms["cloud_rad_effect"].sel(lat=slice(-30, 30))
cloud_rad_effect = cloud_rad_effect.where(
    atms["IWP"].sel(lat=slice(-30, 30)) > iwp_cutoff
)

# %% plot maps of IWP and T_h
fig, axes = plt.subplots(
    3, 1, figsize=(14, 8), subplot_kw={"projection": ccrs.PlateCarree()}
)

iwp.plot(
    ax=axes[0],
    transform=ccrs.PlateCarree(),
    levels=np.linspace(iwp_cutoff, 7, 20),
    extend="max",
    cmap="cool",
)

temp.plot(
    ax=axes[1],
    transform=ccrs.PlateCarree(),
    levels=np.linspace(200, 260, 20),
    extend="both",
    cmap="RdBu_r",
)

pres.plot(
    ax=axes[2],
    transform=ccrs.PlateCarree(),
    levels=np.linspace(100, 430, 20),
    extend="both",
    cmap=cm.deep,
)

for ax in axes:
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

fig.tight_layout()


# %% simple scatterplot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(pres, temp, s=0.3, color="black")
axes[0, 0].set_ylabel("T_h / K")

axes[0, 1].scatter(iwp, temp, s=0.3, color="black")
axes[0, 1].set_xlabel("FWP / kg m$^{-2}$")
axes[0, 1].set_xscale("log")

axes[1, 0].scatter(pres, iwp, s=0.3, color="black")
axes[1, 0].set_xlabel("p_h / hPa")
axes[1, 0].set_ylabel("FWP / kg m$^{-2}$")
axes[1, 0].set_yscale("log")

axes[1, 1].scatter(iwp, cloud_rad_effect * -1, s=0.3, color="black")
axes[1, 1].set_xlabel("FWP / kg m$^{-2}$")
axes[1, 1].set_xscale("log")
axes[1, 1].set_ylabel("Cloud Radiative Effect / W m$^{-2}$")

# %% plot mean profiles of frozen hydrometeors
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey="row")

# IWC
iwc_mean = atms["IWC"].sel(lat=slice(-30, 30)).mean(["lat", "lon"])
iwc_std = atms["IWC"].sel(lat=slice(-30, 30)).std(["lat", "lon"])
axes[0].plot(iwc_mean, iwc_mean.pressure / 100, color="black")
axes[0].fill_betweenx(
    iwc_mean.pressure / 100,
    iwc_mean - iwc_std,
    iwc_mean + iwc_std,
    color="black",
    alpha=0.5,
)
axes[0].set_ylabel("Pressure / hPa")
axes[0].set_xlabel("IWC / kg m$^{-3}$")

# Graupel
graupel_mean = atms["graupel"].sel(lat=slice(-30, 30)).mean(["lat", "lon"])
graupel_std = atms["graupel"].sel(lat=slice(-30, 30)).std(["lat", "lon"])
axes[1].plot(graupel_mean, graupel_mean.pressure / 100, color="black")
axes[1].fill_betweenx(
    graupel_mean.pressure / 100,
    graupel_mean - graupel_std,
    graupel_mean + graupel_std,
    color="black",
    alpha=0.5,
)
axes[1].set_xlabel("Graupel / kg m$^{-3}$")

# Snow
snow_mean = atms["snow"].sel(lat=slice(-30, 30)).mean(["lat", "lon"])
snow_std = atms["snow"].sel(lat=slice(-30, 30)).std(["lat", "lon"])
axes[2].plot(snow_mean, snow_mean.pressure / 100, color="black")
axes[2].fill_betweenx(
    snow_mean.pressure / 100,
    snow_mean - snow_std,
    snow_mean + snow_std,
    color="black",
    alpha=0.5,
)
axes[2].set_xlabel("Snow / kg m$^{-3}$")

for ax in axes:
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# %% scatterplot of cloud radiative effect vx IWP at locations with no low cloud below
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

cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Cloud Radiative Effect / W m$^{-2}$")
ax.set_xscale("log")
ax.set_xlim([1e-4, 1e1])
ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
fig.tight_layout()
fig.savefig("plots/cloud_rad_effect_vs_iwp.png", dpi=300)

# %% plot high cloud albedo against IWP

fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"].where(mask_hc_no_lc).sel(lat=slice(-30, 30)),
    atms["high_cloud_albedo"].where(mask_hc_no_lc).sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d["allsky_sw_down"]
    .isel(pressure=-1)
    .where(mask_hc_no_lc)
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)
cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("High Cloud Albedo")
ax.set_xscale("log")
fig.savefig("plots/high_cloud_albedo_vs_iwp.png", dpi=300)


# %% plot emissivity against IWP

fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"]
    .where(mask_hc_no_lc)
    .where(atms["h_cloud_top_pressure"])
    .sel(lat=slice(-30, 30)),
    atms["high_cloud_emissivity"]
    .where(mask_hc_no_lc)
    .where(atms["h_cloud_top_pressure"])
    .sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d["allsky_sw_down"]
    .isel(pressure=-1)
    .where(mask_hc_no_lc)
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)
cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("High Cloud Emissivity")
ax.set_xscale("log")
fig.savefig("plots/high_cloud_emissivity_vs_iwp.png", dpi=300)

# %% Scatterplot high cloud temperature vs surface temperature

fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    aux["surface temperature"].where(mask_hc_no_lc).sel(lat=slice(-30, 30)),
    atms["h_cloud_temperature"].where(mask_hc_no_lc).sel(lat=slice(-30, 30)),
    s=0.5,
    c=atms["h_cloud_top_pressure"].where(mask_hc_no_lc).sel(lat=slice(-30, 30)) / 100,
    cmap="viridis",
)

cb = fig.colorbar(sc)
cb.set_label("p_h / hPa")
ax.set_xlabel("Surface Temperature / K")
ax.set_ylabel("High Cloud Temperature / K")

# %% plot corrected high cloud albedo against IWP
albedo_corr = atms["high_cloud_albedo"] * (
    fluxes_3d["allsky_sw_down"].isel(pressure=-1)
    / fluxes_3d["clearsky_sw_down"].isel(pressure=-1).max()
) 

fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"].where(mask_hc_no_lc).sel(lat=slice(-30, 30)),
    albedo_corr.where(mask_hc_no_lc).sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d["allsky_sw_down"]
    .isel(pressure=-1)
    .where(mask_hc_no_lc)
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)
cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("High Cloud Albedo")
ax.set_xscale("log")
ax.set_ylim([0, 1])


# %% plot mean profiles of lw fluxes 

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    fluxes_3d["allsky_lw_up"].sel(lat=slice(-30, 30)).mean(["lat", "lon"]),
    fluxes_3d["allsky_lw_up"].pressure / 100,
    label="allsky",
)
ax.plot(
    fluxes_3d["clearsky_lw_up"].sel(lat=slice(-30, 30)).mean(["lat", "lon"]),
    fluxes_3d["clearsky_lw_up"].pressure / 100,
    label="clearsky",
)

ax.invert_yaxis()
ax.legend()
# %%
