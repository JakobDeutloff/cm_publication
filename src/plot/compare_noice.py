# %% import
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

# %% load data from freddis runs
path_freddi = "/work/bm1183/m301049/icon_arts_processed/"

run = "fullrange_flux_mid1deg_noice/"
fluxes_3d_noice = xr.open_dataset(path_freddi + run + "fluxes_3d.nc")
aux = xr.open_dataset(path_freddi + run + "aux.nc")

run = "fullrange_flux_mid1deg/"
fluxes_3d_ice = xr.open_dataset(path_freddi + run + "fluxes_3d.nc")
atms = xr.open_dataset(path_freddi + run + "atms_full.nc")

# %% plot fluxes with and without ice

fig, axes = plt.subplots(
    3,
    2,
    figsize=(15, 5),
    sharex="col",
    sharey="row",
    subplot_kw={"projection": ccrs.PlateCarree()},
)

# LW fluxes
fluxes_3d_ice["allsky_lw_up"].isel(pressure=-1).sel(lat=slice(-30, 30)).plot.contourf(
    ax=axes[0, 0],
    cmap="viridis",
    levels=np.arange(60, 440, 20),
    transform=ccrs.PlateCarree(),
)
axes[0, 0].set_title("allsky_lw_up toa with ice")

fluxes_3d_noice["allsky_lw_up"].isel(pressure=-1).sel(lat=slice(-30, 30)).plot.contourf(
    ax=axes[1, 0],
    cmap="viridis",
    levels=np.arange(60, 440, 20),
    transform=ccrs.PlateCarree(),
)
axes[1, 0].set_title("allsky_lw_up toa without ice")

(
    fluxes_3d_ice["allsky_lw_up"].isel(pressure=-1)
    - fluxes_3d_noice["allsky_lw_up"].isel(pressure=-1)
).sel(lat=slice(-30, 30)).plot.contourf(
    ax=axes[2, 0], cmap="RdBu_r", transform=ccrs.PlateCarree()
)
axes[2, 0].set_title("allsky_lw_up toa with ice - without ice")

# SW fluxes
fluxes_3d_ice["allsky_sw_up"].isel(pressure=-1).sel(lat=slice(-30, 30)).plot.contourf(
    ax=axes[0, 1],
    cmap="viridis",
    levels=np.arange(0, 1300, 100),
    transform=ccrs.PlateCarree(),
)
axes[0, 1].set_title("allsky_sw_up toa with ice")

fluxes_3d_noice["allsky_sw_up"].isel(pressure=-1).sel(lat=slice(-30, 30)).plot.contourf(
    ax=axes[1, 1],
    cmap="viridis",
    levels=np.arange(0, 1300, 100),
    transform=ccrs.PlateCarree(),
)
axes[1, 1].set_title("allsky_sw_up toa without ice")

(
    fluxes_3d_ice["allsky_sw_up"].isel(pressure=-1)
    - fluxes_3d_noice["allsky_sw_up"].isel(pressure=-1)
).sel(lat=slice(-30, 30)).plot.contourf(
    ax=axes[2, 1],
    cmap="RdBu_r",
    levels=np.arange(-400, 500, 100),
    extend="both",
    transform=ccrs.PlateCarree(),
)
axes[2, 1].set_title("allsky_sw_up toa with ice - without ice")

for ax in axes.flatten():
    ax.coastlines()

fig.tight_layout()


# %%
