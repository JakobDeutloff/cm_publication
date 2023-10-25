# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs  

# %% load data from freddis runs
path_freddi = "/work/bm1183/m301049/freddi_runs/"
atms = xr.open_dataset(path_freddi + "atms.nc")
fluxes_3d = xr.open_dataset(path_freddi + "fluxes_3d.nc")
fluxes_2d = xr.open_dataset(path_freddi + "fluxes_2d.nc")
aux = xr.open_dataset(path_freddi + "aux.nc")

# %% calculate IWP and LWP
cell_height = atms["geometric height"].diff("pressure")
atms["IWP"] = ((atms["IWC"] + atms["snow"] + atms["graupel"]) * cell_height).sum("pressure")
atms["LWP"] = ((atms["rain"] + atms["LWC"]) * cell_height).sum("pressure")

# %% find high clouds with no low clouds below
mask_hc_no_lc = (atms["IWP"] > 1) & (atms["LWP"] < 0.1)
lon_3d, lat_3d = np.meshgrid(atms["lon"], atms["lat"])
lons = lon_3d[mask_hc_no_lc]
lats = lat_3d[mask_hc_no_lc]

# %% plot location of profiles
fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": ccrs.PlateCarree()})
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.scatter(lons, lats, s=0.1, color="black")
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False

# %% select random profile from sample

raw_lat = 0
raw_lon = -1

idx_lat = np.argmin(np.abs(lats - raw_lat))
idx_lon = np.argmin(np.abs(lons - raw_lon))
lat = lats[idx_lat]
lon = lons[idx_lon]

# %%  plot profiles at point


def plot_profiles(lat, lon):
    fig, axes = plt.subplots(2, 4, figsize=(10, 10), sharey="row")
    data = atms.sel(lat=lat, lon=lon, method="nearest")
    fluxes = fluxes_3d.sel(lat=lat, lon=lon, method="nearest")
    flx_2d = fluxes_2d.sel(lat=lat, lon=lon, method="nearest")
    height = data["geometric height"] / 1e3

    # plot frozen hydrometeors
    axes[0, 0].plot(data["IWC"], height, label="IWC", color="k")
    axes[0, 0].plot(data["snow"], height, label="snow", color="k", linestyle="--")
    axes[0, 0].plot(data["graupel"], height, label="graupel", color="k", linestyle=":")
    axes[0, 0].set_ylabel("height / km")
    axes[0, 0].set_xlabel("F. Hyd. / kg m$^{-3}$")
    axes[0, 0].legend()

    # plot liquid hydrometeors
    axes[0, 1].plot(data["LWC"], height, label="LWC", color="k")
    axes[0, 1].plot(data["rain"], height, label="rain", color="k", linestyle="--")
    axes[0, 1].set_xlabel("L. Hyd. / kg m$^{-3}$")
    axes[0, 1].legend()

    # plot temperature
    axes[0, 2].plot(data["temperature"], height, color="black")
    axes[0, 2].set_xlabel("Temperature / K")

    # plot LW fluxes up
    axes[1, 0].plot(fluxes["allsky_lw_up"], height, label="allsky", color="k")
    axes[1, 0].plot(
        fluxes["clearsky_lw_up"], height, label="clearsky", color="k", linestyle="--"
    )
    axes[1, 0].set_ylabel("height / km")
    axes[1, 0].set_xlabel("LW Up / W m$^{-2}$")
    axes[1, 0].legend()

    # plot LW fluxes down
    axes[1, 1].plot(fluxes["allsky_lw_down"], height, label="allsky", color="k")
    axes[1, 1].plot(
        fluxes["clearsky_lw_down"], height, label="clearsky", color="k", linestyle="--"
    )
    axes[1, 1].set_xlabel("LW Down / W m$^{-2}$")
    axes[1, 1].legend()

    # plot SW fluxes up
    axes[1, 2].plot(fluxes["allsky_sw_up"], height, label="allsky", color="k")
    axes[1, 2].plot(
        fluxes["clearsky_sw_up"], height, label="clearsky", color="k", linestyle="--"
    )
    axes[1, 2].set_xlabel("SW Up / W m$^{-2}$")
    axes[1, 2].legend()

    # plot SW fluxes down
    axes[1, 3].plot(fluxes["allsky_sw_down"], height, label="allsky", color="k")
    axes[1, 3].plot(
        fluxes["clearsky_sw_down"], height, label="clearsky", color="k", linestyle="--"
    )
    axes[1, 3].set_xlabel("SW Down / W m$^{-2}$")
    axes[1, 3].legend()

    for ax in axes.flatten():
        ax.set_ylim(0, 20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # plot coordinates and toa fluxes
    axes[0, 3].remove()
    fig.text(
        0.9,
        0.8,
        f"lat: {lat.round(2)}\nlon: {lon.round(2)}",
        ha="center",
        va="center",
        fontsize=11,
    )

    fig.tight_layout()


# %%
plot_profiles(lat, lon)


# %%
