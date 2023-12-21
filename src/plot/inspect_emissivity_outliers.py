# %% import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# %% load freddis data
path = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg/"
atms = xr.open_dataset(path + run + "atms_full.nc")
fluxes_3d = xr.open_dataset(path + run + "fluxes_3d_full.nc")

run = "fullrange_flux_mid1deg_noice/"
fluxes_3d_noice = xr.open_dataset(path + run + "fluxes_3d_full.nc")

lw_vars = xr.open_dataset("data/lw_vars.nc")

# %% find profiles with high clouds and no low clouds below and above 8 km
mask_height = atms["geometric height"].isel(pressure=(atms["IWC"] + atms["graupel"] + atms['snow']).argmax("pressure")) >= 8e3

# %% find coords of emssivity IWP pair
em = 0.1
IWP = 1e-5
diff = np.abs(lw_vars["high_cloud_emissivity"].where(mask_height) - em) + np.abs(atms["IWP"].where(mask_height) - IWP)
lat, lon = int(diff.argmin(dim=["lat", "lon"])["lat"].values), int(
    diff.argmin(dim=["lat", "lon"])["lon"].values
)
lat, lon = (
    atms.isel(lat=lat, lon=lon)["lat"].values,
    atms.isel(lat=lat, lon=lon)["lon"].values,
)

# %% plot profiles
lw_vars_point = lw_vars.sel(lat=lat, lon=lon)
pressure = atms.sel(lat=lat, lon=lon)["pressure"] / 100

sigma = 5.67e-8

fig, axes = plt.subplots(1, 4, figsize=(12, 5), sharey="row")

# plot lw out  profiles
axes[0].plot(
    -fluxes_3d["allsky_lw_up"].sel(lat=lat, lon=lon),
    pressure,
    color="k",
    label="allsky",
)
axes[0].plot(
    -fluxes_3d_noice["allsky_lw_up"].sel(lat=lat, lon=lon),
    pressure,
    color="k",
    label="allsky noice",
    linestyle="--",
)
axes[0].plot(
    -fluxes_3d["clearsky_lw_up"].sel(lat=lat, lon=lon),
    pressure,
    color="k",
    label="clearsky",
    linestyle=":",
)
axes[0].axhline(
    lw_vars_point["h_cloud_top_pressure"].values / 100, color="lime", linestyle=":"
)
axes[0].plot(
    sigma * lw_vars_point["h_cloud_temperature"] ** 4,
    lw_vars_point["h_cloud_top_pressure"] / 100,
    color="lime",
    linestyle="",
    marker="o",
)
axes[0].set_xlabel("LW Up / W m$^{-2}$")
axes[0].set_ylabel("Pressure / hPa")
axes[0].legend()

# plot temperature profile
axes[1].plot(atms["temperature"].sel(lat=lat, lon=lon), pressure, color="k")
axes[1].set_xlabel("Temperature / K")

# plot hydrometeors
ax3 = axes[2].twiny()
axes[2].plot(atms["IWC"].sel(lat=lat, lon=lon), pressure, color="k", label="IWC")
axes[2].plot(
    atms["snow"].sel(lat=lat, lon=lon),
    pressure,
    color="k",
    linestyle="--",
    label="snow",
)
axes[2].plot(
    atms["graupel"].sel(lat=lat, lon=lon),
    pressure,
    color="k",
    linestyle=":",
    label="graupel",
)
ax3.plot(
    atms["LWC"].sel(lat=lat, lon=lon), pressure, color="red", linestyle="-", label="LWC"
)
ax3.plot(
    atms["rain"].sel(lat=lat, lon=lon),
    pressure,
    color="red",
    linestyle="--",
    label="rain",
)
axes[2].set_xlabel("Frozen Hydrometeors / kg kg$^{-1}$")
ax3.set_xlabel("Liquid Hydrometeors / kg kg$^{-1}$", color="red")
axes[2].legend()

# put numbers of emissivity, IWP and LWP as text
axes[3].text(
    0.5,
    0.5,
    f"emissivity: {lw_vars_point['high_cloud_emissivity'].values:.2f}\nIWP: {atms['IWP'].sel(lat=lat, lon=lon).values}\nLWP: {atms['LWP'].sel(lat=lat, lon=lon).values:.2f}",
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[3].transAxes,
)


for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].invert_yaxis()
ax3.spines["right"].set_visible(False)
axes[3].spines["left"].set_visible(False)
axes[3].spines["bottom"].set_visible(False)
axes[3].set_xticks([])

fig.tight_layout()
# %%
