# %% import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pickle
from src.icon_arts_analysis import calc_cre, bin_and_average_cre

# %% load freddis data
path = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg/"
atms = xr.open_dataset(path + run + "atms_full.nc")
fluxes_3d = xr.open_dataset(path + run + "fluxes_3d_full.nc")
# noice fluxes
run = "fullrange_flux_mid1deg_noice/"
fluxes_3d_noice = xr.open_dataset(path + run + "fluxes_3d_full.nc")

# %% calculate cre
fluxes_toa = fluxes_3d.isel(pressure=-1)
fluxes_toa_noice = fluxes_3d_noice.isel(pressure=-1)
cre_clearsky = xr.Dataset(coords={"lat": atms.lat, "lon": atms.lon})
cre_noice = cre_clearsky.copy()

cre_clearsky = calc_cre(fluxes_toa, mode='clearsky')
cre_noice = calc_cre(fluxes_toa, fluxes_toa_noice, mode='noice')

# %% find profiles with high clouds and no low clouds below and above 8 km
idx_height = atms["IWC"].argmax("pressure")
mask_height = atms["geometric height"].isel(pressure=idx_height) >= 8e3

# %% calculate cre in bins and interpolate
IWP_bins = np.logspace(-5, 2, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
lon_bins = np.linspace(-180, 180, num=36)

cre_binned = {}
cre_interpolated = {}
cre_interpolated_average = {}

# %% all clouds
cre_binned["all"], cre_interpolated["all"], cre_interpolated_average["all"] = bin_and_average_cre(
    cre_noice.where(mask_height).sel(lat=slice(-30, 30)),
    IWP_bins,
    lon_bins,
    atms,
    modus="noice",
)
# %% high cloud with no low coud below
cre_binned["ice_only"], cre_interpolated["ice_only"], cre_interpolated_average["ice_only"] = bin_and_average_cre(
    cre_noice.where(mask_height).sel(lat=slice(-30, 30)),
    IWP_bins,
    lon_bins,
    atms,
    modus="ice_only",
)
# %% high cloud over low cloud
cre_binned["ice_over_lc"], cre_interpolated["ice_over_lc"], cre_interpolated_average["ice_over_lc"] = bin_and_average_cre(
    cre_noice.where(mask_height).sel(lat=slice(-30, 30)),
    IWP_bins,
    lon_bins,
    atms,
    modus="ice_over_lc",
)
# %% plot interpolated mean CRE
fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey="row")

pcol = axes[0, 0].pcolor(
    IWP_bins,
    lon_bins,
    cre_binned["ice_only"]["net"].T,
    cmap="seismic",
    vmin=-600,
    vmax=600,
)
axes[0, 0].set_ylabel("Longitude [deg]")
axes[0, 0].set_title("CRE binned ice only")
axes[0, 1].pcolor(
    IWP_bins,
    lon_bins,
    cre_interpolated["ice_only"]["net"].T,
    cmap="seismic",
    vmin=-600,
    vmax=600,
)
axes[0, 1].set_title("CRE interpolated ice only")
axes[1, 0].pcolor(
    IWP_bins,
    lon_bins,
    cre_binned["all"]["net"].T,
    cmap="seismic",
    vmin=-600,
    vmax=600,
)
axes[1, 0].set_ylabel("Longitude [deg]")
axes[1, 0].set_xlabel("IWP [kg m$^{-2}$]")
axes[1, 0].set_title("CRE binned all clouds")
axes[1, 1].pcolor(
    IWP_bins,
    lon_bins,
    cre_interpolated["all"]["net"].T,
    cmap="seismic",
    vmin=-600,
    vmax=600,
)
axes[1, 1].set_xlabel("IWP [kg m$^{-2}$]")
axes[1, 1].set_title("CRE interpolated all clouds")

for ax in axes.flatten():
    ax.set_xscale("log")

fig.colorbar(
    pcol,
    label="High Cloud Radiative Effect",
    location="bottom",
    ax=axes[:],
    shrink=0.7,
    extend="min",
    pad=0.1,
)

fig.savefig("plots/CRE_binned_by_IWP_and_lon.png", dpi=300, bbox_inches="tight")

# %% plot mean CRE vs IWP
fig, axes = plt.subplots(1, 3, sharey="row", figsize=(12, 4))

end = -13

# ice only
axes[0].plot(
    IWP_points[:end],
    cre_interpolated_average['ice_only']["net"][:end],
    label="Net CRE",
    color="k",
)
axes[0].plot(
    IWP_points[:end],
    cre_interpolated_average['ice_only']["sw"][:end],
    label="SW CRE",
    color="blue",
)
axes[0].plot(
    IWP_points[:end],
    cre_interpolated_average['ice_only']["lw"][:end],
    label="LW CRE",
    color="r",
)

# ice over lc
axes[1].plot(
    IWP_points[:end],
    cre_interpolated_average['ice_over_lc']["net"][:end],
    label="Net CRE",
    color="k",
)
axes[1].plot(
    IWP_points[:end],
    cre_interpolated_average['ice_over_lc']["sw"][:end],
    label="SW CRE",
    color="blue",
)
axes[1].plot(
    IWP_points[:end],
    cre_interpolated_average['ice_over_lc']["lw"][:end],
    label="LW CRE",
    color="r",
)

# noice
axes[2].plot(
    IWP_points[:end],
    cre_interpolated_average['all']["net"][:end],
    label="Net CRE",
    color="k",
)
axes[2].plot(
    IWP_points[:end],
    cre_interpolated_average['all']["sw"][:end],
    label="SW CRE",
    color="blue",
)
axes[2].plot(
    IWP_points[:end],
    cre_interpolated_average['all']["lw"][:end],
    label="LW CRE",
    color="r",
)


for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlabel("IWP / kg m$^{-2}$")
    ax.axhline(0, color="k", linestyle="--")

axes[0].set_ylabel("Cloud Radiative Effect / W m$^{-2}$")
axes[0].set_title("High Clouds no Low Clouds")
axes[1].set_title("High Clouds over Low Clouds")
axes[2].set_title("All High Clouds")


# legend outside of axes
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.15))
fig.savefig("plots/mean_CRE_vs_IWP.png", dpi=300, bbox_inches="tight")


# %% save mean CRE
with open("data/hc_cre.pkl", "wb") as f:
    pickle.dump([IWP_points, cre_interpolated_average], f)


# %%
