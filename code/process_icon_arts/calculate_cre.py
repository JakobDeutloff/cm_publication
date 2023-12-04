# %% import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle

# %% load freddis data
path = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg/"
atms = xr.open_dataset(path + run + "atms_full.nc")
fluxes_3d = xr.open_dataset(path + run + "fluxes_3d_full.nc")
aux = xr.open_dataset(path + run + "aux.nc")
# noice fluxes
run = "fullrange_flux_mid1deg_noice/"
fluxes_3d_noice = xr.open_dataset(path + run + "fluxes_3d_full.nc")

# %% calculate cloud radiative effect

fluxes_toa = fluxes_3d.isel(pressure=-1)
fluxes_toa_noice = fluxes_3d_noice.isel(pressure=-1)
cre_total = xr.Dataset(coords={"lat": atms.lat, "lon": atms.lon})
cre_noice = cre_total.copy()

# clearsky and allsky from normal simulations
cre_total["cloud_rad_effect"] = (
    fluxes_toa["allsky_sw_down"]
    + fluxes_toa["allsky_sw_up"]
    + fluxes_toa["allsky_lw_down"]
    + fluxes_toa["allsky_lw_up"]
    - (fluxes_toa["clearsky_sw_down"] + fluxes_toa["clearsky_sw_up"])
    - (fluxes_toa["clearsky_lw_down"] + fluxes_toa["clearsky_lw_up"])
)

cre_total["sw_cloud_rad_effect"] = (
    fluxes_toa["allsky_sw_down"]
    + fluxes_toa["allsky_sw_up"]
    - (fluxes_toa["clearsky_sw_down"] + fluxes_toa["clearsky_sw_up"])
)

cre_total["lw_cloud_rad_effect"] = (
    fluxes_toa["allsky_lw_down"]
    + fluxes_toa["allsky_lw_up"]
    - (fluxes_toa["clearsky_lw_down"] + fluxes_toa["clearsky_lw_up"])
)

# allsky from normal simulation, clerasky as allsky from noice simulation
cre_noice["cloud_rad_effect"] = (
    fluxes_toa["allsky_sw_down"]
    + fluxes_toa["allsky_sw_up"]
    + fluxes_toa["allsky_lw_down"]
    + fluxes_toa["allsky_lw_up"]
    - (fluxes_toa_noice["allsky_sw_down"] + fluxes_toa_noice["allsky_sw_up"])
    - (fluxes_toa_noice["allsky_lw_down"] + fluxes_toa_noice["allsky_lw_up"])
)

cre_noice["sw_cloud_rad_effect"] = (
    fluxes_toa["allsky_sw_down"]
    + fluxes_toa["allsky_sw_up"]
    - (fluxes_toa_noice["allsky_sw_down"] + fluxes_toa_noice["allsky_sw_up"])
)

cre_noice["lw_cloud_rad_effect"] = (
    fluxes_toa["allsky_lw_down"]
    + fluxes_toa["allsky_lw_up"]
    - (fluxes_toa_noice["allsky_lw_down"] + fluxes_toa_noice["allsky_lw_up"])
)

# %% find profiles with high clouds and no low clouds below and above 8 km
idx_height = atms["IWC"].argmax("pressure")
mask_height = atms["geometric height"].isel(pressure=idx_height) >= 8e3

# %% find deep convective towers 



# %% define functions for calculating CRE
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


def bin_and_average_cre(cre, IWP_bins, lon_bins, atms, modus="ice_only"):
    if modus == "ice_only":
        mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)
    elif modus == "ice_over_lc":
        mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] > 1e-10)
    else:
        mask_hc_no_lc = True

    dummy = np.zeros([len(IWP_bins) - 1, len(lon_bins) - 1])
    cre_arr = {"all": dummy.copy(), "sw": dummy.copy(), "lw": dummy.copy()}

    for i in range(len(IWP_bins) - 1):
        IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i + 1])
        for j in range(len(lon_bins) - 1):
            lon_mask = (fluxes_toa.lon > lon_bins[j]) & (
                fluxes_toa.lon <= lon_bins[j + 1]
            )

            cre_arr["all"][i, j] = float(
                (cre["cloud_rad_effect"].where(IWP_mask & lon_mask & mask_hc_no_lc))
                .mean()
                .values
            )
            cre_arr["sw"][i, j] = float(
                (cre["sw_cloud_rad_effect"].where(IWP_mask & lon_mask & mask_hc_no_lc))
                .mean()
                .values
            )
            cre_arr["lw"][i, j] = float(
                (cre["lw_cloud_rad_effect"].where(IWP_mask & lon_mask & mask_hc_no_lc))
                .mean()
                .values
            )

    # Interpolate
    interp_cre = {
        "all": cre_arr["all"].copy(),
        "sw": cre_arr["sw"].copy(),
        "lw": cre_arr["lw"].copy(),
    }
    for key in interp_cre.keys():
        interp_cre[key] = interpolate(cre_arr[key])

    return cre_arr, interp_cre


# %% calculate cre and interpolate
IWP_bins = np.logspace(-5, 2, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
lon_bins = np.linspace(-180, 180, num=36)

cre_binned = {}
cre_interpolated = {}

# %% high cloud with no low coud below
cre_binned["ice_only"], cre_interpolated["ice_only"] = bin_and_average_cre(
    cre_noice.where(mask_height).sel(lat=slice(-30, 30)),
    IWP_bins,
    lon_bins,
    atms,
    modus="ice_only",
)
# %% all clouds
cre_binned["noice"], cre_interpolated["noice"] = bin_and_average_cre(
    cre_noice.where(mask_height).sel(lat=slice(-30, 30)),
    IWP_bins,
    lon_bins,
    atms,
    modus="noice",
)
# %% high cloud over low cloud
cre_binned["ice_over_lc"], cre_interpolated["ice_over_lc"] = bin_and_average_cre(
    cre_noice.where(mask_height).sel(lat=slice(-30, 30)),
    IWP_bins,
    lon_bins,
    atms,
    modus="ice_over_lc",
)

# %% high clouds with no low clouds below from clearsky fluxes
(
    cre_binned["ice_only_clearsky"],
    cre_interpolated["ice_only_clearsky"],
) = bin_and_average_cre(
    cre_total.where(mask_height).sel(lat=slice(-30, 30)),
    IWP_bins,
    lon_bins,
    atms,
    modus="ice_only",
)

# %% plot binned and interpolated CRE
fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharey="row")

pcol = axes[0, 0].pcolor(
    IWP_bins,
    lon_bins,
    cre_binned["ice_only"]["all"].T,
    cmap="seismic",
    vmin=-600,
    vmax=600,
)
axes[0, 0].set_ylabel("Longitude [deg]")
axes[0, 0].set_title("CRE binned ice only")
axes[0, 1].pcolor(
    IWP_bins,
    lon_bins,
    cre_interpolated["ice_only"]["all"].T,
    cmap="seismic",
    vmin=-600,
    vmax=600,
)
axes[0, 1].set_title("CRE interpolated ice only")
axes[1, 0].pcolor(
    IWP_bins,
    lon_bins,
    cre_binned["noice"]["all"].T,
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
    cre_interpolated["noice"]["all"].T,
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

# %% Average over longitude bins
mean_cre_noice = {
    "all": np.nanmean(cre_binned["noice"]["all"], axis=1),
    "sw": np.nanmean(cre_binned["noice"]["sw"], axis=1),
    "lw": np.nanmean(cre_binned["noice"]["lw"], axis=1),
}
mean_cre_noice_interp = {
    "all": np.nanmean(cre_interpolated["noice"]["all"], axis=1),
    "sw": np.nanmean(cre_interpolated["noice"]["sw"], axis=1),
    "lw": np.nanmean(cre_interpolated["noice"]["lw"], axis=1),
}

mean_cre_ice_only = {
    "all": np.nanmean(cre_binned["ice_only"]["all"], axis=1),
    "sw": np.nanmean(cre_binned["ice_only"]["sw"], axis=1),
    "lw": np.nanmean(cre_binned["ice_only"]["lw"], axis=1),
}

mean_cre_ice_only_interp = {
    "all": np.nanmean(cre_interpolated["ice_only"]["all"], axis=1),
    "sw": np.nanmean(cre_interpolated["ice_only"]["sw"], axis=1),
    "lw": np.nanmean(cre_interpolated["ice_only"]["lw"], axis=1),
}

mean_cre_ice_over_lc = {
    "all": np.nanmean(cre_binned["ice_over_lc"]["all"], axis=1),
    "sw": np.nanmean(cre_binned["ice_over_lc"]["sw"], axis=1),
    "lw": np.nanmean(cre_binned["ice_over_lc"]["lw"], axis=1),
}

mean_cre_ice_over_lc_interp = {
    "all": np.nanmean(cre_interpolated["ice_over_lc"]["all"], axis=1),
    "sw": np.nanmean(cre_interpolated["ice_over_lc"]["sw"], axis=1),
    "lw": np.nanmean(cre_interpolated["ice_over_lc"]["lw"], axis=1),
}

mean_cre_ice_only_clearsky = {
    "all": np.nanmean(cre_binned["ice_only_clearsky"]["all"], axis=1),
    "sw": np.nanmean(cre_binned["ice_only_clearsky"]["sw"], axis=1),
    "lw": np.nanmean(cre_binned["ice_only_clearsky"]["lw"], axis=1),
}

mean_cre_ice_only_clearsky_interp = {
    "all": np.nanmean(cre_interpolated["ice_only_clearsky"]["all"], axis=1),
    "sw": np.nanmean(cre_interpolated["ice_only_clearsky"]["sw"], axis=1),
    "lw": np.nanmean(cre_interpolated["ice_only_clearsky"]["lw"], axis=1),
}


# %% plot mean CRE vs IWP
fig, axes = plt.subplots(1, 3, sharey="row", figsize=(10, 4))

end = -13

# ice only
axes[0].plot(
    IWP_points[:end],
    mean_cre_ice_only["all"][:end],
    label="CRE ",
    color="k",
    alpha=0.3,
)
axes[0].plot(
    IWP_points[:end],
    mean_cre_ice_only_interp["all"][:end],
    label="CRE interpolated",
    color="k",
)
axes[0].plot(
    IWP_points[:end],
    mean_cre_ice_only["sw"][:end],
    label="SW CRE",
    color="blue",
    alpha=0.3,
)
axes[0].plot(
    IWP_points[:end],
    mean_cre_ice_only_interp["sw"][:end],
    label="SW CRE interpolated",
    color="blue",
)
axes[0].plot(
    IWP_points[:end],
    mean_cre_ice_only["lw"][:end],
    label="LW CRE",
    color="r",
    alpha=0.3,
)
axes[0].plot(
    IWP_points[:end],
    mean_cre_ice_only_interp["lw"][:end],
    label="LW CRE interpolated",
    color="r",
)

# noice
axes[1].plot(
    IWP_points[:end],
    mean_cre_noice["all"][:end],
    label="CRE",
    color="k",
    alpha=0.3,
)
axes[1].plot(
    IWP_points[:end],
    mean_cre_noice_interp["all"][:end],
    label="CRE interpolated",
    color="k",
)
axes[1].plot(
    IWP_points[:end],
    mean_cre_noice["sw"][:end],
    label="SW CRE",
    color="blue",
    alpha=0.3,
)
axes[1].plot(
    IWP_points[:end],
    mean_cre_noice_interp["sw"][:end],
    label="SW CRE interpolated",
    color="blue",
)
axes[1].plot(
    IWP_points[:end],
    mean_cre_noice["lw"][:end],
    label="LW CRE",
    color="r",
    alpha=0.3,
)
axes[1].plot(
    IWP_points[:end],
    mean_cre_noice_interp["lw"][:end],
    label="LW CRE interpolated",
    color="r",
)

# ice over lc
axes[2].plot(
    IWP_points[:end],
    mean_cre_ice_over_lc["all"][:end],
    label="CRE",
    color="k",
    alpha=0.3,
)

axes[2].plot(
    IWP_points[:end],
    mean_cre_ice_over_lc_interp["all"][:end],
    label="CRE interpolated",
    color="k",
)

axes[2].plot(
    IWP_points[:end],
    mean_cre_ice_over_lc["sw"][:end],
    label="SW CRE",
    color="blue",
    alpha=0.3,
)

axes[2].plot(
    IWP_points[:end],
    mean_cre_ice_over_lc_interp["sw"][:end],
    label="SW CRE interpolated",
    color="blue",
)

axes[2].plot(
    IWP_points[:end],
    mean_cre_ice_over_lc["lw"][:end],
    label="LW CRE",
    color="r",
    alpha=0.3,
)

axes[2].plot(
    IWP_points[:end],
    mean_cre_ice_over_lc_interp["lw"][:end],
    label="LW CRE interpolated",
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
axes[1].set_title("All Clouds")
axes[2].set_title("High Clouds over Low Clouds")


# legend outside of axes
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))

fig.tight_layout()
fig.savefig("plots/mean_CRE_vs_IWP.png", dpi=300)


# %% save mean CRE
with open("data/hc_cre.pkl", "wb") as f:
    pickle.dump([IWP_points[:end], mean_cre_interp["all"][:end]], f)

with open("data/hc_cre_sw.pkl", "wb") as f:
    pickle.dump([IWP_points[:end], mean_cre_interp["sw"][:end]], f)

with open("data/hc_cre_lw.pkl", "wb") as f:
    pickle.dump([IWP_points[:end], mean_cre_interp["lw"][:end]], f)

# %%
