"""
Script to check whether it makes sense to seperate high and 
low clouds as liquid and ice or whether they are part of the same cloud at high IWP.
"""

# %% import
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from src.read_data import load_atms_and_fluxes, load_derived_vars
from calc_variables import calc_connected

# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
lw_vars, sw_vars, lc_vars = load_derived_vars()


# %% mask
mask = lw_vars["mask_height"] & (atms["IWP"] > 1e-6) & (atms["LWP"] > 1e-6)
# bin data
iwp_bins = np.logspace(-5, 1, 7)

# calculate cloud condensate as sum of all hydrometeors
liq_cld_cond = atms["LWC"] + atms["rain"]
ice_cld_cond = atms["IWC"] + atms["snow"] + atms["graupel"]
cld_cond = liq_cld_cond + ice_cld_cond


# %% plot mean and std of cld condensate in all iwp bins
def plot_condensate(ax, min, max, mask):
    ax2 = ax.twiny()
    mean_liq = (
        liq_cld_cond.where((atms["IWP"] >= min) & (atms["IWP"] < max) & mask)
        .sel(lat=slice(-30, 30))
        .mean(["lat", "lon"])
    )
    mean_ice = (
        ice_cld_cond.where((atms["IWP"] >= min) & (atms["IWP"] < max) & mask)
        .sel(lat=slice(-30, 30))
        .mean(["lat", "lon"])
    )

    mean_rain = (
        atms["rain"]
        .where((atms["IWP"] >= min) & (atms["IWP"] < max) & mask)
        .sel(lat=slice(-30, 30))
        .mean(["lat", "lon"])
    )
    mean_lwc = (
        atms["LWC"]
        .where((atms["IWP"] >= min) & (atms["IWP"] < max) & mask)
        .sel(lat=slice(-30, 30))
        .mean(["lat", "lon"])
    )
    mean_iwc = (
        atms["IWC"]
        .where((atms["IWP"] >= min) & (atms["IWP"] < max) & mask)
        .sel(lat=slice(-30, 30))
        .mean(["lat", "lon"])
    )
    mean_snow = (
        atms["snow"]
        .where((atms["IWP"] >= min) & (atms["IWP"] < max) & mask)
        .sel(lat=slice(-30, 30))
        .mean(["lat", "lon"])
    )
    mean_graupel = (
        atms["graupel"]
        .where((atms["IWP"] >= min) & (atms["IWP"] < max) & mask)
        .sel(lat=slice(-30, 30))
        .mean(["lat", "lon"])
    )

    ax.plot(mean_liq, atms["pressure"] / 100, color="k", label="Liquid")
    ax.plot(mean_rain, atms["pressure"] / 100, color="k", linestyle="--", label="Rain")
    ax.plot(mean_lwc, atms["pressure"] / 100, color="k", linestyle=":", label="LWC")
    ax2.plot(mean_ice, atms["pressure"] / 100, color="b", label="Ice")
    ax2.plot(mean_iwc, atms["pressure"] / 100, color="b", linestyle=":", label="IWC")
    ax2.plot(mean_snow, atms["pressure"] / 100, color="b", linestyle="--", label="Snow")
    ax2.plot(mean_graupel, atms["pressure"] / 100, color="b", linestyle="-.", label="Graupel")
    ax2.set_ylim([50, 1000])
    ax2.set_xlabel("Ice Cond. [kg/kg]", color="b")
    ax.set_xlabel("Liquid Cond. [kg/kg]")
    ax.spines["right"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    return ax2


def plot_connected(ax, min, max, mask):

    mask_selection = mask & (atms["IWP"] > min) & (atms["IWP"] < max)
    connected = calc_connected(atms.where(mask_selection).sel(lat=slice(-30, 30)), rain=False)
    n_profiles = (~np.isnan(connected) * 1).sum().values
    connected_profiles = connected.sum().values

    ax.text(0.05, 0.90, f"{min:.0e} kg/kg - {max:.0e} kg/kg", transform=ax.transAxes)
    ax.text(0.05, 0.85, f"Number of Profiles: {n_profiles:.0f}", transform=ax.transAxes)
    ax.text(
        0.05,
        0.80,
        f"Connected Profiles: {connected_profiles.sum()/n_profiles*100:.1f}%",
        transform=ax.transAxes,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])


fig, axes = plt.subplots(2, 6, figsize=(20, 12), sharey="row")
for i in range(6):
    ax2 = plot_condensate(axes[0, i], iwp_bins[i], iwp_bins[i + 1], mask)
    plot_connected(axes[1, i], iwp_bins[i], iwp_bins[i + 1], mask)

axes[0, 0].invert_yaxis()
axes[0, 0].set_ylabel("Pressure [hPa]")
handles, labels = axes[0, 0].get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(
    handles + handles2, labels + labels2, bbox_to_anchor=(0.5, 0.3), loc="lower center", ncols=4
)
fig.savefig("plots/inspect_cloud_separation.png", dpi=300, bbox_inches="tight")


# %% check the profiles at high IWP that are only connected if rain is included


def plot_n_profiles(ax, connected):
    n_profiles = int((~np.isnan(connected) * 1).sum())
    ax.text(0.05, 0.90, f"Number of Profiles: {n_profiles:.0f}", transform=ax.transAxes)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])


connected_rain = calc_connected(
    atms.where(mask & (atms["IWP"] > 1) & (atms["IWP"] <= 10)).sel(lat=slice(-30, 30)), rain=True
)
n_profiles_rain = (~np.isnan(connected_rain) * 1).sum().values
connected_profiles_rain = connected_rain.sum().values

connected_no_rain = calc_connected(
    atms.where(mask & (atms["IWP"] > 1) & (atms["IWP"] <= 10)).sel(lat=slice(-30, 30)), rain=False
)
n_profiles_no_rain = (~np.isnan(connected_no_rain) * 1).sum().values
connected_profiles_no_rain = connected_no_rain.sum().values

fig, axes = plt.subplots(2, 3, figsize=(10, 10), sharey="row", sharex="row")

plot_condensate(axes[0, 0], 1, 10, mask=(mask & (connected_rain == 1)))
plot_condensate(axes[0, 1], 1, 10, mask=(mask & (connected_rain == 1) & (connected_no_rain == 1)))
plot_condensate(axes[0, 2], 1, 10, mask=(mask & (connected_rain == 1) & (connected_no_rain == 0)))


for ax in axes[0, :]:
    ax.set_ylabel("Pressure [hPa]")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

axes[0, 0].invert_yaxis()
axes[0, 0].set_title("With rain")
axes[0, 1].set_title("With and without rain")
axes[0, 2].set_title("With rain, not without")

connect = np.ones(connected_rain.shape) * np.nan
connect[(connected_rain == 1)] = 1
plot_n_profiles(axes[1, 0], connect)
connect = np.ones(connected_rain.shape) * np.nan
connect[(connected_rain == 1) & (connected_no_rain == 1)] = 1
plot_n_profiles(axes[1, 1], connect)
connect = np.ones(connected_rain.shape) * np.nan
connect[(connected_rain == 1) & (connected_no_rain == 0)] = 1
plot_n_profiles(axes[1, 2], connect)

handles, labels = axes[0, 0].get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(
    handles + handles2, labels + labels2, bbox_to_anchor=(0.5, 0.3), loc="lower center", ncols=4
)


# %%
