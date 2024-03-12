# %% import
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from calc_variables import calc_connected

# %% load data
path = "/work/bm1183/m301049/nextgems_profiles/"
atms = xr.open_dataset(path + "all_profiles_Processed_ozone.nc")

# %% bin data
iwp_bins = np.logspace(-5, 1, 7)

# %% calculate cloud condensate as sum of all hydrometeors
liq_cld_cond = atms["clw"] + atms["qr"]
ice_cld_cond = atms["cli"] + atms["qs"] + atms["qg"]
cld_cond = liq_cld_cond + ice_cld_cond


# %% plot mean and std of cld condensate in all iwp bins
def plot_condensate(ax, min, max, mask):
    ax2 = ax.twiny()
    mean_liq = liq_cld_cond.where((atms["IWP"] >= min) & (atms["IWP"] < max)).mean(
        ["sw_points", "profile", "iwp_points"]
    )
    mean_ice = ice_cld_cond.where((atms["IWP"] >= min) & (atms["IWP"] < max)).mean(
        ["sw_points", "profile", "iwp_points"]
    )

    mean_rain = (
        atms["rain"]
        .where((atms["IWP"] >= min) & (atms["IWP"] < max))
        .mean(["sw_points", "profile", "iwp_points"])
    )
    mean_lwc = (
        atms["LWC"]
        .where((atms["IWP"] >= min) & (atms["IWP"] < max))
        .mean(["sw_points", "profile", "iwp_points"])
    )
    mean_iwc = (
        atms["IWC"]
        .where((atms["IWP"] >= min) & (atms["IWP"] < max))
        .mean(["sw_points", "profile", "iwp_points"])
    )
    mean_snow = (
        atms["snow"]
        .where((atms["IWP"] >= min) & (atms["IWP"] < max))
        .mean(["sw_points", "profile", "iwp_points"])
    )
    mean_graupel = (
        atms["graupel"]
        .where((atms["IWP"] >= min) & (atms["IWP"] < max))
        .mean(["sw_points", "profile", "iwp_points"])
    )

    ax.plot(mean_liq, atms["level_full"] / 100, color="k", label="Liquid")
    ax.plot(mean_rain, atms["level_full"] / 100, color="k", linestyle="--", label="Rain")
    ax.plot(mean_lwc, atms["level_full"] / 100, color="k", linestyle=":", label="LWC")
    ax2.plot(mean_ice, atms["level_full"] / 100, color="b", label="Ice")
    ax2.plot(mean_iwc, atms["level_full"] / 100, color="b", linestyle=":", label="IWC")
    ax2.plot(mean_snow, atms["level_full"] / 100, color="b", linestyle="--", label="Snow")
    ax2.plot(mean_graupel, atms["level_full"] / 100, color="b", linestyle="-.", label="Graupel")
    ax2.set_ylim([50, 1000])
    ax2.set_xlabel("Ice Cond. [kg/m^2]", color="b")
    ax.set_xlabel("Liquid Cond. [kg/m^2]")
    ax.spines["right"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    return ax2


def plot_connected(ax, min, max, mask):

    mask_selection = mask & (atms["IWP"] > min) & (atms["IWP"] < max)
    connected = calc_connected(atms.where(mask_selection), rain=False)
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
    ax2 = plot_condensate(axes[0, i], iwp_bins[i], iwp_bins[i + 1])
    plot_connected(axes[1, i], iwp_bins[i], iwp_bins[i + 1])

axes[0, 0].invert_yaxis()
axes[0, 0].set_ylabel("level_full")
handles, labels = axes[0, 0].get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(
    handles + handles2, labels + labels2, bbox_to_anchor=(0.5, 0.3), loc="lower center", ncols=4
)
# fig.savefig("plots/inspect_cloud_separation.png", dpi=300, bbox_inches="tight")

# %%
