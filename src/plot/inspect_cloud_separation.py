"""
Script to check whether it makes sense to seperate high and 
low clouds as liquid and ice or whether they are part of the same cloud at high IWP.
"""
# %% import
import numpy as np
import matplotlib.pyplot as plt
from src.read_data import load_atms_and_fluxes, load_cre, load_derived_vars

# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
cre_binned, cre_interpolated, cre_average = load_cre()
lw_vars, sw_vars, lc_vars = load_derived_vars()

# %%

# select ice clouds with low clouds below
mask = (atms['IWP'] > 1e-6) & (atms['LWP'] > 1e-6)

# bin data
iwp_bins = np.logspace(-5, 1, 7)

# calculate cloud condensate as sum of all hydrometeors
liq_cld_cond = atms["LWC"] + atms["rain"]
ice_cld_cond = atms["IWC"] + atms["snow"] + atms["graupel"]
cld_cond = liq_cld_cond + ice_cld_cond

# %% define conetctedness 
frac_no_cloud = 0.1  # fraction of maximum cloud condensate in colum to define no cloud
no_ice_cloud = (ice_cld_cond > (frac_no_cloud * ice_cld_cond.max('pressure'))) * 1
no_liq_cloud = (liq_cld_cond > (frac_no_cloud * liq_cld_cond.max('pressure'))) * 1
no_cld = no_liq_cloud + no_ice_cloud

# %% plot mean and std of cld condensate in all iwp bins
def plot_condensate(ax, min, max):
    ax2 = ax.twiny()
    liq = liq_cld_cond.where((atms["IWP"] >= min) & (atms["IWP"] < max) & mask).sel(
        lat=slice(-30, 30)
    )
    ice = ice_cld_cond.where((atms["IWP"] >= min) & (atms["IWP"] < max) & mask).sel(
        lat=slice(-30, 30)
    )
    mean_liq = liq.mean(["lat", "lon"])
    mean_ice = ice.mean(["lat", "lon"])
    ax.plot(mean_liq, atms["pressure"] / 100, color="k")
    ax.set_ylim([50, 1000])
    ax2.plot(mean_ice, atms["pressure"] / 100, color="b")
    ax2.set_xlabel("Ice Cond. [kg/kg]", color='b')
    ax.set_xlabel("Liquid Cond. [kg/kg]")
    ax.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)

def plot_connected(ax, min, max):
    sample = no_cld.where((atms["IWP"] >= min) & (atms["IWP"] < max) & mask).sel(
        lat=slice(-30, 30)
    )
    valid = sample.sum('pressure') > 0
    n_profiles = int((valid*1).sum().values)
    # get lat an lon coordinates of valid profiles
    lon, lat = np.meshgrid(valid.lon, valid.lat)
    lat_valid = lat[valid]
    lon_valid = lon[valid]

    unconnected_profiles = np.zeros(n_profiles)
    for i in range(n_profiles):
        liq_point = liq_cld_cond.sel(lat=lat_valid[i], lon=lon_valid[i])
        ice_point = ice_cld_cond.sel(lat=lat_valid[i], lon=lon_valid[i])
        p_top_idx = ice_point.argmax('pressure').values
        p_bot_idx = liq_point.argmax('pressure').values
        cld_range = no_cld.sel(lat=lat_valid[i], lon=lon_valid[i]).isel(pressure=slice(p_bot_idx, p_top_idx))
        for j in range(len(cld_range.pressure)-2):
            if cld_range.isel(pressure=slice(j, j+2)).sum() == 0:
                unconnected_profiles[i] += 1
                break
    
    ax.text(0.05, 0.90, f"{min:.0e} kg/kg - {max:.0e} kg/kg", transform=ax.transAxes)
    ax.text(0.05, 0.85, f"Number of Profiles: {n_profiles:.0f}", transform=ax.transAxes)
    ax.text(0.05, 0.80, f"Unconnected Profiles: {unconnected_profiles.sum()/n_profiles*100:.1f}%", transform=ax.transAxes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
        

fig, axes = plt.subplots(2, 6, figsize=(20, 12), sharey='row')
for i in range(6):
    plot_condensate(axes[0, i], iwp_bins[i], iwp_bins[i + 1])
    plot_connected(axes[1, i], iwp_bins[i], iwp_bins[i + 1])

axes[0, 0].invert_yaxis()
axes[0, 0].set_ylabel("Pressure [hPa]")
fig.savefig("plots/inspect_cloud_separation.png", dpi=300, bbox_inches="tight")


# %%
