# %% imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.interpolate import griddata
import pickle

# %% load freddis data
path = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg/"
atms = xr.open_dataset(path + run + "atms_full.nc")
fluxes_3d = xr.open_dataset(path + run  + "fluxes_3d_full.nc")
run = "fullrange_flux_mid1deg_noice/"
fluxes_3d_noice = xr.open_dataset(path + run + "fluxes_3d_full.nc")
lw_vars = xr.open_dataset("data/lw_vars.nc") 


# %% calculate high cloud albedo
sw_vars = xr.Dataset()

def calc_hc_albedo(a_cs, a_as):
    return (a_as - a_cs) / (a_cs * (a_as-2) + 1)

sw_vars["noice_albedo"] = np.abs(
    fluxes_3d_noice.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_3d_noice.isel(pressure=-1)["allsky_sw_down"]
)
sw_vars["allsky_albedo"] = np.abs(
    fluxes_3d.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_3d.isel(pressure=-1)["allsky_sw_down"]
)
sw_vars["high_cloud_albedo"] = calc_hc_albedo(sw_vars["noice_albedo"], sw_vars["allsky_albedo"])

# %% calculate mean albedos by weighting with the incoming SW radiation in IWP bins
IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
SW_down_bins = np.linspace(0, 1360, 30)
binned_hc_albedo = np.zeros([len(IWP_bins) - 1, len(SW_down_bins) - 1]) * np.nan

for i in range(len(IWP_bins) - 1):
    IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] <= IWP_bins[i + 1])
    for j in range(len(SW_down_bins) - 1):
        SW_mask = (fluxes_3d["allsky_sw_down"].isel(pressure=-1) > SW_down_bins[j]) & (
            fluxes_3d["allsky_sw_down"].isel(pressure=-1) <= SW_down_bins[j + 1]
        )
        binned_hc_albedo[i, j] = float(
            (
                sw_vars["high_cloud_albedo"]
                .where(IWP_mask & SW_mask & lw_vars['mask_height'] & lw_vars['mask_hc_no_lc'])
                .sel(lat=slice(-30, 30))
            )
            .mean()
            .values
        )

# %% interpolate albedo bins
non_nan_indices = np.array(np.where(~np.isnan(binned_hc_albedo)))
non_nan_values = binned_hc_albedo[~np.isnan(binned_hc_albedo)]
nan_indices = np.array(np.where(np.isnan(binned_hc_albedo)))
interpolated_values = griddata(
    non_nan_indices.T, non_nan_values, nan_indices.T, method="linear"
)
binned_hc_albedo_interp = binned_hc_albedo.copy()
binned_hc_albedo_interp[np.isnan(binned_hc_albedo)] = interpolated_values

# %% average over SW albedo bins
mean_hc_albedo_SW = np.zeros(len(IWP_points))
mean_hc_albedo_SW_interp = np.zeros(len(IWP_points))
SW_down = (SW_down_bins[1:] + SW_down_bins[:-1]) / 2  # center of SW bins
for i in range(len(IWP_bins) - 1):
    nan_mask = ~np.isnan(binned_hc_albedo[i, :])
    mean_hc_albedo_SW[i] = np.sum(
        binned_hc_albedo[i, :][nan_mask] * SW_down[nan_mask]
    ) / np.sum(SW_down[nan_mask])
    nan_mask_interp = ~np.isnan(binned_hc_albedo_interp[i, :])
    mean_hc_albedo_SW_interp[i] = np.sum(
        binned_hc_albedo_interp[i, :][nan_mask_interp] * SW_down[nan_mask_interp]
    ) / np.sum(SW_down[nan_mask_interp])


# %% plot albedo in SW bins
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

pcol = axes[0].pcolormesh(
    IWP_bins, SW_down_bins, binned_hc_albedo.T , cmap="viridis"
)
axes[1].pcolormesh(
    IWP_bins, SW_down_bins, binned_hc_albedo_interp.T , cmap="viridis"
)

axes[0].set_ylabel("SWin at TOA / W m$^{-2}$")
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("IWP / kg m$^{-2}$")
    ax.set_xlim([1e-4, 5e1])

fig.colorbar(pcol, label="High Cloud Albedo", location="bottom", ax=axes[:], shrink=0.8)

# %% fit polynom to weighted and interpolated albedo
iwp_mask = IWP_points <=1
p = np.polyfit(np.log10(IWP_points[iwp_mask]), mean_hc_albedo_SW_interp[iwp_mask], 5)
poly = np.poly1d(p)
fitted_curve = poly(np.log10(IWP_points[iwp_mask]))

# %% plot fitted albedo in scatterplot with IWP
fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"].where(lw_vars["mask_height"] & lw_vars["mask_hc_no_lc"]).sel(lat=slice(-30, 30)),
    sw_vars["high_cloud_albedo"].where(lw_vars["mask_height"] & lw_vars["mask_hc_no_lc"]).sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d.isel(pressure=-1)["allsky_sw_down"]
    .where(lw_vars["mask_height"] & lw_vars["mask_hc_no_lc"])
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)

ax.plot(IWP_points[iwp_mask], mean_hc_albedo_SW_interp[iwp_mask], label="Mean Albedo", color="k")
ax.plot(IWP_points[iwp_mask], fitted_curve, label="Fitted Polynomial", color="red", linestyle='--')

cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("High Cloud Albedo")
ax.set_xscale("log")
ax.set_xlim([1e-5, 1e1])
ax.set_ylim([0, 1])
ax.legend()
fig.tight_layout()
fig.savefig("plots/albedo.png", dpi=300)

# %% save coefficients as pkl file
with open("data/fitted_albedo.pkl", "wb") as f:
    pickle.dump(p, f)

sw_vars.to_netcdf("data/sw_vars.nc")

# %%
