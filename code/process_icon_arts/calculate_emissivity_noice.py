# %% import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle

# %% load freddis data
path = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg/"
fluxes_3d = xr.open_dataset(path + run + "fluxes_3d_full.nc")
run = "fullrange_flux_mid1deg_noice/"
fluxes_3d_noice = xr.open_dataset(path + run + "fluxes_3d_full.nc")
atms = xr.open_dataset(path + run + "atms_full.nc")

# %% find profiles with high clouds and no low clouds below and above 8 km
idx_height = (atms["IWC"] + atms['snow'] + atms['graupel']).argmax("pressure")
mask_graupel = atms.isel(pressure=idx_height)["pressure"] < 35000
mask_iwc = atms["IWC"].max('pressure') > 1e-12
mask_valid = mask_graupel & mask_iwc

# %% initialize dataset for new variables
lw_vars = xr.Dataset()

# %% calculate high cloud temperature from vertically integrated IWP
IWC_emission = 1e-3  # IWP where high clouds become opaque

p_top_idx_thin = atms["IWC"].argmax("pressure")
p_top_bool_thick = atms["IWC_cumsum"] > IWC_emission
p_top_idx_thick = p_top_bool_thick.argmin("pressure")
p_top_idx = xr.where(p_top_idx_thick > p_top_idx_thin, p_top_idx_thick, p_top_idx_thin)
p_top = atms.isel(pressure=p_top_idx).pressure
T_h_lw = atms["temperature"].sel(pressure=p_top)
lw_vars["h_cloud_temperature"] = T_h_lw
lw_vars["h_cloud_top_pressure"] = p_top

mask_height = p_top < 35000
mask_valid = mask_valid & mask_height

# %% calculate high cloud emissivity
sigma = 5.67e-8  # W m-2 K-4
LW_out_as = fluxes_3d.isel(pressure=-1)["allsky_lw_up"]
LW_out_cs = fluxes_3d_noice.isel(pressure=-1)["allsky_lw_up"]
correction = (
    LW_out_as
    - fluxes_3d.sel(pressure=lw_vars["h_cloud_top_pressure"])[
        "allsky_lw_up"
    ]
)
rad_hc = -lw_vars["h_cloud_temperature"] ** 4 * sigma
hc_emissivity = (LW_out_as - LW_out_cs) / (rad_hc - LW_out_cs)
lw_vars["high_cloud_emissivity"] = hc_emissivity
lw_vars["rad_correction"] = correction
lw_vars["mask_valid"] = mask_valid

# %% aveage over IWP bins
IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
mean_hc_emissivity = np.zeros(len(IWP_bins) - 1)
mean_correction_factor = np.zeros(len(IWP_bins) - 1)

for i in range(len(IWP_bins) - 1):
    IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i + 1])
    mean_hc_emissivity[i] = float(
        lw_vars["high_cloud_emissivity"]
        .where(IWP_mask & mask_valid)
        .sel(lat=slice(-30, 30))
        .mean()
        .values
    )
    mean_correction_factor[i] = float(
        lw_vars["rad_correction"].where(IWP_mask & mask_valid).sel(lat=slice(-30, 30)).mean().values
    )

# %% fit polynomial to mean emissivity
mean_hc_emissivity_adapted = mean_hc_emissivity.copy()
mean_hc_emissivity_adapted[IWP_points>1e-1] = 1
p_emm = np.polyfit(np.log10(IWP_points), mean_hc_emissivity_adapted, 9)


def hc_emissivity(IWP, coeffs):
    fitted_vals = np.poly1d(coeffs)(np.log10(IWP))
    return fitted_vals


fitted_emissivity = hc_emissivity(IWP_points, p_emm)

# %% fit polynomial to mean correction factor
p_cor = np.polyfit(
    np.log10(IWP_points[~np.isnan(mean_correction_factor)]),
    mean_correction_factor[~np.isnan(mean_correction_factor)],
    9,
)


def hc_correction_factor(IWP, coeffs):
    fitted_vals = np.poly1d(coeffs)(np.log10(IWP))
    return fitted_vals


fitted_correction_factor = hc_correction_factor(IWP_points, p_cor)

# %% plot mean hv emissivity in scatterplot with IWP
fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"].where(mask_valid).sel(lat=slice(-30, 30)),
    lw_vars["high_cloud_emissivity"]
    .where(mask_valid)
    .sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d["allsky_sw_down"]
    .isel(pressure=-1)
    .where(mask_valid)
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)

ax.plot(IWP_points, mean_hc_emissivity, color="lime", label="Median Emissivity")
ax.plot(
    IWP_points, fitted_emissivity, color="r", label="Fitted Polynomial", linestyle="--"
)
ax.axhline(1, color="grey", linestyle="--")

cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("High Cloud Emissivity")
ax.set_xscale("log")
ax.set_xlim(1e-5, 10)
ax.set_ylim(-0.2, 1.5)
ax.legend()
fig.savefig("plots/emissivity.png", dpi=300, bbox_inches="tight")

# %% plot mean correction factior in scatterplot with IWP
fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"].where(mask_valid).sel(lat=slice(-30, 30)),
    lw_vars["rad_correction"].where(mask_valid).sel(lat=slice(-30, 30)),
    s=0.5,
    color="k",
)

ax.plot(
    IWP_points, mean_correction_factor, color="lime", label="Mean Correction Factor"
)
ax.plot(
    IWP_points,
    fitted_correction_factor,
    color="r",
    label="Fitted Polynomial",
    linestyle="--",
)

ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Correction Factor")
ax.set_xscale("log")
ax.legend()
fig.savefig("plots/correction_factor.png", dpi=300, bbox_inches="tight")

# %% save coefficients as pkl file
with open("data/fitted_emissivity.pkl", "wb") as f:
    pickle.dump(p_emm, f)

with open("data/fitted_correction_factor.pkl", "wb") as f:
    pickle.dump(p_cor, f)

# save lw_vars as netcdf
lw_vars.to_netcdf("data/lw_vars.nc")

# %%
