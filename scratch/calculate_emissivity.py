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
run = "fullrange_flux_mid1deg_noice/"
fluxes_3d_noice = xr.open_dataset(path + run + "fluxes_3d_full.nc")

# %% find profiles with high clouds and no low clouds below and above 8 km
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)
idx_height = atms["IWC"].argmax("pressure")
mask_height = atms["geometric height"].isel(pressure=idx_height) >= 8e3

# %% calculate high cloud temperature from LW out differences
diff_at_cloud_top = 0.9  # fraction of LW out difference at cloud top compared to toa
lw_out_diff = np.abs(fluxes_3d["allsky_lw_up"] - fluxes_3d["clearsky_lw_up"])
lw_out_diff_frac = lw_out_diff / lw_out_diff.isel(pressure=-1)
bool_lw_out = lw_out_diff_frac < diff_at_cloud_top

# find lowest pressure where bool_lw_out is true
p_top = bool_lw_out.pressure.where(bool_lw_out).min("pressure")
p_top = p_top.fillna(atms.pressure.max())
T_h_lw = atms["temperature"].sel(pressure=p_top).where(mask_height)
atms["h_cloud_temperature"] = T_h_lw 
atms["h_cloud_top_pressure"] = p_top.where(mask_height)

# %% calculate high cloud emissivity
idx_p_tropopause = atms['temperature'].sel(lat=slice(-30, 30)).mean(['lat', 'lon']).argmin('pressure')
p_tropopause = atms['pressure'].isel(pressure=idx_p_tropopause)
sigma = 5.67e-8  # W m-2 K-4
LW_out_as = fluxes_3d.isel(pressure=-1)["allsky_lw_up"]
LW_out_cs = fluxes_3d.isel(pressure=-1)["clearsky_lw_up"]
correction = LW_out_as - fluxes_3d.sel(pressure=atms["h_cloud_top_pressure"], method='nearest')["allsky_lw_up"]
rad_hc = -atms["h_cloud_temperature"] ** 4 * sigma 
atms["high_cloud_emissivity"] = (LW_out_as - correction - LW_out_cs) / (rad_hc - LW_out_cs)
atms['rad_correction'] = correction

# %% plot emissivity against IWP
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)

fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"].where(mask_hc_no_lc & mask_height).sel(lat=slice(-30, 30)),
    atms["high_cloud_emissivity"]
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d["allsky_sw_down"]
    .isel(pressure=-1)
    .where(mask_hc_no_lc)
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)
cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("High Cloud Emissivity")
ax.set_xscale("log")
ax.set_ylim(0, 1.5)
fig.savefig("plots/high_cloud_emissivity_vs_iwp.png", dpi=300)

# %% plot correction vs IWP
fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(atms["IWP"].where(mask_hc_no_lc & mask_height).sel(lat=slice(-30, 30)),
                correction.where(mask_hc_no_lc & mask_height).sel(lat=slice(-30, 30)))
ax.set_xscale("log")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("C / W m$^{-2}$")

# %% aveage over IWP bins 
IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
mean_hc_emissivity = np.zeros(len(IWP_bins) - 1)
mean_correction_factor = np.zeros(len(IWP_bins) - 1)

for i in range(len(IWP_bins) - 1):
    IWP_mask = (atms["IWP"] > IWP_bins[i]) & (atms["IWP"] < IWP_bins[i + 1])
    mean_hc_emissivity[i] = float(
        (
            atms["high_cloud_emissivity"]
            .where(IWP_mask & mask_hc_no_lc & mask_height)
            .sel(lat=slice(-30, 30))
        )
        .mean()
        .values
    )
    mean_correction_factor[i] = float(
        (
            atms["rad_correction"]
            .where(IWP_mask & mask_hc_no_lc & mask_height)
            .sel(lat=slice(-30, 30))
            .mean()
            .values
        )
    )

# %% fit polynomial to mean emissivity
p_emm = np.polyfit(np.log10(IWP_points[mean_hc_emissivity <= 1]), mean_hc_emissivity[mean_hc_emissivity <= 1], 9)

def hc_emissivity(IWP, coeffs): 
    fitted_vals = np.poly1d(coeffs)(np.log10(IWP))
    return fitted_vals

fitted_emissivity = hc_emissivity(IWP_points, p_emm)

# %% fit polynomial to mean correction factor
p_cor = np.polyfit(np.log10(IWP_points[~np.isnan(mean_correction_factor)]), mean_correction_factor[~np.isnan(mean_correction_factor)], 9)

def hc_correction_factor(IWP, coeffs):
    fitted_vals = np.poly1d(coeffs)(np.log10(IWP))
    return fitted_vals

fitted_correction_factor = hc_correction_factor(IWP_points, p_cor)

# %% plot mean hv emissivity in scatterplot with IWP
fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"]
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30)),
    atms["high_cloud_emissivity"]
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30)),
    s=0.5,
    c=fluxes_3d["allsky_sw_down"]
    .isel(pressure=-1)
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30)),
    cmap="viridis",
)

ax.plot(IWP_points, mean_hc_emissivity, color="lime", label="Mean Emissivity")
ax.plot(IWP_points, fitted_emissivity, color="r", label="Fitted Polynomial", linestyle='--')
ax.axhline(1, color="grey", linestyle='--')

cb = fig.colorbar(sc)
cb.set_label("SWin at TOA / W m$^{-2}$")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("High Cloud Emissivity")
ax.set_xscale("log")
ax.set_ylim([0, 1.7])
ax.set_xlim(1e-5, 10)
ax.legend()

# %% plot mean correction factior in scatterplot with IWP
fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"]
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30)),
    atms["rad_correction"]
    .where(mask_hc_no_lc & mask_height)
    .sel(lat=slice(-30, 30)),
    s=0.5,
    color='k'
)

ax.plot(IWP_points, mean_correction_factor, color="lime", label="Mean Correction Factor")
ax.plot(IWP_points, fitted_correction_factor, color="r", label="Fitted Polynomial", linestyle='--')

ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_ylabel("Correction Factor")
ax.set_xscale("log")
ax.legend()

# %% save coefficients as pkl file
with open("data/fitted_emissivity.pkl", "wb") as f:
    pickle.dump(p_emm, f)

with open("data/fitted_correction_factor.pkl", "wb") as f:
    pickle.dump(p_cor, f)

# %%
