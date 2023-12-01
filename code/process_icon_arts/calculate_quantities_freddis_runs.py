# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

# %% load data from freddis runs
path_freddi = "/work/bm1183/m301049/icon_arts_processed/"
run = "fullrange_flux_mid1deg_noice/"
atms = xr.open_dataset(path_freddi + run + "atms.nc")
fluxes_3d = xr.open_dataset(path_freddi + run + "fluxes_3d.nc")
aux = xr.open_dataset(path_freddi + run + "aux.nc")

# %% change convention of fluxes - down is positive
fluxes = [
    "allsky_sw_down",
    "allsky_sw_up",
    "allsky_lw_down",
    "allsky_lw_up",
    "clearsky_sw_down",
    "clearsky_sw_up",
    "clearsky_lw_down",
    "clearsky_lw_up",
]
for flux in fluxes:
    fluxes_3d[flux] = fluxes_3d[flux] * -1

# %% calculate IWP and LWP
cell_height = atms["geometric height"].diff(
    "pressure"
)  # not correct, we would need height ad half levels
atms["IWP"] = ((atms["IWC"] + atms["snow"] + atms["graupel"]) * cell_height).sum(
    "pressure"
)
atms["LWP"] = ((atms["rain"] + atms["LWC"]) * cell_height).sum("pressure")

# %% calculate heating rates from fluxes (vertical levels are not quite correct)
g = 9.81
cp = 1005
seconds_per_day = 24 * 60 * 60
p = fluxes_3d["pressure"]
p_half = (p[1:].values + p[:-1].values) / 2
fluxes_3d = fluxes_3d.assign_coords(p_half=p_half)

allsky_hr_lw = (
    (g / cp)
    * (
        (fluxes_3d["allsky_lw_up"] + fluxes_3d["allsky_lw_down"]).diff("pressure")
        / fluxes_3d["pressure"].diff("pressure")
    )
    * seconds_per_day
)
allsky_hr_lw["pressure"] = p_half
allsky_hr_lw = allsky_hr_lw.rename({"pressure": "p_half"})
fluxes_3d["allsky_hr_lw"] = allsky_hr_lw

clearsky_hr_lw = (
    (g / cp)
    * (
        (fluxes_3d["clearsky_lw_up"] + fluxes_3d["clearsky_lw_down"]).diff("pressure")
        / fluxes_3d["pressure"].diff("pressure")
    )
    * seconds_per_day
)
clearsky_hr_lw["pressure"] = p_half
clearsky_hr_lw = clearsky_hr_lw.rename({"pressure": "p_half"})
fluxes_3d["clearsky_hr_lw"] = clearsky_hr_lw

allsky_hr_sw = (
    (g / cp)
    * (
        (fluxes_3d["allsky_sw_up"] + fluxes_3d["allsky_sw_down"]).diff("pressure")
        / fluxes_3d["pressure"].diff("pressure")
    )
    * seconds_per_day
)
allsky_hr_sw["pressure"] = p_half
allsky_hr_sw = allsky_hr_sw.rename({"pressure": "p_half"})
fluxes_3d["allsky_hr_sw"] = allsky_hr_sw

clearsky_hr_sw = (
    (g / cp)
    * (
        (fluxes_3d["clearsky_sw_up"] + fluxes_3d["clearsky_sw_down"]).diff("pressure")
        / fluxes_3d["pressure"].diff("pressure")
    )
    * seconds_per_day
)
clearsky_hr_sw["pressure"] = p_half
clearsky_hr_sw = clearsky_hr_sw.rename({"pressure": "p_half"})
fluxes_3d["clearsky_hr_sw"] = clearsky_hr_sw

# %% reduction of outgoing longwave radiation from clearsky profile
clearsky_lw_out = fluxes_3d["clearsky_lw_up"].mean(["lat", "lon"])
reduction_fraction = (
    clearsky_lw_out - clearsky_lw_out.isel(pressure=-1)
) / clearsky_lw_out

fig, ax = plt.subplots()
ax.plot(reduction_fraction, clearsky_lw_out.pressure)
ax.invert_yaxis()

# %% Exclude clouds below 8 km
min_height = 8e3  # m
p_iwc_max = atms["IWC"].argmax("pressure")
p_iwc_max = p_iwc_max.fillna(80)
height_mask = atms["geometric height"].isel(pressure=p_iwc_max) > min_height

# %% calculate high cloud temperature from LW out differences
diff_at_cloud_top = 0.9  # fraction of LW out difference at cloud top compared to toa
lw_out_diff = np.abs(fluxes_3d["allsky_lw_up"] - fluxes_3d["clearsky_lw_up"])
lw_out_diff_frac = lw_out_diff / lw_out_diff.isel(pressure=-1)
bool_lw_out = lw_out_diff_frac < diff_at_cloud_top

# find lowest pressure where bool_lw_out is true
p_top = bool_lw_out.pressure.where(bool_lw_out).min("pressure")
p_top = p_top.fillna(atms.pressure.max())
T_h_lw = atms["temperature"].sel(pressure=p_top).where(height_mask)
atms["h_cloud_temperature"] = T_h_lw 
atms["h_cloud_top_pressure"] = p_top.where(height_mask)

# %% calculate high cloud emissivity
idx_p_tropopause = atms['temperature'].sel(lat=slice(-30, 30)).mean(['lat', 'lon']).argmin('pressure')
p_tropopause = atms['pressure'].isel(pressure=idx_p_tropopause)
sigma = 5.67e-8  # W m-2 K-4
LW_out_as = fluxes_3d.isel(pressure=-1)["allsky_lw_up"]
LW_out_cs = fluxes_3d.isel(pressure=-1)["clearsky_lw_up"]
correction = LW_out_as - fluxes_3d.sel(pressure=atms["h_cloud_top_pressure"], method='nearest')["allsky_lw_up"]
reduction = 1  #- reduction_fraction.sel(pressure=p_top) # reduction of LW out due to clearsky profile
rad_hc = -atms["h_cloud_temperature"] ** 4 * sigma 
atms["high_cloud_emissivity"] = (LW_out_as - correction - LW_out_cs) / (rad_hc - LW_out_cs)
atms['rad_correction'] = correction

# %% plot emissivity against IWP
mask_hc_no_lc = (atms["IWP"] > 1e-6) & (atms["LWP"] < 1e-10)

fig, ax = plt.subplots(figsize=(7, 5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sc = ax.scatter(
    atms["IWP"].where(mask_hc_no_lc & height_mask).sel(lat=slice(-30, 30)),
    atms["high_cloud_emissivity"]
    .where(mask_hc_no_lc & height_mask)
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

sc = ax.scatter(atms["IWP"].where(mask_hc_no_lc & height_mask).sel(lat=slice(-30, 30)),
                correction.where(mask_hc_no_lc & height_mask).sel(lat=slice(-30, 30)))
ax.set_xscale("log")

# %% calculate high cloud albedo
atms["clearsky_albedo"] = np.abs(
    fluxes_3d.isel(pressure=-1)["clearsky_sw_up"]
    / fluxes_3d.isel(pressure=-1)["clearsky_sw_down"]
)
atms["allsky_albedo"] = np.abs(
    fluxes_3d.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_3d.isel(pressure=-1)["allsky_sw_down"]
)
atms["high_cloud_albedo"] = (atms["allsky_albedo"] - atms["clearsky_albedo"]) / (
    1 - atms["clearsky_albedo"]
)

# %% calculate high cloud radiative effect
fluxes_toa = fluxes_3d.isel(pressure=-1)
atms["cloud_rad_effect"] = (
    fluxes_toa["allsky_sw_down"]
    + fluxes_toa["allsky_sw_up"]
    + fluxes_toa["allsky_lw_down"]
    + fluxes_toa["allsky_lw_up"]
    - (fluxes_toa["clearsky_sw_down"] + fluxes_toa["clearsky_sw_up"])
    - (fluxes_toa["clearsky_lw_down"] + fluxes_toa["clearsky_lw_up"])
)

atms["sw_cloud_rad_effect"] = (
    fluxes_toa["allsky_sw_down"]
    + fluxes_toa["allsky_sw_up"]
    - (fluxes_toa["clearsky_sw_down"] + fluxes_toa["clearsky_sw_up"])
)

atms["lw_cloud_rad_effect"] = (
    fluxes_toa["allsky_lw_down"]
    + fluxes_toa["allsky_lw_up"]
    - (fluxes_toa["clearsky_lw_down"] + fluxes_toa["clearsky_lw_up"])
)

# %% save results
atms.to_netcdf(path_freddi + "atms_full.nc")
fluxes_3d.to_netcdf(path_freddi + "fluxes_3d_full.nc")

# %%
