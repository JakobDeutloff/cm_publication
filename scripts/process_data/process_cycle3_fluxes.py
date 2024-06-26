# %% import
import xarray as xr
import matplotlib.pyplot as plt
import os
from src.read_data import load_atms_and_fluxes

# %% load old atms data 
atms_old, flux_old, fluxe_noice_old = load_atms_and_fluxes()


# %% read data
path = "/work/um0878/users/mbrath/StarARTS/results/processed_fluxes/flux_iconc3_Nf10000"
runs = ["", "_noliquid", "_nofrozen"]
file = "fluxes_flux_iconc3_Nf10000"

fluxes_raw = {}
fluxes = {}
for run in runs:
    fluxes_raw[run] = xr.open_dataset(f"{path}{run}/{file}{run}.nc")

path = "/work/bm1183/m301049/iwp_framework/ngc3/raw_data/"
atms_unstructured = xr.open_dataset(path + "unstructured/atms_unstructured.nc")
atms = xr.open_dataset(path + "atms.nc")
aux_unstr = xr.open_dataset(path + "unstructured/aux_unstructured.nc")
aux = xr.open_dataset(path + "aux.nc")

# %% assign correct local time to aux
aux = aux.assign(time_local=lambda d: d.time.dt.hour + d.lon / 15)
aux["time_local"] = aux["time_local"].where(aux["time_local"] < 24, aux["time_local"] - 24)
aux_unstr = aux_unstr.assign(time_local=lambda d: d.time.dt.hour + d.lon / 15)
aux_unstr["time_local"] = aux_unstr["time_local"].where(aux_unstr["time_local"] < 24, aux_unstr["time_local"] - 24)

# %% scatterplot of t_surf and LW out at surface
fig, ax = plt.subplots()
ax.scatter(
    fluxes_raw[""]["auxiliary_variables"].sel(quantities="t_surface"),
    fluxes_raw[""]['allsky_thermal'].sel(direction='upward').isel(pressure=0),
)

# %% restructure fluxes dataset to match atms dimensions and coordinates
for run in runs:
    flux = fluxes_raw[run]
    flux["iwp_points"] = atms_unstructured["iwp_points"].drop_vars(["profile", 'local_time_points'])
    flux["local_time_points"] = atms_unstructured["local_time_points"].drop_vars(["iwp_points", 'profile'])
    flux["profile"] = atms_unstructured["profile"].drop_vars(["iwp_points", "local_time_points"])
    flux['ICW'] = atms_unstructured['IWC'].drop_vars(["level_full", "iwp_points", "local_time_points", "profile"])
    # Recreate the MultiIndex
    flux_stacked = flux.set_index(index=["iwp_points", "local_time_points", "profile"])
    # Unstack the MultiIndex
    flux_unstacked = flux_stacked.unstack("index")
    flux_unstacked = flux_unstacked.drop_vars(["time", "cell", "lat", "lon"])
    # restore old key convention
    flux_unstacked["allsky_sw_up"] = flux_unstacked["allsky_solar"].sel(direction="upward")
    flux_unstacked["allsky_sw_down"] = flux_unstacked["allsky_solar"].sel(direction="downward")
    flux_unstacked["clearsky_sw_up"] = flux_unstacked["clearsky_solar"].sel(direction="upward")
    flux_unstacked["clearsky_sw_down"] = flux_unstacked["clearsky_solar"].sel(direction="downward")
    flux_unstacked["allsky_lw_up"] = flux_unstacked["allsky_thermal"].sel(direction="upward")
    flux_unstacked["allsky_lw_down"] = flux_unstacked["allsky_thermal"].sel(direction="downward")
    flux_unstacked["clearsky_lw_up"] = flux_unstacked["clearsky_thermal"].sel(direction="upward")
    flux_unstacked["clearsky_lw_down"] = flux_unstacked["clearsky_thermal"].sel(
        direction="downward"
    )
    flux_unstacked = flux_unstacked.drop_vars("allsky_solar")
    flux_unstacked = flux_unstacked.drop_vars("clearsky_solar")
    flux_unstacked = flux_unstacked.drop_vars("allsky_thermal")
    flux_unstacked = flux_unstacked.drop_vars("clearsky_thermal")
    flux_unstacked = flux_unstacked.drop_vars("direction")
    fluxes[run] = flux_unstacked

    # save to netcdf
    path = "/work/bm1183/m301049/iwp_framework/ngc3/data/"
    flux_unstacked.to_netcdf(f"{path}fluxes{run}.nc")

# %% testplot of cre in allsky fluxes
flux = fluxes['_nofrozen'].isel(pressure=-1)
cre_sw = (flux["allsky_sw_down"] - flux["allsky_sw_up"]) - (
    flux["clearsky_sw_down"] - flux["clearsky_sw_up"]
)
cre_lw = (flux["allsky_lw_down"] - flux["allsky_lw_up"]) - (
    flux["clearsky_lw_down"] - flux["clearsky_lw_up"]
)
cre_sw = cre_sw.mean(["profile", "local_time_points"])
cre_lw = cre_lw.mean(["profile", "local_time_points"])
fig, ax = plt.subplots()
ax.plot(aux.iwp_points, cre_sw, label="sw")
ax.plot(aux.iwp_points, cre_lw, label="lw")
ax.legend()
ax.set_xscale("log")
ax.set_ylabel("CRE / W m-2")
ax.set_xlabel("IWP / kg m-2")

# %% test if IWP relation is the way it should be
fig, ax = plt.subplots()
ax.scatter(
    atms_unstructured["iwp_points"],
    (atms_unstructured["IWC"] + atms_unstructured["snow"] + atms_unstructured["graupel"]).sum(
        "level_full"
    ),
)
ax.set_xscale("log")
ax.set_yscale("log")

# %% test if binning is correct and if the data is the same
fig, axes = plt.subplots(2, 2)

fluxes['']['auxiliary_variables'].sel(quantities="t_surface").mean("profile").plot(ax=axes[0, 0])
aux["surface_temperature"].mean("profile").plot(ax=axes[0, 1])

(-1* fluxes['']['allsky_sw_down']).isel(pressure=-1).mean("profile").plot(ax=axes[1, 0])
aux["rsdt"].mean("profile").plot(ax=axes[1, 1])

for ax in axes.flatten():
    ax.set_yscale("log")

fig.tight_layout()

# %% plot distribution of incoming solar radiation
fig, ax = plt.subplots()
fluxes['']['allsky_sw_down'].isel(pressure=-1).plot.hist(bins=100, ax=ax, label='ARTS')
(-1 * aux["rsdt"]).plot.hist(bins=100, ax=ax, alpha=0.5, label='ICON')
ax.set_ylim(0, 1e3)
ax.legend()
ax.set_ylabel("Frequency")
ax.set_xlabel("Incoming solar radiation / W m-2")

# %% scatter IWP sum vs outgoing SW in noliquid run
fig, ax = plt.subplots()
ax.scatter(
    fluxes[""]["ICW"].sum("level_full"),
    fluxes[""]["allsky_lw_up"].isel(pressure=-1),
)
ax.set_xscale("log")
# %% calculate Mean CRE
cre_lw = {}
cre_sw = {}
for run in runs: 
    flux = fluxes[run].isel(pressure=-1)
    cre_sw[run] = (flux["allsky_sw_down"] - flux["allsky_sw_up"]) - (
        flux["clearsky_sw_down"] - flux["clearsky_sw_up"]
    )
    cre_lw[run] = (flux["allsky_lw_down"] - flux["allsky_lw_up"]) - (
        flux["clearsky_lw_down"] - flux["clearsky_lw_up"]
    )
    cre_sw[run] = cre_sw[run].mean()
    cre_lw[run] = cre_lw[run].mean()
    print(f"CRE SW {run}: {cre_sw[run].values}")
    print(f"CRE LW {run}: {cre_lw[run].values}")

# %% calculate Mean CRE from Raw fluxes
cre_lw = {}
cre_sw = {}
for run in runs: 
    flux = fluxes_raw[run].isel(pressure=-1)
    cre_sw[run] = (flux["allsky_solar"].sel(direction="downward") - flux["allsky_solar"].sel(direction="upward")) - (
        flux["clearsky_solar"].sel(direction="downward") - flux["clearsky_solar"].sel(direction="upward")
    )
    cre_lw[run] = (flux["allsky_thermal"].sel(direction="downward") - flux["allsky_thermal"].sel(direction="upward")) - (
        flux["clearsky_thermal"].sel(direction="downward") - flux["clearsky_thermal"].sel(direction="upward")
    )
    cre_sw[run] = cre_sw[run].mean()
    cre_lw[run] = cre_lw[run].mean()
    print(f"CRE SW {run}: {cre_sw[run].values}")
    print(f"CRE LW {run}: {cre_lw[run].values}")

# %% pick position and plot 
fig, ax = plt.subplots()
profile = fluxes['']['allsky_lw_up'].isel(iwp_points=40, local_time_points=6, profile=20)
profiel_clearsky = fluxes['']['clearsky_lw_up'].isel(iwp_points=40, local_time_points=6, profile=20)
ax.plot(profile.values, profile.pressure.values, label='allsky')
ax.plot(profiel_clearsky.values, profiel_clearsky.pressure.values, label='clearsky')
ax.invert_yaxis()
ax.set_yscale('log')
ax.legend()

# %% plot hist of latitudes in aux
fig, ax = plt.subplots()
aux.lat.plot.hist(ax=ax, bins=100)

# %% 
fig, ax = plt.subplots()
fluxes['']['auxiliary_variables'].sel(quantities="sun_zenith_longitude").plot.hist(ax=ax, bins=100)

# %% scatter aux time vs rsdt 
fig, ax = plt.subplots()
ax.scatter(aux.time_local, aux.rsdt, s=0.1)
ax.scatter(aux.time_local, -fluxes['']['allsky_sw_down'].isel(pressure=-1), s=0.1)
ax.axvline(12)



# %% plot longitude
fig, ax  = plt.subplots()
data_freddi = fluxes['']['auxiliary_variables'].sel(quantities='longitude').mean("profile").values
data_aux = aux.lon.values.flatten()
ax.hist(data_aux)

# %% scatter SW radiation at 0 UTC around 0 lon 
fig, ax = plt.subplots()
rsdt = aux.where((aux.lon<5)).rsdt
ax.scatter(rsdt.time.dt.hour, rsdt.values, s=0.5)
ax.axvline(12, alpha=0.5, color='red')
# %%
