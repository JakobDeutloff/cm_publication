# %%
import pickle
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# %% load data
path = "/work/bm1183/m301049/icon_arts_processed/fullrange_flux_mid1deg_noice/"

atms = pickle.load(open(path + "atms.pkl", "rb"))
results = pickle.load(open(path + "results.pkl", "rb"))
aux = pickle.load(open(path + "aux.pkl", "rb"))
info = pickle.load(open(path + "info.pkl", "rb"))

# %% convert arts arrays to dict
atms = atms.to_dict()
aux = aux.to_dict()

# %% build results xarray

# Coords
lat = list(results[0]["latitude"][:, 0])
lon = list(results[0]["longitude"][0, :])
pressure_levels = list(atms["grid2"])
freq_lw = info['f_grid_lw']
freq_sw = info['f_grid_sw']

# 3d fluxes
allsky_sw_down = results[2]["allsky_sw"][:, :, :, 0]
allsky_sw_up = results[2]["allsky_sw"][:, :, :, 1]
allsky_lw_down = results[0]["allsky_lw"][:, :, :, 0]
allsky_lw_up = results[0]["allsky_lw"][:, :, :, 1]
clearsky_sw_down = results[3]["clearsky_sw"][:, :, :, 0]
clearsky_sw_up = results[3]["clearsky_sw"][:, :, :, 1]
clearsky_lw_down = results[1]["clearsky_lw"][:, :, :, 0]
clearsky_lw_up = results[1]["clearsky_lw"][:, :, :, 1]

# 2d fluxes 
toa_allsky_sw_down = results[2]["toa_spc_allsky_sw"][:, :, :, 0]
toa_allsky_sw_up = results[2]["toa_spc_allsky_sw"][:, :, :, 1]
toa_allsky_lw_down = results[0]["toa_spc_allsky_lw"][:, :, :, 0]
toa_allsky_lw_up = results[0]["toa_spc_allsky_lw"][:, :, :, 1]
toa_clearsky_sw_down = results[3]["toa_spc_clearsky_sw"][:, :, :, 0]
toa_clearsky_sw_up = results[3]["toa_spc_clearsky_sw"][:, :, :, 1]
toa_clearsky_lw_down = results[1]["toa_spc_clearsky_lw"][:, :, :, 0]
toa_clearsky_lw_up = results[1]["toa_spc_clearsky_lw"][:, :, :, 1]

fluxes_3d = xr.Dataset(
    {
        "allsky_sw_up": (["lat", "lon", "pressure"], allsky_sw_up),
        "allsky_sw_down": (["lat", "lon", "pressure"], allsky_sw_down),
        "allsky_lw_up": (["lat", "lon", "pressure"], allsky_lw_up),
        "allsky_lw_down": (["lat", "lon", "pressure"], allsky_lw_down),
        "clearsky_sw_up": (["lat", "lon", "pressure"], clearsky_sw_up),
        "clearsky_sw_down": (["lat", "lon", "pressure"], clearsky_sw_down),
        "clearsky_lw_up": (["lat", "lon", "pressure"], clearsky_lw_up),
        "clearsky_lw_down": (["lat", "lon", "pressure"], clearsky_lw_down),
        "pressure_levels": (["pressure"], pressure_levels),
    },
    coords={"lat": lat, "lon": lon, "pressure": pressure_levels},
)

fluxes_2d = xr.Dataset(
    {
        "toa_allsky_sw_up": (["lat", "lon", "freq_sw"], toa_allsky_sw_up),
        "toa_allsky_sw_down": (["lat", "lon", "freq_sw"], toa_allsky_sw_down),
        "toa_allsky_lw_up": (["lat", "lon", "freq_lw"], toa_allsky_lw_up),
        "toa_allsky_lw_down": (["lat", "lon", "freq_lw"], toa_allsky_lw_down),
        "toa_clearsky_sw_up": (["lat", "lon", "freq_sw"], toa_clearsky_sw_up),
        "toa_clearsky_sw_down": (["lat", "lon", "freq_sw"], toa_clearsky_sw_down),
        "toa_clearsky_lw_up": (["lat", "lon", "freq_lw"], toa_clearsky_lw_up),
        "toa_clearsky_lw_down": (["lat", "lon", "freq_lw"], toa_clearsky_lw_down),
    }, 
    coords={"lat": lat, "lon": lon, "freq_sw": freq_sw, "freq_lw": freq_lw}
)

# %% build atms xarray
var_names = atms['grid1']

atms_sorted = {}
for i in range(len(var_names)):
    var = atms['grid1'][i]
    atms_sorted[var] = (["pressure", "lat", "lon"], atms['data'][i, :, :, :])

atms_xr = xr.Dataset(
    atms_sorted,
    coords={"lat": lat, "lon": lon, "pressure": pressure_levels},
)

# rename some variables
atms_xr = atms_xr.rename({
    "T": "temperature",
    "z": "geometric height",
    "abs_species-H2O": "H2O",
    "abs_species-O2": "O2",
    "abs_species-N2": "N2",
    "abs_species-CO2": "CO2",
    "abs_species-O3": "O3",
    "scat_species-LWC-mass_density": "LWC",
    "scat_species-IWC-mass_density": "IWC",
    "scat_species-RWC-mass_density": "rain",
    "scat_species-SWC-mass_density": "snow",
    "scat_species-GWC-mass_density": "graupel",
})

# %% build aux xarray
var_names = aux['grid1']

aux_sorted = {}
for i in range(len(var_names)):
    var = aux['grid1'][i]
    aux_sorted[var] = (["lat", "lon"], aux['data'][i, :, :])

aux_xr = xr.Dataset(
    aux_sorted,
    coords={"lat": lat, "lon": lon},
)

# rename some variables
aux_xr = aux_xr.rename(
    {
        "t_surface": "surface temperature",
        "z_surface": "surface height",
        "surface_windspeed": "surface windspeed",
        "land_sea_mask": "land sea mask",
    }
)

# %% save xarrays
fluxes_3d.to_netcdf(path + "fluxes_3d.nc")
fluxes_2d.to_netcdf(path + "fluxes_2d.nc")
atms_xr.to_netcdf(path + "atms.nc")
aux_xr.to_netcdf(path + "aux.nc")

