# %% import
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import xarray as xr

# %% load freddis data
path = "/work/bm1183/m301049/iwp_framework/ngc3/"
atms = xr.open_dataset(path + "data/atms_proc.nc")
fluxes_allsky = xr.open_dataset(path + "data/fluxes.nc")
ds_c3 = xr.open_dataset(path + 'data/representative_sample_c3_conn3.nc')

# %% initialize dataset for new variables
lw_vars = xr.Dataset()

# %% set mask
mask_parameterisation = atms["mask_height"] & ~atms["mask_low_cloud"]

# %% calculate brightness temp from fluxes
def calc_brightness_temp(flux):
    return (-flux / 5.67e-8) ** (1 / 4)


# only look at clouds with cloud tops above 350 hPa and IWP > 1e-1 so that e = 1 can be assumed
mask_bright = atms["mask_height"] & (atms["IWP"] > 1e-1) 

flux = fluxes_allsky["allsky_lw_up"].isel(pressure=-1)
T_bright = calc_brightness_temp(flux)
T_bright = T_bright.where(mask_bright, 500)

# tropopause pressure
level_trop_ixd = atms["temperature"].argmin("level_full")
p_trop = atms["pressure"].isel(level_full=level_trop_ixd)

# get pressure levels and ice_cumsum where T == T_bright in Troposphere
T_profile = atms["temperature"].where(atms["pressure"] > p_trop, 0)
level_bright_idx = np.abs(T_profile - T_bright).argmin("level_full")  # fills with 0 for NaNs
ice_cumsum = atms["IWC_cumsum"].isel(level_full=level_bright_idx)
mean_ice_cumsum = ice_cumsum.where(mask_bright).median()
print(mean_ice_cumsum.values)

# %% calculate cloud top temperature 
from src.calc_variables import calculate_h_cloud_temperature
temp, pressure = calculate_h_cloud_temperature(
    atms, fluxes_allsky, convention='icon', IWP_emission=3.3e-3, option="emission"
)
atms["hc_top_temperature"], atms["hc_top_pressure"] = temp, pressure

# %% calculate high cloud emissivity
sigma = 5.67e-8  # W m-2 K-4
LW_out_as = fluxes_allsky.isel(pressure=-1)["allsky_lw_up"]
LW_out_cs = fluxes_allsky.isel(pressure=-1)["clearsky_lw_up"]
rad_hc = -atms["hc_top_temperature"] ** 4 * sigma
hc_emissivity = (LW_out_as - LW_out_cs) / (rad_hc - LW_out_cs)
hc_emissivity = xr.where((hc_emissivity < -0.1) | (hc_emissivity > 1.5), np.nan, hc_emissivity)
lw_vars["high_cloud_emissivity"] = hc_emissivity

# %% fit logistic function to mean high cloud emissivity
IWP_bins = np.logspace(-6, np.log10(30) , 51)

# prepare x and required y data
x = np.log10(atms['iwp_points'])
mean_emissivity = lw_vars["high_cloud_emissivity"].where(mask_parameterisation).mean(["local_time_points", "profile"])
y = mean_emissivity.values
nan_mask = ~np.isnan(y)
x = x[nan_mask]
y = y[nan_mask]

# prepare weights
n_cells = len(ds_c3.time) * len(ds_c3.cell)
hist, edges = np.histogram(ds_c3["IWP"].where(ds_c3["mask_height"]), bins=IWP_bins)
hist = hist / n_cells
hist = hist[nan_mask]

#initial guess
p0 = [-2.2591527 ,  3.19284716]

def logistic(params, x):
    return 1 / (1 + np.exp(-params[1] * (x - params[0])))


def loss(params):
    return ((logistic(params, x) - y) * hist) / hist.sum()

res = least_squares(loss, p0, xtol=1e-10)
logistic_curve = logistic(res.x, np.log10(atms['iwp_points']))

# %% plot mean hv emissivity in scatterplot with IWP
fig, ax = plt.subplots()
ax.scatter(atms['IWP'].where(mask_parameterisation), lw_vars["high_cloud_emissivity"].where(mask_parameterisation), s=1)
ax.plot(atms['iwp_points'], logistic_curve, color="r", label="Fitted logistic", linestyle="--")
ax.axhline(1, color="grey", linestyle="--")
ax.plot(10**x, y, color='red')
ax.set_xscale('log')
ax.legend()

# %% save coefficients as pkl file
path = "/work/bm1183/m301049/iwp_framework/ngc3/"

lw_vars.to_netcdf(path + "data/lw_vars.nc")
mean_emissivity.to_netcdf(path + "data/mean_hc_emissivity.nc")

with open(path + "parameters/hc_emissivity_params.pkl", "wb") as f:
    pickle.dump(np.array([1., res.x[0], res.x[1]]), f)

# %%
