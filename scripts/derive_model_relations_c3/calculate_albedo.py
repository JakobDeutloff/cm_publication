# %% imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import least_squares

# %% load  data
path = "/work/bm1183/m301049/iwp_framework/ngc3/"
atms = xr.open_dataset(path + "data/atms_proc.nc")
fluxes_allsky = xr.open_dataset(path + "data/fluxes.nc")
fluxes_noice = xr.open_dataset(path + "data/fluxes_nofrozen.nc")
fluxes_noliquid = xr.open_dataset(path + "data/fluxes_noliquid.nc")

# %% initialize data
sw_vars = xr.Dataset()

# %% set mask
mask_parameterisation = atms["mask_height"] & ~atms["mask_low_cloud"]


# %% calculate high cloud albedo
def calc_hc_albedo(a_cs, a_as):
    return (a_as - a_cs) / (a_cs * (a_as - 2) + 1)


sw_vars["noice_albedo"] = np.abs(
    fluxes_noice.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_noice.isel(pressure=-1)["allsky_sw_down"]
)
sw_vars["allsky_albedo"] = np.abs(
    fluxes_allsky.isel(pressure=-1)["allsky_sw_up"]
    / fluxes_allsky.isel(pressure=-1)["allsky_sw_down"]
)
sw_vars["clearsky_albedo"] = np.abs(
    fluxes_allsky.isel(pressure=-1)["clearsky_sw_up"]
    / fluxes_allsky.isel(pressure=-1)["clearsky_sw_down"]
)
cs_albedo = xr.where(atms["connected"] == 1, sw_vars["clearsky_albedo"], sw_vars["noice_albedo"])
sw_vars["high_cloud_albedo"] = calc_hc_albedo(cs_albedo, sw_vars["allsky_albedo"])
sw_vars["high_cloud_albedo"] = xr.where((sw_vars["high_cloud_albedo"] < 0) | (sw_vars['high_cloud_albedo'] > 1), np.nan, sw_vars["high_cloud_albedo"])

# %% calculate mean albedos by weighting with the incoming SW radiation in IWP bins
mean_hc_albedo = (
    sw_vars["high_cloud_albedo"].where(mask_parameterisation) * fluxes_allsky.isel(pressure=-1)["allsky_sw_down"].where(mask_parameterisation)
).sum(["local_time_points", "profile"]) / fluxes_allsky.isel(pressure=-1)["allsky_sw_down"].where(mask_parameterisation).sum(
    ["local_time_points", "profile"]
)
# %% fit logistic function to mean albedo
# prepare x and required y data
x = np.log10(atms['iwp_points'])
y = mean_hc_albedo.copy().values
nan_mask = ~np.isnan(y)
x = x[nan_mask]
y = y[nan_mask]

# initial guess
p0 = [0.75, -1.3, 1.9]


def logistic(params, x):
    return params[0] / (1 + np.exp(-params[2] * (x - params[1])))


def loss(params):
    return logistic(params, x) - y


res = least_squares(loss, p0, xtol=1e-12)
logistic_curve = logistic(res.x, np.log10(atms['iwp_points']))

# %% plot fitted albedo in scatterplot with IWP


fig, ax = plt.subplots()

ax.scatter(
    atms["IWP"].where(mask_parameterisation),
    sw_vars["high_cloud_albedo"].where(mask_parameterisation),
    c=fluxes_allsky["allsky_sw_down"].isel(pressure=-1).where(mask_parameterisation),
    s=0.1
)


ax.plot(mean_hc_albedo.iwp_points, mean_hc_albedo, label="Mean Albedo", color="k")
ax.plot(atms['iwp_points'], logistic_curve, label="Fitted Logistic", color="red", linestyle="--")
ax.legend()
ax.set_xscale("log")
ax.set_ylim(0, 1)

plt.show()

# %% save coefficients as pkl file
path = "/work/bm1183/m301049/iwp_framework/ngc3/"

sw_vars.to_netcdf(path + "data/sw_vars.nc")
mean_hc_albedo.to_netcdf(path + "data/mean_hc_albedo.nc")

with open(path + "parameters/hc_albedo_params.pkl", "wb") as f:
    pickle.dump(res.x, f)

# %%
