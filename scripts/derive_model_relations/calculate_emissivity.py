# %% import
import numpy as np
import xarray as xr
import pickle
import pandas as pd
from src.read_data import load_atms_and_fluxes
from src.plot_functions import scatterplot
from src.helper_functions import cut_data
from scipy.optimize import least_squares
import xarray as xr
import os

# %% load freddis data
atms, fluxes_allsky, fluxes_noice = load_atms_and_fluxes()
ds_monsoon = xr.open_dataset("/work/bm1183/m301049/iwp_framework/mons/data/full_snapshot_proc.nc")


# %% initialize dataset for new variables
lw_vars = xr.Dataset()
mean_lw_vars = pd.DataFrame()

# %% set mask
mask_parameterisation = atms["mask_height"] & ~atms["mask_low_cloud"]

# %% calculate high cloud emissivity
sigma = 5.67e-8  # W m-2 K-4
LW_out_as = fluxes_allsky.isel(pressure=-1)["allsky_lw_up"]
LW_out_cs = fluxes_noice.isel(pressure=-1)["clearsky_lw_up"]
rad_hc = -atms["hc_top_temperature"] ** 4 * sigma
hc_emissivity = (LW_out_as - LW_out_cs) / (rad_hc - LW_out_cs)
hc_emissivity = xr.where((hc_emissivity < -0.1) | (hc_emissivity > 1.5), np.nan, hc_emissivity)
lw_vars["high_cloud_emissivity"] = hc_emissivity


# %% aveage over IWP bins
IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
mean_hc_emissivity = (
    cut_data(lw_vars["high_cloud_emissivity"], mask_parameterisation)
    .groupby_bins(
        cut_data(atms["IWP"], mask_parameterisation),
        IWP_bins,
        labels=IWP_points,
    )
    .mean()
)

mean_lw_vars.index = IWP_points
mean_lw_vars.index.name = "IWP"
mean_lw_vars["binned_emissivity"] = mean_hc_emissivity


# %% fit logistic function to mean high cloud emissivity

# prepare x and required y data
x = np.log10(IWP_points)
y = mean_lw_vars["binned_emissivity"].copy()
nan_mask = ~np.isnan(y)
x = x[nan_mask]
y = y[nan_mask]

#initial guess
p0 = [-2.27406137, 3.25181808]

def logistic(params, x):
    return 1 / (1 + np.exp(-params[1] * (x - params[0])))


def loss(params):
    return (logistic(params, x) - y) 

res = least_squares(loss, p0)
logistic_curve = logistic(res.x, np.log10(IWP_points))

# %% plot mean hv emissivity in scatterplot with IWP
fig, ax = scatterplot(
    cut_data(atms["IWP"], mask_parameterisation),
    cut_data(lw_vars["high_cloud_emissivity"], mask_parameterisation),
    cut_data(
        fluxes_noice.isel(pressure=-1)["clearsky_sw_down"],
        mask_parameterisation,
    ),
    xlabel="IWP / kg m$^{-2}$",
    ylabel="High Cloud Emissivity",
    cbar_label="SW Down / W m$^{-2}$",
    xlim=[1e-5, 1e1],
    ylim=[-0.2, 1.5],
)

ax.plot(IWP_points, mean_hc_emissivity, color="lime", label="Mean Emissivity")
ax.plot(IWP_points, logistic_curve, color="r", label="Fitted logistic", linestyle="--")
ax.axhline(1, color="grey", linestyle="--")
ax.legend()

# %% save coefficients as pkl file
path = "/work/bm1183/m301049/iwp_framework/mons/"

os.remove(path + "data/lw_vars.nc")
os.remove(path + "parameters/hc_emissivity_params.pkl")
os.remove(path + "data/lw_vars_mean.pkl")

lw_vars.to_netcdf(path + "data/lw_vars.nc")
with open(path + "parameters/hc_emissivity_params.pkl", "wb") as f:
    pickle.dump(np.array([1., res.x[0], res.x[1]]), f)
with open(path + "data/lw_vars_mean.pkl", "wb") as f:
    pickle.dump(mean_lw_vars, f)


# %%
