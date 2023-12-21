# %% import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle
import pandas as pd
from src.read_data import load_atms_and_fluxes
from src.plot_functions import scatterplot
from scipy.optimize import curve_fit

# %% load freddis data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()

# %% initialize dataset for new variables
lw_vars = xr.Dataset()
mean_lw_vars = pd.DataFrame()

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

# %% find profiles with high clouds and no thick graupel layers below above 350 hPa
mask_hc_no_lc = (atms["IWP"] > 1e-7) & (atms["LWP"] < 1e-7)
mask_height = p_top < 35000
lw_vars["mask_height"] = mask_height
lw_vars["mask_hc_no_lc"] = mask_hc_no_lc

# %% calculate high cloud emissivity
sigma = 5.67e-8  # W m-2 K-4
LW_out_as = fluxes_3d.isel(pressure=-1)["allsky_lw_up"]
LW_out_cs = fluxes_3d_noice.isel(pressure=-1)["allsky_lw_up"]
rad_hc = -lw_vars["h_cloud_temperature"] ** 4 * sigma
hc_emissivity = (LW_out_as - LW_out_cs) / (rad_hc - LW_out_cs)
lw_vars["high_cloud_emissivity"] = hc_emissivity


# %% aveage over IWP bins
def cut_data(data, mask):
    return data.where(mask).sel(lat=slice(-30, 30))


IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2
mean_hc_emissivity = (
    cut_data(lw_vars["high_cloud_emissivity"], mask_height & mask_hc_no_lc)
    .groupby_bins(
        cut_data(atms["IWP"], mask_height & mask_hc_no_lc), IWP_bins, labels=IWP_points
    )
    .mean()
)

mean_lw_vars.index = IWP_points
mean_lw_vars.index.name = "IWP"
mean_lw_vars["binned_emissivity"] = mean_hc_emissivity


# %% fit logistic function to mean high cloud emissivity
def logistic(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))


x = np.log10(IWP_points)
y = mean_lw_vars["binned_emissivity"]
y[IWP_points > 1e-1] = 1
nan_mask = ~np.isnan(y)
x = x[nan_mask]
y = y[nan_mask]

popt, pcov = curve_fit(logistic, x, y)
popt[0] = 1
logistic_curve = logistic(np.log10(IWP_points), *popt)

# %% plot mean hv emissivity in scatterplot with IWP

fig, ax = scatterplot(
    cut_data(atms["IWP"], mask_height & mask_hc_no_lc),
    cut_data(lw_vars["high_cloud_emissivity"], mask_height & mask_hc_no_lc),
    cut_data(
        fluxes_3d_noice.isel(pressure=-1)["clearsky_sw_down"],
        mask_height & mask_hc_no_lc,
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
fig.savefig("plots/emissivity.png", dpi=300, bbox_inches="tight")

# %% save coefficients as pkl file
path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"

lw_vars.to_netcdf(path + "lw_vars.nc")

with open(path + "mean_lw_vars.pkl", "wb") as f:
    pickle.dump(mean_lw_vars, f)

with open(path + "/hc_emissivity_params.pkl", "wb") as f:
    pickle.dump(popt, f)


# %%
