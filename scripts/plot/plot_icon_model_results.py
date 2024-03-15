# %% import 
from src.plot_functions import plot_model_output_icon, plot_connectedness, plot_sum_cre
import pickle
import xarray as xr
import numpy as np
import os

# %% load data
path = '/work/bm1183/m301049/cm_results/'
run = "icon_c3_zero_lc"

with open(path + run + '.pkl', 'rb') as f:
    result = pickle.load(f)

# sample = xr.open_dataset(f"/work/bm1183/m301049/nextgems_profiles/cycle4/representative_sample_c4_conn3.nc")
sample = xr.open_dataset(f"/work/bm1183/m301049/nextgems_profiles/cycle3/representative_sample_c3_conn3.nc")

# %% create folder
if not os.path.exists("plots/model_results/" + run):
    os.mkdir("plots/model_results/" + run)

# %% plot results
iwp_bins = np.logspace(-5, np.log10(50), num=60)
fig, axes = plot_model_output_icon(result, iwp_bins , sample.where(sample['mask_height']))
fig.savefig(f"plots/model_results/{run}/results.png", dpi=300, bbox_inches="tight")

# %% Fold Distribution with CRE
fig, axes = plot_sum_cre(result, sample, iwp_bins)
fig.savefig(f'plots/model_results/{run}/cre_integrated.png', dpi=300, bbox_inches='tight')

# %%  plot connectedness 
liq_cld_cond = sample["LWC"] + sample["rain"]
ice_cld_cond = sample["IWC"] + sample["snow"] + sample["graupel"]
mask = sample["mask_height"] & (sample["IWP"] > 1e-6) & (sample["LWP"] > 1e-6)
iwp_bins = np.logspace(-5, 1, 7)
fig, axes = plot_connectedness(sample, mask, liq_cld_cond, ice_cld_cond)
fig.savefig(f"plots/model_results/{run}/connectedness.png", dpi=300, bbox_inches="tight")

# %%
