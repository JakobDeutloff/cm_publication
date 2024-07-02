# %% import 
from src.plot_functions import plot_model_output_icon_with_cre
from src.read_data import load_parameters
import pickle
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# %% load data
path = '/work/bm1183/m301049/iwp_framework/ngc3/model_output/'
run = "first_try"

with open(path + run + '.pkl', 'rb') as f:
    result = pickle.load(f)
sample = xr.open_dataset(f"/work/bm1183/m301049/iwp_framework/ngc3/data/representative_sample_c3_conn3.nc")
atms = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/atms_proc.nc")
fluxes_noice = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/fluxes_nofrozen.nc")
cre = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/cre_high_clouds.nc")
sw_vars = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/sw_vars.nc")
lw_vars = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/lw_vars.nc")
low_trop_vars = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/lower_trop_vars.nc")
mean_hc_albedo = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/mean_hc_albedo.nc")
mean_hc_emissivity = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/mean_hc_emissivity.nc")
mean_alpha_t = xr.open_dataset("/work/bm1183/m301049/iwp_framework/ngc3/data/mean_alpha_t.nc")
params = load_parameters(experiment='ngc3')

# %% iwp hist
IWP_bins = np.logspace(-6, np.log10(30), 51) 
n_cells = len(sample.time) * len(sample.cell)
hist, edges = np.histogram(sample["IWP"].where(sample['mask_height']), bins=IWP_bins)
hist = hist / n_cells
mean_cre = cre.where(atms['mask_height']).mean(['local_time_points', 'profile'])

fig, ax = plt.subplots()
ax.plot(mean_cre['iwp_points'], mean_cre['sw'], label='sw', color='blue')
ax.plot(mean_cre['iwp_points'], mean_cre['lw'], label='lw', color='red')
ax.plot(mean_cre['iwp_points'], mean_cre['net'], label='net', color='black')
ax.plot(result.index, result['SW_cre'], label='sw model', color='blue', linestyle='--')
ax.plot(result.index, result['LW_cre'], label='lw model', color='red', linestyle='--')
ax.plot(result.index, result['LW_cre'] + result['SW_cre'], label='net model', color='black', linestyle='--')
ax.set_xscale('log')

cres={}
cres_arts = {}
cres_arts['net'] = (mean_cre['net'] * hist).sum()
cres_arts['sw'] = (mean_cre['sw'] * hist).sum()
cres_arts['lw'] = (mean_cre['lw'] * hist).sum()
cres['sw'] = (result['SW_cre'] * hist).sum()
cres['lw'] = (result['LW_cre'] * hist).sum()
cres['net'] = cres['lw'] + cres['sw']
for key in cres.keys():
    print(f"{key}: Model:{cres[key]:.2f}, ARTS:{cres_arts[key]:.2f} ")
    print(f"relative error: {(cres[key] - cres_arts[key]) / cres_arts[key]:.2f}")


# %%  plot fancy results 
fig, ax = plot_model_output_icon_with_cre(
    result=result,
    IWP_bins=np.logspace(-6, np.log10(30), 51),
    atms=atms,
    fluxes_noice=fluxes_noice,
    sw_vars=sw_vars,
    lw_vars=lw_vars,
    mean_hc_albedo=mean_hc_albedo.to_array(),
    mean_hc_emissivity=mean_hc_emissivity.to_array(),
    mean_alpha_t=mean_alpha_t.to_array(),
    low_trop_vars=low_trop_vars,
    params=params,
    cre=cre
)
fig.savefig('plots/model_results/cycle3/results.png', dpi=500, bbox_inches='tight')


# %% 
