# %% import
import matplotlib.pyplot as plt
from src.read_data import load_icon_snapshot, load_model_output
import numpy as np
from scipy.signal import savgol_filter

# %% load data
ds_monsoon = load_icon_snapshot()
result = load_model_output("prefinal")

# %% multiply hist with cre result
IWP_bins = np.logspace(-5, 1, num=50)
IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2

n_profiles = ds_monsoon['IWP'].count().values
hist, edges = np.histogram(ds_monsoon['IWP'].where(ds_monsoon['mask_height']), bins=IWP_bins)
hist = hist / n_profiles

cre_sw_weighted = hist * result['SW_cre']
cre_lw_weighted = hist * result['LW_cre']
cre_net_weighted = cre_sw_weighted + cre_lw_weighted

# %% plot schematic of cre 
def control_plot(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('k')
    ax.spines[['bottom', 'left']].set_linewidth(2)
    ax.set_xlabel("$I$ / kg $m^{-2}$", color='k')
    ax.set_xscale('log')
    ax.set_xlim(1e-5, 1e1)
    ax.set_xticks([1e-4, 1])
    # set color of xticks and labels to grey
    ax.xaxis.set_tick_params(color='k', labelcolor='k')
    ax.xaxis.set_minor_locator(plt.NullLocator())

net_cre = result['SW_cre'] + result['LW_cre']
net_cre_smooth = savgol_filter(net_cre, 11, 3)
hist_smooth = savgol_filter(hist, 20, 3)
hist_params = np.polyfit(np.log10(IWP_points), hist_smooth, 5)
hist_smooth = np.polyval(hist_params, np.log10(IWP_points))
cre_net_weighted_smooth = savgol_filter(cre_net_weighted, 11, 3)

# %% cre 
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
control_plot(ax)
ax.plot(net_cre.index, net_cre_smooth, linewidth=2, color='black') 
ax.set_yticks([-50, 0, 50])
ax.yaxis.set_tick_params(color='k', labelcolor='k')
ax.axhline(0, color='grey', linewidth=1, linestyle='--')
ax.set_ylabel('$C(I)$ / W $m^{-2}$', color='k')
fig.savefig('plots/cre_scheme.png', dpi=300, bbox_inches='tight')

# %% IWP dist 
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
control_plot(ax)
ax.plot(IWP_points, hist_smooth, linewidth=2, color='black')
ax.set_ylabel('$P(I)$', color='k')
ax.set_yticks([0, 0.02])
ax.yaxis.set_tick_params(color='k', labelcolor='k')
fig.savefig('plots/iwp_scheme.png', dpi=300, bbox_inches='tight')


# %% folded cre
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
control_plot(ax)
ax.fill_between(net_cre.index, cre_net_weighted_smooth, color='grey', alpha=0.5, edgecolor='none', hatch='//')
ax.plot(net_cre.index, cre_net_weighted_smooth, linewidth=2, color='black')
ax.set_yticks([-0.4, 0, 0.4])
ax.yaxis.set_tick_params(color='k', labelcolor='k')
ax.set_ylabel('$C(I) \cdot P(I)$ / W $m^{-2}$', color='k')
fig.savefig('plots/folded_cre_scheme.png', dpi=300, bbox_inches='tight')

# %%
