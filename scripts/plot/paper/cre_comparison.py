# %% import
import numpy as np
import matplotlib.pyplot as plt
from src.read_data import load_cre

# %% load data
cre_binned, cre_interpolated, cre_interpolated_average = load_cre()

# %% plot mean CRE vs IWP
IWP_points = cre_interpolated_average["IWP"]
fig, axes = plt.subplots(1, 5, sharey="row", figsize=(15, 3), sharex='row')
plt.rcParams.update({'font.size': 8})


def plot_cre(ax, mode, iwp):
    ax.plot(
        iwp,
        cre_interpolated_average[mode + "_net"],
        label="Net CRE",
        color="k",
    )
    ax.plot(
        iwp,
        cre_interpolated_average[mode + "_sw"],
        label="SW CRE",
        color="blue",
    )
    ax.plot(
        iwp,
        cre_interpolated_average[mode + "_lw"],
        label="LW CRE",
        color="r",
    )

# normal cre - all clouds included
plot_cre(axes[0], mode="all_clouds", iwp=IWP_points)

# ice only
plot_cre(axes[1], mode='ice_only', iwp=IWP_points)

# ice over low clouds
plot_cre(axes[2], mode="ice_over_lc", iwp=IWP_points)

# all high clouds
plot_cre(axes[3], mode="all", iwp=IWP_points)

# considering connectedness
plot_cre(axes[4], mode="connected", iwp=IWP_points)


for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlabel("IWP / kg m$^{-2}$")
    ax.axhline(0, color="k", linestyle="--")
    ax.set_xlim(1e-5, 1)
    ax.set_xticks([1e-5, 1e-3, 1e-1])

axes[0].set_ylabel("Cloud Radiative Effect / W m$^{-2}$")
axes[0].set_title("All Clouds")
axes[1].set_title("High Clouds without Low Clouds")
axes[2].set_title("High Clouds over Low Clouds")
axes[3].set_title("All High Clouds")
axes[4].set_title("All High Clouds Connected")

# legend outside of axes
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.15))
fig.savefig("plots/paper/CREs.png", dpi=300, bbox_inches="tight")

# %% 
