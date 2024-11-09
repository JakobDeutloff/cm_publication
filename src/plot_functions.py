"""
Plot functions 
"""

import matplotlib.pyplot as plt
from src.helper_functions import cut_data, cut_data_mixed, hor_mean
import xarray as xr
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl


def scatterplot(
    x_data,
    y_data,
    color_data=None,
    xlabel="IWP / kg m$^{-2}$",
    ylabel="",
    title=None,
    xlim=None,
    ylim=None,
    logx=True,
    logy=False,
    logc=False,
    cbar_label=None,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if color_data is not None:
        c = ax.scatter(
            x_data,
            y_data,
            marker="o",
            s=0.2,
            c=color_data,
            cmap="viridis",
        )
        fig.colorbar(c, ax=ax, label=cbar_label)
    else:
        ax.scatter(x_data, y_data, marker="o", s=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_condensate(
    sample, ax, min, max, mask, liq_cld_cond, ice_cld_cond, mode="icon"
):
    """
    Plot the condensate variables (liquid, rain, LWC, ice, IWC, snow, graupel) against a vertical coordinate.

    Parameters:
    sample (pandas.DataFrame): The sample data containing the condensate variables.
    ax (matplotlib.axes.Axes): The axes object to plot the data on.
    min (float): The minimum value of IWP (Ice Water Path) to consider for plotting.
    max (float): The maximum value of IWP (Ice Water Path) to consider for plotting.
    mask (pandas.Series): The mask to apply on the data.
    liq_cld_cond (xarray.DataArray): The liquid condensate data.
    ice_cld_cond (xarray.DataArray): The ice condensate data.
    mode (str, optional): The mode of plotting. Defaults to "icon".

    Returns:
    matplotlib.axes.Axes: The axes object with the plotted data.
    """
    ax2 = ax.twiny()

    if mode == "icon":
        vert_coord = sample["level_full"]
    else:
        vert_coord = sample["pressure"] / 100

    mean_liq = hor_mean(
        liq_cld_cond.where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask), mode
    )
    mean_ice = hor_mean(
        ice_cld_cond.where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask), mode
    )

    mean_rain = hor_mean(
        sample["rain"].where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask),
        mode,
    )
    mean_lwc = hor_mean(
        sample["LWC"].where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask), mode
    )
    mean_iwc = hor_mean(
        sample["IWC"].where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask), mode
    )
    mean_snow = hor_mean(
        sample["snow"].where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask),
        mode,
    )
    mean_graupel = hor_mean(
        sample["graupel"].where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask),
        mode,
    )

    ax.plot(mean_liq, vert_coord, color="k", label="Liquid")
    ax.plot(mean_rain, vert_coord, color="k", linestyle="--", label="Rain")
    ax.plot(mean_lwc, vert_coord, color="k", linestyle=":", label="Cloud Liquid")
    ax2.plot(mean_ice, vert_coord, color="b", label="Ice")
    ax2.plot(mean_iwc, vert_coord, color="b", linestyle=":", label="Cloud Ice")
    ax2.plot(mean_snow, vert_coord, color="b", linestyle="--", label="Snow")
    ax2.plot(mean_graupel, vert_coord, color="b", linestyle="-.", label="Graupel")
    ax2.set_xlabel("Ice / kg m$^{-3}$", color="b")
    ax.set_xlabel("Liq. / kg m$^{-3}$")
    ax.spines["right"].set_visible(False)
    if mode == "icon":
        ax.set_ylim(30, 90)
    else:
        ax.set_ylim(1000, 100)
    ax2.spines["right"].set_visible(False)

    return ax2


def plot_num_connected(sample, ax, min, max, mask):
    """
    Plot the number of connected profiles within a given range of IWP values.

    Parameters:
    - sample: DataFrame
        The sample data containing the IWP and connected columns.
    - ax: AxesSubplot
        The matplotlib axes object to plot on.
    - min: float
        The minimum IWP value for the range.
    - max: float
        The maximum IWP value for the range.
    - mask: boolean array
        A boolean mask to filter the sample data.

    Returns:
    None
    """

    mask_selection = mask & (sample["IWP"] > min) & (sample["IWP"] < max)
    connected = sample["connected"].where(mask_selection)
    n_profiles = (~np.isnan(connected) * 1).sum().values
    connected_profiles = connected.sum().values

    ax.text(
        0.05,
        0.8,
        f"$I$-Bin : {min:.0e} - {max:.0e}" + " / kg m$^{-2}$",
        transform=ax.transAxes,
    )
    ax.text(
        0.05, 0.675, f"Number of Profiles: {n_profiles:.0f}", transform=ax.transAxes
    )
    ax.text(
        0.05,
        0.55,
        f"Connected Profiles: {connected_profiles.sum()/n_profiles*100:.1f}%",
        transform=ax.transAxes,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])


def plot_connectedness(sample, mask, liq_cld_cond, ice_cld_cond, mode="icon"):

    """
    Plots the connectedness of cloud condensate and the number of connected regions
    as a function of ice water path (IWP) bins.

    Parameters:
    - sample: The sample data.
    - mask: The mask data.
    - liq_cld_cond: The liquid cloud condensate data.
    - ice_cld_cond: The ice cloud condensate data.
    - mode: The mode of the plot. Default is "icon".

    Returns:
    - fig: The matplotlib figure object.
    - axes: The matplotlib axes object.
    """

    fig, axes = plt.subplots(
        2, 3, figsize=(8, 6.8), sharey="row", gridspec_kw={"height_ratios": [3, 1]}
    )
    iwp_bins = np.logspace(-5, 1, 4)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 2))

    for i in range(3):
        ax2 = plot_condensate(
            sample,
            axes[0, i],
            iwp_bins[i],
            iwp_bins[i + 1],
            mask,
            liq_cld_cond,
            ice_cld_cond,
            mode,
        )
        ax2.xaxis.set_major_formatter(formatter)
        plot_num_connected(sample, axes[1, i], iwp_bins[i], iwp_bins[i + 1], mask)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles + handles2,
        labels + labels2,
        bbox_to_anchor=(0.5, 0.1),
        loc="lower center",
        ncols=4,
    )
    if mode == "icon":
        axes[0, 0].set_ylabel("Model Level")
        axes[0, 0].invert_yaxis()
    else:
        axes[0, 0].set_ylabel("Pressure / hPa")
        axes[0, 0].set_yticks([1000, 500, 100])

    # Set the x-axis formatter
    for ax in axes.flatten():
        ax.xaxis.set_major_formatter(formatter)

    return fig, axes

def plot_model_output_arts_with_cre(
    result,
    IWP_bins,
    atms,
    fluxes_3d_noice,
    lw_vars,
    mean_lw_vars,
    sw_vars,
    mean_sw_vars,
    f_lc_vals,
    params,
    cre_average,
):
    fig = plt.figure(figsize=(12, 18))
    mask_tuning = atms["mask_height"] & ~atms["mask_low_cloud"]
    IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2

    # hc temperature
    ax1 = fig.add_subplot(6, 2, 1)
    ax1.scatter(
        cut_data(atms["IWP"], atms["mask_height"]),
        cut_data(atms["hc_top_temperature"], atms["mask_height"]),
        s=0.1,
        color="grey",
    )
    ax1.plot(result["T_hc"], color="red", linestyle="--", label=r"$T_{\mathrm{hc}}(I)$")
    ax1.set_ylabel(r"HC Temperature / K")
    ax1.set_yticks([200, 240])
    ax1.legend()

    # emissivity
    ax2 = fig.add_subplot(6, 2, 2)
    ax2.scatter(
        cut_data(atms["IWP"], mask_tuning),
        cut_data(lw_vars["high_cloud_emissivity"], mask_tuning),
        s=0.1,
        color="grey",
    )

    ax2.plot(mean_lw_vars["binned_emissivity"], color="orange", label="Mean")
    ax2.plot(
        result["em_hc"],
        color="red",
        label=r"$\varepsilon_{\mathrm{hc}}(I)$",
        linestyle="--",
    )
    ax2.set_ylabel(r"HC Emissivity")
    ax2.set_yticks([0, 1])
    ax2.legend()

    # lc fraction
    ax3 = fig.add_subplot(6, 2, 3)
    ax3.plot(
        IWP_points,
        f_lc_vals["raw"],
        label=r"LWP > $10^{-4} ~ \mathrm{kg ~ m^{-2}}$",
        color="grey",
    )
    ax3.plot(
        IWP_points,
        f_lc_vals["unconnected"],
        label="Mean",
        color="orange",
        linestyle="-",
    )
    ax3.plot(
        result["lc_fraction"],
        color="red",
        linestyle="--",
        label=r"$f_{\mathrm{lc}}$",
    )
    ax3.legend()
    ax3.set_ylabel(r"Low-Cloud Fraction")
    ax3.set_yticks([0, 0.5, 1])

    # alpha
    ax4 = fig.add_subplot(6, 2, 4)
    sc_alpha = ax4.scatter(
        cut_data(atms["IWP"], mask_tuning),
        cut_data(sw_vars["high_cloud_albedo"], mask_tuning),
        s=0.1,
        c=cut_data(fluxes_3d_noice["allsky_sw_down"].isel(pressure=-1), mask_tuning),
        cmap="viridis",
    )
    ax4.plot(mean_sw_vars["interpolated_albedo"], color="orange", label="Mean")
    ax4.plot(
        result["alpha_hc"],
        color="red",
        linestyle="--",
        label=r"$\alpha_{\mathrm{hc}}(I)$",
    )
    ax4.set_ylabel(r"HC Albedo")
    ax4.legend()
    ax4.set_yticks([0, 0.8])

    # R_t
    ax5 = fig.add_subplot(6, 2, 5)
    colors = ["black", "grey", "blue"]
    cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
    sc_rt = ax5.scatter(
        cut_data(atms["IWP"], atms["mask_height"]),
        (-1)
        * cut_data_mixed(
            fluxes_3d_noice["clearsky_lw_up"].isel(pressure=-1),
            fluxes_3d_noice["allsky_lw_up"].isel(pressure=-1),
            atms["mask_height"],
            atms["connected"],
        ),
        c=cut_data_mixed(
            (atms["LWP"] * 0) + 1e-12,
            atms["LWP"],
            atms["mask_height"],
            atms["connected"],
        ),
        cmap=cmap,
        norm=LogNorm(vmin=1e-6, vmax=1e0),
        s=0.1,
    )
    mean_rt = (
        cut_data_mixed(
            fluxes_3d_noice["clearsky_lw_up"].isel(pressure=-1),
            fluxes_3d_noice["allsky_lw_up"].isel(pressure=-1),
            atms["mask_height"],
            atms["connected"],
        )
        .groupby_bins(cut_data(atms["IWP"], atms["mask_height"]), bins=IWP_bins)
        .mean()
    ) * (-1)
    ax5.plot(IWP_points, mean_rt, color="orange", label="Mean")
    ax5.axhline(
        -params["R_cs"], color="black", linestyle="--", label=r"$R_{\mathrm{cs}}$"
    )
    ax5.axhline(
        -params["R_l"], color="navy", linestyle="--", label=r"$R_{\mathrm{lc}}$"
    )
    ax5.plot(-result["R_t"], color="red", linestyle="--", label=r"$R_{\mathrm{t}}(I)$")
    ax5.set_ylabel(r"LT LW Emissions / $\mathrm{W ~ m^{-2}}$")
    ax5.legend()
    ax5.set_ylim(200, 350)
    ax5.set_yticks([225, 275, 325])

    # a_t
    ax6 = fig.add_subplot(6, 2, 6)
    sc_at = ax6.scatter(
        cut_data(atms["IWP"], atms["mask_height"]),
        cut_data_mixed(
            sw_vars["clearsky_albedo"],
            sw_vars["allsky_albedo"],
            atms["mask_height"],
            atms["connected"],
        ),
        c=cut_data_mixed(
            (atms["LWP"] * 0) + 1e-12,
            atms["LWP"],
            atms["mask_height"],
            atms["connected"],
        ),
        cmap=cmap,
        norm=LogNorm(vmin=1e-6, vmax=1e0),
        s=0.1,
    )
    mean_a_t = (
        cut_data_mixed(
            sw_vars["clearsky_albedo"],
            sw_vars["allsky_albedo"],
            atms["mask_height"],
            atms["connected"],
        )
        .groupby_bins(cut_data(atms["IWP"], atms["mask_height"]), bins=IWP_bins)
        .mean()
    )
    ax6.plot(IWP_points, mean_a_t, color="orange", label="Mean")
    ax6.axhline(
        params["a_cs"], color="black", linestyle="--", label=r"$\alpha_{\mathrm{cs}}$"
    )
    ax6.axhline(
        params["a_l"], color="navy", linestyle="--", label=r"$\alpha_{\mathrm{lc}}$"
    )
    ax6.plot(
        result["alpha_t"], color="red", linestyle="--", label=r"$\alpha_{\mathrm{t}}$"
    )
    ax6.set_ylabel(r"LT Albedo")
    ax6.legend()
    ax6.set_yticks([0, 0.8])

    # CRE
    ax7 = fig.add_subplot(4, 1, 3)
    ax7.plot(
        cre_average["IWP"], cre_average["connected_sw"], color="blue", linestyle="--"
    )
    ax7.plot(
        cre_average["IWP"], cre_average["connected_lw"], color="red", linestyle="--"
    )
    ax7.plot(
        cre_average["IWP"],
        cre_average["connected_sw"] + cre_average["connected_lw"],
        color="black",
        linestyle="--",
    )
    ax7.plot(result.index, result["SW_cre"], color="blue")
    ax7.plot(result.index, result["LW_cre"], color="red")
    ax7.plot(result.index, result["SW_cre"] + result["LW_cre"], color="black")
    ax7.set_xscale("log")
    # make legend with fake handles and labels
    handles = [
        plt.Line2D([0], [0], color="grey", linestyle="--"),
        plt.Line2D([0], [0], color="grey"),
        plt.Line2D([0], [0], color="red", linestyle="-"),
        plt.Line2D([0], [0], color="blue", linestyle="-"),
        plt.Line2D([0], [0], color="black", linestyle="-"),
    ]
    labels = ["ARTS", "Conceptual Model", "LW", "SW", "Net"]
    ax7.legend(handles, labels)

    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    labels = ["a", "b", "c", "d", "e", "f", "g"]
    for ax in axes:
        ax.set_xscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
        ax.set_xticklabels("")
        ax.set_xlim(1e-5, 10)
        # plot label at top right corner of axis
        ax.text(
            0.05,
            1.08,
            labels.pop(0),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="right",
        )

    ax5.set_xticklabels(["1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1e0", "1e1"])
    ax6.set_xticklabels(["1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1e0", "1e1"])
    ax7.set_xticklabels(["1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1e0", "1e1"])
    ax7.set_xlabel("$I$ / kg m$^{-2}$")
    ax7.set_ylabel("$C(I)$ / W m$^{-2}$")
    ax7.set_yticks([-200, 0, 200])

    # add colorbars
    fig.subplots_adjust(right=0.9)
    cbar_ax1 = fig.add_axes([0.91, 0.64, 0.01, 0.11])
    cbar_ax2 = fig.add_axes([0.91, 0.51, 0.01, 0.11])
    fig.colorbar(sc_alpha, cax=cbar_ax1, label="SW Down / W m$^{-2}$")
    fig.colorbar(sc_rt, cax=cbar_ax2, label="LWP / kg m$^{-2}$")

    # plot CRE

    return fig, axes
