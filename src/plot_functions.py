import matplotlib.pyplot as plt
from src.helper_functions import cut_data, cut_data_mixed
from src.helper_functions import hor_mean
import xarray as xr
import numpy as np
from matplotlib.ticker import ScalarFormatter


def plot_profiles(lat, lon, atms, fluxes_3d):
    fig, axes = plt.subplots(2, 5, figsize=(12, 10), sharey="row")
    data = atms.sel(lat=lat, lon=lon, method="nearest").sel(pressure=slice(100000, 5000))
    fluxes = fluxes_3d.sel(lat=lat, lon=lon, method="nearest").sel(
        pressure=slice(100000, 5000), p_half=slice(100000, 5000)
    )

    height = data["pressure"] / 100
    half_height = fluxes["p_half"] / 100

    for ax in axes.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

    # plot frozen hydrometeors
    axes[0, 0].plot(data["IWC"], height, label="IWC", color="k")
    axes[0, 0].plot(data["snow"], height, label="snow", color="k", linestyle="--")
    axes[0, 0].plot(data["graupel"], height, label="graupel", color="k", linestyle=":")
    axes[0, 0].set_xlabel("F. Hyd. / kg m$^{-3}$")
    axes[0, 0].legend()

    # plot liquid hydrometeors
    axes[0, 1].plot(data["LWC"], height, label="LWC", color="k")
    axes[0, 1].plot(data["rain"], height, label="rain", color="k", linestyle="--")
    axes[0, 1].set_xlabel("L. Hyd. / kg m$^{-3}$")
    axes[0, 1].legend()

    # plot temperature
    axes[0, 2].plot(data["temperature"], height, color="black")
    axes[0, 2].set_xlabel("Temperature / K")

    # plot LW fluxes up
    axes[1, 0].plot(-1 * fluxes["allsky_lw_up"], height, label="allsky", color="k")
    axes[1, 0].plot(
        -1 * fluxes["clearsky_lw_up"],
        height,
        label="clearsky",
        color="k",
        linestyle="--",
    )
    axes[1, 0].plot()
    axes[1, 0].set_xlabel("LW Up / W m$^{-2}$")
    axes[1, 0].legend()

    # plot LW fluxes down
    axes[1, 1].plot(fluxes["allsky_lw_down"], height, label="allsky", color="k")
    axes[1, 1].plot(fluxes["clearsky_lw_down"], height, label="clearsky", color="k", linestyle="--")
    axes[1, 1].set_xlabel("LW Down / W m$^{-2}$")
    axes[1, 1].legend()

    # plot SW fluxes up
    axes[1, 2].plot(-1 * fluxes["allsky_sw_up"], height, label="allsky", color="k")
    axes[1, 2].plot(
        -1 * fluxes["clearsky_sw_up"],
        height,
        label="clearsky",
        color="k",
        linestyle="--",
    )
    axes[1, 2].set_xlabel("SW Up / W m$^{-2}$")
    axes[1, 2].legend()

    # plot SW fluxes down
    axes[1, 3].plot(fluxes["allsky_sw_down"], height, label="allsky", color="k")
    axes[1, 3].plot(fluxes["clearsky_sw_down"], height, label="clearsky", color="k", linestyle="--")
    axes[1, 3].set_xlabel("SW Down / W m$^{-2}$")
    axes[1, 3].legend()

    # plot lw heating rates
    axes[1, 4].plot(-1 * fluxes["allsky_hr_lw"], half_height, label="allsky", color="k")
    axes[1, 4].plot(
        -1 * fluxes["clearsky_hr_lw"],
        half_height,
        label="clearsky",
        color="k",
        linestyle="--",
    )
    axes[1, 4].set_xlabel("LW HR / K d$^{-1}$")
    axes[1, 4].axvline(0, linestyle=":", color="k")
    axes[1, 4].legend()

    # plot sw heating rates
    axes[0, 3].plot(fluxes["allsky_hr_sw"], height, label="allsky", color="k")
    axes[0, 3].plot(fluxes["clearsky_hr_sw"], height, label="clearsky", color="k", linestyle="--")
    axes[0, 3].set_xlabel("SW HR / K d$^{-1}$")
    axes[0, 3].axvline(0, linestyle=":", color="k")
    axes[0, 3].legend()

    axes[1, 0].set_ylabel("pressure / hPa")
    axes[0, 0].set_ylabel("pressure / hPa")

    # plot coordinates and toa fluxes
    axes[0, 4].remove()
    fig.text(
        0.9,
        0.8,
        f"lat: {lat.round(2)}\nlon: {lon.round(2)}\nIWP: {atms.sel(lat=lat, lon=lon, method='nearest')['IWP'].values.round(2)}",
        ha="center",
        va="center",
        fontsize=11,
    )

    fig.tight_layout()


def plot_profiles_noice(lat, lon, atms, fluxes_3d, fluxes_3d_noice):
    fig, axes = plt.subplots(2, 5, figsize=(12, 10), sharey="row")
    data = atms.sel(lat=lat, lon=lon, method="nearest").sel(pressure=slice(100000, 5000))
    fluxes = fluxes_3d.sel(lat=lat, lon=lon, method="nearest").sel(
        pressure=slice(100000, 5000), p_half=slice(100000, 5000)
    )
    fluxes_noice = fluxes_3d_noice.sel(lat=lat, lon=lon, method="nearest").sel(
        pressure=slice(100000, 5000), p_half=slice(100000, 5000)
    )

    height = data["pressure"] / 100
    half_height = fluxes["p_half"] / 100

    for ax in axes.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].invert_yaxis()
    axes[1, 0].invert_yaxis()

    # plot frozen hydrometeors
    axes[0, 0].plot(data["IWC"], height, label="IWC", color="k")
    axes[0, 0].plot(data["snow"], height, label="snow", color="k", linestyle="--")
    axes[0, 0].plot(data["graupel"], height, label="graupel", color="k", linestyle=":")
    axes[0, 0].set_xlabel("F. Hyd. / kg m$^{-3}$")
    axes[0, 0].legend()

    # plot liquid hydrometeors
    axes[0, 1].plot(data["LWC"], height, label="LWC", color="k")
    axes[0, 1].plot(data["rain"], height, label="rain", color="k", linestyle="--")
    axes[0, 1].set_xlabel("L. Hyd. / kg m$^{-3}$")
    axes[0, 1].legend()

    # plot temperature
    axes[0, 2].plot(data["temperature"], height, color="black")
    axes[0, 2].set_xlabel("Temperature / K")

    # plot LW fluxes up
    axes[1, 0].plot(-1 * fluxes["allsky_lw_up"], height, label="allsky", color="k")
    axes[1, 0].plot(
        -1 * fluxes_noice["allsky_lw_up"],
        height,
        label="noice",
        color="k",
        linestyle="--",
    )
    axes[1, 0].plot()
    axes[1, 0].set_xlabel("LW Up / W m$^{-2}$")
    axes[1, 0].legend()

    # plot LW fluxes down
    axes[1, 1].plot(fluxes["allsky_lw_down"], height, label="allsky", color="k")
    axes[1, 1].plot(
        fluxes_noice["allsky_lw_down"], height, label="noice", color="k", linestyle="--"
    )
    axes[1, 1].set_xlabel("LW Down / W m$^{-2}$")
    axes[1, 1].legend()

    # plot SW fluxes up
    axes[1, 2].plot(-1 * fluxes["allsky_sw_up"], height, label="allsky", color="k")
    axes[1, 2].plot(
        -1 * fluxes_noice["allsky_sw_up"],
        height,
        label="noice",
        color="k",
        linestyle="--",
    )
    axes[1, 2].set_xlabel("SW Up / W m$^{-2}$")
    axes[1, 2].legend()

    # plot SW fluxes down
    axes[1, 3].plot(fluxes["allsky_sw_down"], height, label="allsky", color="k")
    axes[1, 3].plot(
        fluxes_noice["allsky_sw_down"], height, label="noice", color="k", linestyle="--"
    )
    axes[1, 3].set_xlabel("SW Down / W m$^{-2}$")
    axes[1, 3].legend()

    # plot lw heating rates
    axes[1, 4].plot(-1 * fluxes["allsky_hr_lw"], half_height, label="allsky", color="k")
    axes[1, 4].plot(
        -1 * fluxes_noice["allsky_hr_lw"],
        half_height,
        label="noice",
        color="k",
        linestyle="--",
    )
    axes[1, 4].set_xlabel("LW HR / K d$^{-1}$")
    axes[1, 4].axvline(0, linestyle=":", color="k")
    axes[1, 4].legend()

    # plot sw heating rates
    axes[0, 3].plot(fluxes["allsky_hr_sw"], half_height, label="allsky", color="k")
    axes[0, 3].plot(
        fluxes_noice["allsky_hr_sw"],
        half_height,
        label="noice",
        color="k",
        linestyle="--",
    )
    axes[0, 3].set_xlabel("SW HR / K d$^{-1}$")
    axes[0, 3].axvline(0, linestyle=":", color="k")
    axes[0, 3].legend()

    axes[1, 0].set_ylabel("pressure / hPa")
    axes[0, 0].set_ylabel("pressure / hPa")

    # plot coordinates and toa fluxes
    axes[0, 4].remove()
    fig.text(
        0.9,
        0.8,
        f"lat: {lat.round(2)}\nlon: {lon.round(2)}\nIWP: {atms.sel(lat=lat, lon=lon, method='nearest')['IWP'].values.round(2)}",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.tight_layout()

    return fig, axes


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


def plot_model_output_arts(
    result,
    IWP_bins,
    mask,
    atms,
    fluxes_3d_noice,
    lw_vars,
    sw_vars,
    cre_average,
    mode="all",
):
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex="col")

    # hc temperature
    axes[0, 0].scatter(
        cut_data(atms["IWP"], mask),
        cut_data(lw_vars["h_cloud_temperature"], mask),
        s=0.1,
        color="k",
    )
    axes[0, 0].plot(result["T_hc"], color="magenta")
    axes[0, 0].set_ylabel(r"$\mathrm{T_{hc}}$ / K")

    # LWP
    axes[0, 1].scatter(cut_data(atms["IWP"], mask), cut_data(atms["LWP"], mask), s=0.1, color="k")
    axes[0, 1].plot(result["LWP"], color="magenta")
    axes[0, 1].set_ylim(1e-5, 1e1)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel(r"$\mathrm{LWP ~/~ kg ~m^{-2}}$")

    # lc fraction
    axes[1, 0].plot(result["lc_fraction"], color="magenta")
    axes[1, 0].set_ylabel(r"$f$")

    # alpha_t
    alpha_t = cut_data_mixed(
        fluxes_3d_noice["albedo_clearsky"],
        fluxes_3d_noice["albedo_allsky"],
        mask,
        atms["connected"],
    )
    axes[1, 1].scatter(
        cut_data(atms["IWP"], mask),
        alpha_t,
        s=0.1,
        color="k",
    )
    alpha_t.groupby_bins(cut_data(atms["IWP"], mask), bins=IWP_bins).mean().plot(
        ax=axes[1, 1], color="limegreen", label="Average"
    )
    axes[1, 1].plot(result["alpha_t"], color="magenta", label="Model")
    axes[1, 1].set_ylabel(r"$\alpha_t$")

    # R_t
    R_t = cut_data_mixed(
        fluxes_3d_noice["clearsky_lw_up"].isel(pressure=-1),
        fluxes_3d_noice["allsky_lw_up"].isel(pressure=-1),
        mask,
        atms["connected"],
    )
    axes[2, 0].scatter(cut_data(atms["IWP"], mask), R_t, s=0.1, color="k")
    R_t.groupby_bins(cut_data(atms["IWP"], mask), bins=IWP_bins).mean().plot(
        ax=axes[2, 0], color="limegreen", label="Average"
    )
    axes[2, 0].plot(result["R_t"], color="magenta", label="Model")
    axes[2, 0].set_ylabel(r"$\mathrm{R_t}$ / $\mathrm{W ~ m^{-2}}$")

    # hc albedo
    axes[2, 1].scatter(
        cut_data(atms["IWP"], mask),
        cut_data(sw_vars["high_cloud_albedo"], mask),
        s=0.1,
        color="k",
    )
    axes[2, 1].plot(result["alpha_hc"], color="magenta", label="Model")
    axes[2, 1].set_ylabel(r"$\alpha$")
    axes[2, 1].set_ylim(0, 1)

    # hc emissivity
    axes[3, 0].scatter(
        cut_data(atms["IWP"], mask),
        cut_data(lw_vars["high_cloud_emissivity"], mask),
        s=0.1,
        color="k",
        label="data",
    )
    cut_data(lw_vars["high_cloud_emissivity"], mask).groupby_bins(
        cut_data(atms["IWP"], mask), bins=IWP_bins
    ).median().plot(ax=axes[3, 0], color="limegreen", label="Average")
    axes[3, 0].plot(result["em_hc"], color="magenta", label="Model")
    axes[3, 0].set_ylabel(r"$\epsilon$")
    axes[3, 0].set_ylim(0, 1.3)

    # CRE
    axes[3, 1].plot(result["SW_cre"], color="blue", label="SW")
    axes[3, 1].plot(result["LW_cre"], color="red", label="LW")
    axes[3, 1].plot(result["SW_cre"] + result["LW_cre"], color="k", label="Net")
    if mode == "all":
        axes[3, 1].plot(cre_average["IWP"], cre_average["all_sw"], color="blue", linestyle="--")
        axes[3, 1].plot(cre_average["IWP"], cre_average["all_lw"], color="red", linestyle="--")
        axes[3, 1].plot(cre_average["IWP"], cre_average["all_net"], color="k", linestyle="--")
    elif mode == "ice_only":
        axes[3, 1].plot(
            cre_average["IWP"], cre_average["ice_only_sw"], color="blue", linestyle="--"
        )
        axes[3, 1].plot(cre_average["IWP"], cre_average["ice_only_lw"], color="red", linestyle="--")
        axes[3, 1].plot(cre_average["IWP"], cre_average["ice_only_net"], color="k", linestyle="--")
    elif mode == "ice_over_lc":
        axes[3, 1].plot(
            cre_average["IWP"], cre_average["ice_over_lc_sw"], color="blue", linestyle="--"
        )
        axes[3, 1].plot(
            cre_average["IWP"], cre_average["ice_over_lc_lw"], color="red", linestyle="--"
        )
        axes[3, 1].plot(
            cre_average["IWP"], cre_average["ice_over_lc_net"], color="k", linestyle="--"
        )
    elif mode == "connected":
        axes[3, 1].plot(
            cre_average["IWP"], cre_average["connected_sw"], color="blue", linestyle="--"
        )
        axes[3, 1].plot(
            cre_average["IWP"], cre_average["connected_lw"], color="red", linestyle="--"
        )
        axes[3, 1].plot(cre_average["IWP"], cre_average["connected_net"], color="k", linestyle="--")
    else:
        raise ValueError('mode must be one of "all", "ice_only", "ice_over_lc", "connected"')

    axes[3, 1].set_ylabel("CRE / W m${^-2}$")
    axes[3, 1].legend()

    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("")
        ax.set_xlabel("")

    axes[3, 0].set_xlabel("IWP / kg m$^{-2}$")
    axes[3, 1].set_xlabel("IWP / kg m$^{-2}$")
    handles, labels = axes[3, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    return fig, axes


def plot_model_output_icon(
    result,
    IWP_bins,
    sample,
):
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex="col")
    IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2

    # hc temperature
    axes[0, 0].plot(result["T_hc"], color="magenta")
    axes[0, 0].set_ylabel(r"$\mathrm{T_{hc}}$ / K")

    # lc fraction
    binned_lc_frac = ((sample["LWP"] > 1e-4) * 1).groupby_bins(sample["IWP"], bins=IWP_bins).mean()
    axes[0, 1].plot(result["lc_fraction"], color="magenta", label="Unconnected LCs")
    axes[0, 1].plot(IWP_points, binned_lc_frac, color="limegreen", label="All LCs")
    axes[0, 1].set_ylabel(r"$f$")
    axes[0, 1].legend()

    # alpha_t
    axes[1, 1].plot(result["alpha_t"], color="magenta", label="Model")
    axes[1, 1].set_ylabel(r"$\alpha_t$")

    # R_t
    axes[2, 0].plot(result["R_t"], color="magenta", label="Model")
    axes[2, 0].set_ylabel(r"$\mathrm{R_t}$ / $\mathrm{W ~ m^{-2}}$")

    # hc albedo
    axes[2, 1].plot(result["alpha_hc"], color="magenta", label="Model")
    axes[2, 1].set_ylabel(r"$\alpha$")
    axes[2, 1].set_ylim(0, 1)

    # hc emissivity
    axes[3, 0].plot(result["em_hc"], color="magenta", label="Model")
    axes[3, 0].set_ylabel(r"$\epsilon$")
    axes[3, 0].set_ylim(0, 1.3)

    # CRE
    axes[3, 1].plot(result["SW_cre"], color="blue", label="SW")
    axes[3, 1].plot(result["LW_cre"], color="red", label="LW")
    axes[3, 1].plot(result["SW_cre"] + result["LW_cre"], color="k", label="Net")
    axes[3, 1].grid()

    # Plot setup
    axes[3, 1].set_ylabel("CRE / W m${^-2}$")
    axes[3, 1].legend()

    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_xlim(IWP_bins.min(), IWP_bins.max())

    axes[3, 0].set_xlabel("IWP / kg m$^{-2}$")
    axes[3, 1].set_xlabel("IWP / kg m$^{-2}$")

    return fig, axes


def plot_condensate(sample, ax, min, max, mask, liq_cld_cond, ice_cld_cond, mode="icon"):
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

    if mode == 'icon':
        vert_coord = sample['level_full']
    else:
        vert_coord = sample['pressure']/100

    mean_liq = hor_mean(
        liq_cld_cond.where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask), mode
    )
    mean_ice = hor_mean(
        ice_cld_cond.where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask), mode
    )

    mean_rain = hor_mean(
        sample["rain"].where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask), mode
    )
    mean_lwc = hor_mean(
        sample["LWC"].where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask), mode
    )
    mean_iwc = hor_mean(
        sample["IWC"]
        .where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask), mode
    )
    mean_snow = hor_mean(
        sample["snow"]
        .where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask), mode
    )
    mean_graupel = hor_mean(
        sample["graupel"]
        .where((sample["IWP"] >= min) & (sample["IWP"] < max) & mask), mode
    )

    ax.plot(mean_liq, vert_coord, color="k", label="Liquid")
    ax.plot(mean_rain, vert_coord, color="k", linestyle="--", label="Rain")
    ax.plot(mean_lwc, vert_coord, color="k", linestyle=":", label="LWC")
    ax2.plot(mean_ice, vert_coord, color="b", label="Ice")
    ax2.plot(mean_iwc, vert_coord, color="b", linestyle=":", label="IWC")
    ax2.plot(mean_snow, vert_coord, color="b", linestyle="--", label="Snow")
    ax2.plot(mean_graupel, vert_coord, color="b", linestyle="-.", label="Graupel")
    ax2.set_xlabel("Ice Cond. / kg/m$^3$", color="b")
    ax.set_xlabel("Liquid Cond. / kg/m$^3$")
    ax.spines["right"].set_visible(False)
    if mode == 'icon':
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

    ax.text(0.05, 0.90, f"{min:.0e} kg/m$^3$ - {max:.0e} kg/m$^3$", transform=ax.transAxes)
    ax.text(0.05, 0.85, f"Number of Profiles: {n_profiles:.0f}", transform=ax.transAxes)
    ax.text(
        0.05,
        0.80,
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

    fig, axes = plt.subplots(2, 6, figsize=(22, 12), sharey="row")
    iwp_bins = np.logspace(-5, 1, 7)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 2))

    for i in range(6):
        ax2 = plot_condensate(
            sample, axes[0, i], iwp_bins[i], iwp_bins[i + 1], mask, liq_cld_cond, ice_cld_cond, mode
        )
        ax2.xaxis.set_major_formatter(formatter)
        plot_num_connected(sample, axes[1, i], iwp_bins[i], iwp_bins[i + 1], mask)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles + handles2, labels + labels2, bbox_to_anchor=(0.5, 0.3), loc="lower center", ncols=4
    )
    if mode == "icon":
        axes[0, 0].set_ylabel("Model Level")
        axes[0, 0].invert_yaxis()
    else:
        axes[0, 0].set_ylabel("Pressure / hPa")
    
    # Set the x-axis formatter
    for ax in axes.flatten():
        ax.xaxis.set_major_formatter(formatter)

    return fig, axes


def plot_sum_cre(result, sample, iwp_bins, mode='icon'):

    if mode == 'icon':
        n_cells = (len(sample.cell) * len(sample.time))
    else:
        n_cells = len(sample.lat) * len(sample.lon)

    hist, edges = np.histogram(sample["IWP"], bins=iwp_bins)
    hist = hist / n_cells
    sum_sw = (result["SW_cre"] * hist).sum()
    sum_lw = (result["LW_cre"] * hist).sum()
    sum_net = sum_sw + sum_lw

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex="col")

    result["SW_cre"].plot(ax=axes[0], label="SW CRE", color="blue")
    result["LW_cre"].plot(ax=axes[0], label="LW CRE", color="red")
    (result["SW_cre"] + result["LW_cre"]).plot(ax=axes[0], label="Net CRE", color="black")
    axes[0].legend()
    axes[0].axhline(0, color="grey", linestyle="--")
    axes[0].set_ylabel("CRE / Wm$^{-2}$")

    axes[1].stairs(hist, edges, label="IWP", color="black")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("IWP / kgm$^{-2}$")
    axes[1].set_ylabel("Fraction of Gridcells")

    axes[1].text(
        0.7, 0.90, f"SW CRE: {sum_sw:.2f} W/m$^2$", transform=axes[1].transAxes, color="blue"
    )
    axes[1].text(
        0.7, 0.83, f"LW CRE: {sum_lw:.2f} W/m$^2$", transform=axes[1].transAxes, color="red"
    )
    axes[1].text(
        0.7, 0.76, f"Net CRE: {sum_net:.2f} W/m$^2$", transform=axes[1].transAxes, color="black"
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    return fig, axes
