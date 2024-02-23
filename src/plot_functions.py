import matplotlib.pyplot as plt
from src.icon_arts_analysis import cut_data, cut_data_mixed


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


def plot_model_output(
    result,
    IWP_bins,
    mask,
    atms,
    fluxes_3d_noice,
    lw_vars,
    lw_vars_avg,
    sw_vars,
    sw_vars_avg,
    lc_vars,
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
