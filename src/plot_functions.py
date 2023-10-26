import matplotlib.pyplot as plt

def plot_profiles(lat, lon, atms, fluxes_3d):
    fig, axes = plt.subplots(2, 5, figsize=(12, 10), sharey="row")
    data = atms.sel(lat=lat, lon=lon, method="nearest").sel(pressure=slice(100000, 5000))
    fluxes = fluxes_3d.sel(lat=lat, lon=lon, method="nearest").sel(pressure=slice(100000, 5000), p_half=slice(100000, 5000))

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
    axes[1, 0].plot(fluxes["allsky_lw_up"], height, label="allsky", color="k")
    axes[1, 0].plot(
        fluxes["clearsky_lw_up"], height, label="clearsky", color="k", linestyle="--"
    )
    axes[1, 0].set_xlabel("LW Up / W m$^{-2}$")
    axes[1, 0].legend()

    # plot LW fluxes down
    axes[1, 1].plot(fluxes["allsky_lw_down"], height, label="allsky", color="k")
    axes[1, 1].plot(
        fluxes["clearsky_lw_down"], height, label="clearsky", color="k", linestyle="--"
    )
    axes[1, 1].set_xlabel("LW Down / W m$^{-2}$")
    axes[1, 1].legend()

    # plot SW fluxes up
    axes[1, 2].plot(fluxes["allsky_sw_up"], height, label="allsky", color="k")
    axes[1, 2].plot(
        fluxes["clearsky_sw_up"], height, label="clearsky", color="k", linestyle="--"
    )
    axes[1, 2].set_xlabel("SW Up / W m$^{-2}$")
    axes[1, 2].legend()

    # plot SW fluxes down
    axes[1, 3].plot(fluxes["allsky_sw_down"], height, label="allsky", color="k")
    axes[1, 3].plot(
        fluxes["clearsky_sw_down"], height, label="clearsky", color="k", linestyle="--"
    )
    axes[1, 3].set_xlabel("SW Down / W m$^{-2}$")
    axes[1, 3].legend()

    # plot lw heating rates 
    axes[1, 4].plot(fluxes["allsky_hr_lw"], height, label="allsky", color="k")
    axes[1, 4].plot(
        fluxes["clearsky_hr_lw"], height, label="clearsky", color="k", linestyle="--"
    )
    axes[1, 4].set_xlabel("LW HR / K d$^{-1}$")
    axes[1, 4].axvline(0, linestyle=':', color='k')
    axes[1, 4].legend()

    # plot sw heating rates
    axes[0, 3].plot(fluxes["allsky_hr_sw"], height, label="allsky", color="k")
    axes[0, 3].plot(
        fluxes["clearsky_hr_sw"], height, label="clearsky", color="k", linestyle="--"
    )
    axes[0, 3].set_xlabel("SW HR / K d$^{-1}$")
    axes[0, 3].axvline(0, linestyle=':', color='k')
    axes[0, 3].legend()

    axes[1, 0].set_ylabel("pressure / hPa")
    axes[0, 0].set_ylabel("pressure / hPa")

    # plot coordinates and toa fluxes
    axes[0, 4].remove()
    fig.text(
        0.9,
        0.8,
        f"lat: {lat.round(2)}\nlon: {lon.round(2)}",
        ha="center",
        va="center",
        fontsize=11,
    )

    fig.tight_layout()