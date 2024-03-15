import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import healpy as hp


def nnshow(var, nx=3000, ny=3000, ax=None, clabel=None, cl=0, **kwargs):
    """
    var: variable on healpix coordinates (array-like)
    nx: image resolution in x-direction
    ny: image resolution in y-direction
    ax: axis to plot on
    kwargs: additional arguments to imshow
    """
    if ax is None:
        ax = plt.gca()

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    xvals = np.linspace(xlims[0], xlims[1], nx)
    yvals = np.linspace(ylims[0], ylims[1], ny)
    xvals2, yvals2 = np.meshgrid(xvals, yvals)
    # lat lon meshgrid in correct projection
    latlon = ccrs.PlateCarree().transform_points(
        ax.projection, xvals2, yvals2, np.zeros_like(xvals2)
    )
    valid = np.all(np.isfinite(latlon), axis=-1)
    points = latlon[valid].T
    # get pixel number corresponding to lat lon coordinates
    pix = hp.ang2pix(
        hp.npix2nside(len(var)), theta=points[0], phi=points[1], nest=True, lonlat=True
    )
    # get data at lat lon over pixel number - transform is PlateCarree
    res = np.full(latlon.shape[:-1], np.nan, dtype=var.dtype)
    res[valid] = var[pix]

    # If central longitude is not 0, transform data to new central longitude
    if cl != 0:
        b = latlon[:, :, 0]>0
        latlon[:, :, 0][b] = latlon[:, :, 0][b] - cl
        latlon[:, :, 0][~ b] = latlon[:, :, 0][~ b] + cl

    # plot data
    im = ax.imshow(res, extent=xlims + ylims, origin="lower", **kwargs)

    ax.add_feature(cf.COASTLINE, linewidth=0.8)
    return im


def nnshow_world(var, nx=1000, ny=1000, ax=None, **kwargs):
    """
    var: variable on healpix coordinates (array-like)
    nx: image resolution in x-direction
    ny: image resolution in y-direction
    ax: axis to plot on
    kwargs: additional arguments to imshow
    """
    if ax is None:
        ax = plt.gca()

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    xvals = np.linspace(xlims[0], xlims[1], nx)
    yvals = np.linspace(ylims[0], ylims[1], ny)
    xvals2, yvals2 = np.meshgrid(xvals, yvals)
    latlon = ccrs.PlateCarree().transform_points(
        ax.projection, xvals2, yvals2, np.zeros_like(xvals2)
    )
    valid = np.all(np.isfinite(latlon), axis=-1)
    points = latlon[valid].T
    pix = hp.ang2pix(
        hp.npix2nside(len(var)), theta=points[0], phi=points[1], nest=True, lonlat=True
    )
    res = np.full(latlon.shape[:-1], np.nan, dtype=var.dtype)
    res[valid] = var[pix]
    return ax.imshow(res, extent=xlims + ylims, origin="lower", **kwargs)


def worldmap(var, **kwargs):
    cb_label = kwargs.pop("cb_label", None)

    projection = ccrs.Robinson(central_longitude=0)
    fig, ax = plt.subplots(
        figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
    )

    ax.set_global()
    im = nnshow_world(var, ax=ax, **kwargs)
    ax.add_feature(cf.COASTLINE, linewidth=0.8)
    ax.add_feature(cf.BORDERS, linewidth=0.4)
    cb = fig.colorbar(im)
    cb.set_label(cb_label)
    return fig, ax

def plot_lon_section(var, ax, **kwargs):

    var.plot.contourf(ax=ax, x="lat", y="zg", **kwargs)
    time = ax.get_title()
    lon = var.lon.values[0].round().astype(str)
    ax.set_title(time + ', Lon: ' + lon)
    ax.set_ylim([0, 20000])

def plot_lat_section(var, ax, **kwargs):

    var.plot.contourf(ax=ax, x="lon", y="zg", **kwargs)
    time = ax.get_title()
    lon = var.lat.values[0].round().astype(str)
    ax.set_title(time + ', Lat: ' + lon)
    ax.set_ylim([0, 20000])

