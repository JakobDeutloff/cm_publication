# %% import
import intake
import matplotlib.pyplot as plt
from src.healpix_functions import attach_coords
from src.map_plotfunctions import nnshow
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cmocean as cmo
from src.calc_variables import calc_LWP, calc_IWP, convert_to_density, calc_dry_air_properties
from matplotlib.colors import LogNorm
import numpy as np

# %% Load icon cycle 3 data
cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
ds = (
    cat.ICON["ngc3028"](zoom=10, time="PT3H", chunks="auto")
    .to_dask()
    .pipe(attach_coords)
    .sel(time="2023-03-20T00:00:00")
)

# %% convert hydrometeors to density
ds['rho_air'], ds['dry_air'] = calc_dry_air_properties(ds)
vars = ["qr", "clw", "cli", "qg", "qs"]
for var in vars:
    ds[var] = convert_to_density(ds, var)

# %% rename
ds = ds.rename({"qr": "rain", "clw": "LWC", "cli": "IWC", "qg": "graupel", "qs": "snow"})

# %% calculate LWP and IWP
ds["LWP"] = calc_LWP(ds)
ds["IWP"] = calc_IWP(ds)
ds = ds[["LWP", "IWP", "rsut", "rlut"]].load()

# %% plot   
# make colormap for IWP
colors = [
    (1, 1, 1, 0),
    (1, 1, 1, 1),
]  # Start with transparent white (alpha=0), end with solid white (alpha=1)
cmap_name = "transparent_white"
cm_iwp = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)
# make colormap for lwp with light blue like iwp but in light blue
colors = [
    (1, 1, 1, 0),
    (0, 0, 1, 1),
]  # Start with transparent white (alpha=0), end with solid white (alpha=1)
cmap_name = "transparent_blue"
cm_lwp = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)


# set up figure
projection = ccrs.PlateCarree(central_longitude=180)
fig, axes = plt.subplots(3, 1, figsize=(15, 8), subplot_kw={"projection": projection})
fig.set_facecolor('grey')

# set background of axes[0]
path = "/work/bm1183/m301049/pictures/"
bm = "BlueMarbleNG_2004-12-01_rgb_3600x1800.TIFF"
im = plt.imread(path + bm)
axes[0].imshow(im, origin="upper", extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree())

# plot data
for ax in axes:
    ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())

im_lwp = nnshow(
    ds["LWP"], ax=axes[0], cmap=cm_iwp, norm=LogNorm(vmin=1e-3, vmax=10)
)
im_iwp = nnshow(
    ds["IWP"], ax=axes[0], cmap=cm_lwp, norm=LogNorm(vmin=1e-3, vmax=10)
)

lw_im = nnshow(
    ds["rlut"],
    ax=axes[1],
    cmap=cmo.cm.thermal,
)

sw_im = nnshow(
    ds["rsut"],
    ax=axes[2],
    cmap="gray",
)


fig.subplots_adjust(right=0.85)
# make colorbars at right side of axes[0]
cax = fig.add_axes([0.87, 0.65, 0.02, 0.23])
cax.set_facecolor('grey')
cb = fig.colorbar(im_lwp, cax=cax, orientation='vertical', label="LWP / kgm$^{-2}$")

cax = fig.add_axes([0.95, 0.65, 0.02, 0.23])
cax.set_facecolor('grey')
cb = fig.colorbar(im_iwp, cax=cax, orientation='vertical', label="IWP / kgm$^{-2}$")

# make colorbar at right side of axes[1]
cax = fig.add_axes([0.87, 0.38, 0.02, 0.23])
cax.set_facecolor('grey')
cb = fig.colorbar(lw_im, cax=cax, orientation='vertical', label="LW up TOA / Wm$^{-2}$")

# make colorbar at right side of axes[2]
cax = fig.add_axes([0.87, 0.11, 0.02, 0.23])
cax.set_facecolor('grey')
cb = fig.colorbar(sw_im, cax=cax, orientation='vertical', label="SW up TOA / Wm$^{-2}$")

fig.savefig("plots/paper/eyecatcher.svg", bbox_inches="tight")
fig.savefig("plots/paper/eyecatcher.png", dpi=300, bbox_inches="tight")
# %%
