# %% import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from matplotlib.colors import LogNorm
from src.read_data import load_icon_snapshot

#%% load data
ds = load_icon_snapshot()
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


# %% set up figure
projection = ccrs.PlateCarree(central_longitude=180)
fig, ax = plt.subplots(1, 1, figsize=(15, 5), subplot_kw={"projection": projection})
fig.set_facecolor("white")

# set background of axes[0]
path = "/work/bm1183/m301049/pictures/"
bm = "BlueMarbleNG_2004-12-01_rgb_3600x1800.TIFF"
im = plt.imread(path + bm)
ax.imshow(im, origin="upper", extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree())
ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())

im_lwp = ax.imshow(ds['LWP'].values.squeeze(), extent=[-180, 180, -30, 30], cmap=cm_iwp, norm=LogNorm(vmin=1e-4, vmax=10))
im_iwp = ax.imshow(ds["IWP"].values.squeeze(),extent=[-180, 180, -30, 30], cmap=cm_lwp, norm=LogNorm(vmin=1e-4, vmax=10))


fig.subplots_adjust(bottom=0.8)
# make colorbars below ax
cax = fig.add_axes([0.1, 0.1, 0.35, 0.1])
cax.set_facecolor("grey")
cb = fig.colorbar(im_lwp, cax=cax, orientation="horizontal", label="LWP / kgm$^{-2}$")
cb.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

cax = fig.add_axes([0.55, 0.1, 0.35, 0.1])
cax.set_facecolor("grey")
cb = fig.colorbar(im_iwp, cax=cax, orientation="horizontal", label="IWP / kgm$^{-2}$")
cb.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

fig.tight_layout()
fig.savefig("plots/eyecatcher_monsoon.png", dpi=500, bbox_inches="tight")

# %%
