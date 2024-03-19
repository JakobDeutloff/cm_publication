# %% import
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import intake
from src.healpix_functions import attach_coords, sel_region
from src.calc_variables import calc_IWP, convert_to_density, calc_dry_air_properties, calc_cf
from math import log10, floor

# %% Load icon cycle 3 data
cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
ds = (
    cat.ICON["ngc3028"](zoom=10, time="PT3H", chunks="auto")
    .to_dask()
    .pipe(attach_coords)
    .sel(time="2023-03-20T00:00:00")
)
ds = sel_region(ds, -30, 30, 0, 360)

# %% convert hydrometeors to density
ds["rho_air"], ds["dry_air"] = calc_dry_air_properties(ds)
vars = ["cli", "qg", "qs"]
for var in vars:
    ds[var] = convert_to_density(ds, var)

# %% rename
ds = ds.rename({"qr": "rain", "clw": "LWC", "cli": "IWC", "qg": "graupel", "qs": "snow"})

# %% calculate cloud fraction
ds["cf"] = calc_cf(ds)

# %% calculate IWP and load data
ds["IWP"] = calc_IWP(ds)
ds_mem = ds[["IWP", "cf", "zg"]].load()

# %% interpolate from level_full to zg 
# ds_interp = ds.interp(coords={"level_full": ds["zg"], "cell":ds['cell']}, method="linear")

# %% group cf by IWP and average over percentile
quantiles = np.arange(0, 1.01, 0.01)
IWP_quantiles = ds_mem["IWP"].quantile(quantiles, dim=["cell"]).values
IWP_points = (IWP_quantiles[:-1] + IWP_quantiles[1:])/2
cf_quantiles = (
    ds_mem["cf"]
    .groupby_bins(ds_mem["IWP"], IWP_quantiles, labels=IWP_points)
    .mean(dim=["cell"])
)

# %% plot cloud occurence vs IWP percentiles
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

cf = ax.contourf(
    quantiles[1:] * 100, ds_mem.level_full, cf_quantiles, cmap="Blues", levels=np.arange(0.1, 1.1, 0.1)
)
ax.set_ylim(35, 90)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_xlabel("IWP percentiles")
ax.set_ylabel("Model Level")
ax.set_xlim(100, 40)
ax.set_xticks([100, 90, 80, 70, 60, 50, 40])

def round_to_1(x):
   return round(x, -int(floor(log10(abs(x)))))

# Create a second x-axis with IWP percentile labels
xticks = list(ax.get_xticks().astype(int))
labels2 = IWP_quantiles[xticks]
labels2 = [round_to_1(label) for label in labels2]
ax2 = ax.twiny()
ax2.set_xlim(100, 45)
ax2.set_xlabel("Percentile Values / kg/m$^2$")
ax2.set_xticks(xticks)
ax2.set_xticklabels(labels2) 

ax.spines[["top", "right"]].set_visible(False)
ax2.spines[["bottom", "right"]].set_visible(False)
fig.colorbar(cf, ax=ax, label="Cloud Cover")
fig.savefig("plots/paper/cloud_profile.svg", bbox_inches="tight")
fig.savefig("plots/paper/cloud_profile.png", dpi=300, bbox_inches="tight")


# %%
