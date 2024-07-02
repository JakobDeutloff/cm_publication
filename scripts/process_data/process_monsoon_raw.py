# %% import
import xarray as xr
import numpy as np

# %% load data
ds_monsoon = xr.open_dataset(
    "/work/um0878/user_data/mbrath/StarARTS/Monsun_example_data/atm_fields.nc"
).sel(lat=slice(-30, 30))


# %% define functions
def calc_air_density(ds):
    ds = ds.assign(rho_air=lambda d: d.pfull / d.ta / 287.04)
    ds["rho_air"].attrs = {"units": "kg/m^3", "long_name": "dry air density"}
    ds = ds.assign(dry_air=lambda d: 1 - (d.cli + d.clw + d.qs + d.qg + d.qr + d.hus))
    ds["dry_air"].attrs = {"units": "kg/kg", "long_name": "specific mass of dry air"}
    return ds


def convert_hydrometeor_to_density(ds):
    for var in ["cli", "clw", "qs", "qr", "qg"]:
        ds[var] = (ds[var] / ds["dry_air"]) * ds["rho_air"]
        ds[var].attrs["units"] = "kg/m^3"
    return ds


def callc_iwp_lwp(ds):
    ds["IWP"] = ((ds["cli"] + ds["qs"] + ds["qg"]) * ds["dzghalf"]).sum("height")
    ds["IWP"].attrs = {"long_name": "Ice Water Path", "units": "kg/m^2"}
    ds["LWP"] = ((ds["clw"] + ds["qr"]) * ds["dzghalf"]).sum("height")
    ds["LWP"].attrs = {"long_name": "Liquid Water Path", "units": "kg/m^2"}
    return ds

def calc_height(ds):
    g = 9.81
    p_diff = ds["pfull"].diff("height")
    rho = ds["rho"].isel(height=slice(1, None))
    dz = p_diff / (g * rho)
    dz_top = xr.DataArray(
        data=dz.isel(height=0).values,
        dims=["time", "lat", "lon"],
        coords={
            "lat": ds["lat"],
            "lon": ds["lon"],
            "height": ds["height"][0],
        },
    )
    cell_height = xr.concat([dz_top, dz], dim="height") 
    ds["dzghalf"] = cell_height
    ds['geometric_height'] = cell_height.cumsum("height")
    return ds

def calculate_IWC_cumsum(atms):
    """
    Calculate the vertically integrated ice water content.
    """
    ice_mass = (atms["IWC"] + atms["graupel"] + atms["snow"]) * atms["dzghalf"]
    IWC_cumsum = ice_mass.cumsum("height")
    IWC_cumsum.attrs = {
        "units": "kg m^-2",
        "long_name": "Cumulative Ice Water Content",
    }
    return IWC_cumsum


def calculate_h_cloud_temperature(atms, IWP_emission=6.7e-3):
    """
    Calculate the temperature of high clouds.
    """
    top_idx_thin = (atms["IWC"] + atms["snow"] + atms["graupel"]).argmax("height")
    top_idx_thick = np.abs(atms["IWC_cumsum"] - IWP_emission).argmin("height")
    top_idx = xr.where(top_idx_thick < top_idx_thin, top_idx_thick, top_idx_thin)
    top = atms.isel(height=top_idx).height
    T_h = atms["temperature"].sel(height=top)
    T_h.attrs = {"units": "K", "long_name": "High Cloud Top Temperature"}
    top.attrs = {"units": "1", "long_name": "Level Index of High CLoud Top"}
    return T_h, top

# %% calculate variables 
ds_monsoon = calc_air_density(ds_monsoon)
ds_monsoon = convert_hydrometeor_to_density(ds_monsoon)
# %%
ds_monsoon = calc_height(ds_monsoon)
ds_monsoon = callc_iwp_lwp(ds_monsoon)

# %% rename 
ds_monsoon = ds_monsoon.rename(
    {
        'qs': 'snow',
        'qr': 'rain',
        'cli': 'IWC', 
        'clw': 'LWC',
        'qg': 'graupel',
        'pfull': 'pressure',
        'ta': 'temperature',
    }
)

# %% calculate IWC_cumsum and high cloud temperature
ds_monsoon["IWC_cumsum"] = calculate_IWC_cumsum(ds_monsoon)
ds_monsoon["hc_temperature"], ds_monsoon["hc_top_index"] = calculate_h_cloud_temperature(ds_monsoon)
ds_monsoon['mask_height'] = ds_monsoon.sel(height=ds_monsoon["hc_top_index"])["pressure"] < 35000

# %% save
ds_monsoon.to_netcdf("/work/bm1183/m301049/iwp_framework/mons/data/full_snapshot_proc.nc")

# %%
