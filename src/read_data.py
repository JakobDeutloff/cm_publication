# %% import 
import xarray as xr

# %% load data
def load_atms_and_fluxes():
    """
    Load the data from the ICON-ARTS runs.
    
    Returns
    -------
    atms : xarray.Dataset
        Atmospheric variables.
    fluxes_3d : xarray.Dataset
        Fluxes.
    fluxes_3d_noice : xarray.Dataset
        Fluxes without ice.
    """

    path = "/work/bm1183/m301049/icon_arts_processed/"
    run = "fullrange_flux_mid1deg_noice/"
    fluxes_3d_noice = xr.open_dataset(path + run + "fluxes_3d_full.nc")
    atms = xr.open_dataset(path + run + "atms_full.nc")
    run = "fullrange_flux_mid1deg/"
    fluxes_3d = xr.open_dataset(path + run + "fluxes_3d_full.nc")

    return atms, fluxes_3d, fluxes_3d_noice

def load_cre():
    path = '/work/bm1183/m301049/icon_arts_processed/derived_quantities/'
    cre_binned = xr.open_dataset(path + "cre_binned.nc")
    cre_interpolated = xr.open_dataset(path + "cre_interpolated.nc")
    cre_average = xr.open_dataset(path + "cre_interpolated_average.nc")
    return cre_binned, cre_interpolated, cre_average



