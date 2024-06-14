# %% import
import xarray as xr
import pickle


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
    atms = xr.open_dataset("/work/bm1183/m301049/icon_arts_processed/fullrange_flux_mid1deg/atms_full.nc")
    run = "fullrange_flux_mid1deg/"
    fluxes_3d = xr.open_dataset(path + run + "fluxes_3d_full.nc")

    return atms, fluxes_3d, fluxes_3d_noice


def load_cre():
    """
    Load the cloud radiative effect.

    Returns
    -------
    cre_binned : xarray.Dataset
        CRE binned by IWP and SW radiation.
    cre_interpolated : xarray.Dataset
        CRE interpolated over IWP and SW radiation bins.
    cre_average : xarray.Dataset
        CRE_interpolated averaged over SW radiation bins."""

    path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"
    cre_binned = xr.open_dataset(path + "cre_binned.nc")
    cre_interpolated = xr.open_dataset(path + "cre_interpolated.nc")
    cre_average = xr.open_dataset(path + "cre_interpolated_average.nc")
    return cre_binned, cre_interpolated, cre_average


def load_derived_vars():
    """
    Load the gridded derived variables.

    Returns
    -------
    lw_vars : xarray.Dataset
        Longwave variables.
    sw_vars : xarray.Dataset
        Shortwave variables.
    lc_vars : xarray.Dataset
        Low cloud variables.
    """

    path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"
    lw_vars = xr.open_dataset(path + "lw_vars.nc")
    sw_vars = xr.open_dataset(path + "sw_vars.nc")
    lc_vars = xr.open_dataset(path + "lower_trop_vars.nc")
    return lw_vars, sw_vars, lc_vars

def load_binned_derived_variables():
    """
    Load the binned derived variables.

    Returns
    -------
    lw_vars_avg : xarray.Dataset
        Longwave variables binned.
    sw_vars_avg : xarray.Dataset
        Shortwave variables binned.
    lower_trop_vars_avg : xarray.Dataset
        Lower Troposphere variables binned.
    """

    path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"
    with open(path + "mean_sw_vars.pkl", "rb") as f:
        sw_vars_avg = pickle.load(f)
    with open(path + "mean_lw_vars.pkl", "rb") as f:
        lw_vars_avg = pickle.load(f)
    with open(path + "mean_lower_trop_vars.pkl", "rb") as f:
        lower_trop_vars_avg = pickle.load(f)
    return lw_vars_avg, sw_vars_avg, lower_trop_vars_avg

def load_parameters():
    """
    Load the parameters needed for the model.

    Returns
    -------
    hc_albedo : dict
        Parameters for the high cloud albedo.
    hc_emissivity : dict
        Parameters for the high cloud emissivity.
    """

    path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"
    with open(path + "hc_albedo_params.pkl", "rb") as f:
        hc_albedo = pickle.load(f)
    with open(path + "hc_emissivity_params.pkl", "rb") as f:
        hc_emissivity = pickle.load(f)
    with open(path + "C_h2o.pkl", "rb") as f:
        c_h2o = pickle.load(f)
    with open(path + "lower_trop_params.pkl", "rb") as f:
        lower_trop_params = pickle.load(f)

    return {
        "alpha_hc": hc_albedo,
        "em_hc": hc_emissivity,
        "c_h2o": c_h2o,
        "R_l": lower_trop_params["R_l"],
        "R_cs": lower_trop_params["R_cs"],
        "f": lower_trop_params["f"],
        "a_l": lower_trop_params["a_l"],
        "a_cs": lower_trop_params["a_cs"],
    }

def load_binned_atms():
    """
    Load the binned atmospheric variables.

    Returns
    -------
    atms_binned : xarray.Dataset
        Binned atmospheric variables.
    """

    path = "/work/bm1183/m301049/nextgems_profiles/"
    atms_binned = xr.open_dataset(path + "profiles_processed_2.nc")
    return atms_binned
