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
    atms = xr.open_dataset("/work/bm1183/m301049/nextgems_profiles/monsoon/atms_full.nc")
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
    lc_vars = xr.open_dataset(path + "lc_vars.nc")
    return lw_vars, sw_vars, lc_vars


def load_binned_derived_variables():
    """
    Load the binned and averaged derived variables.

    Returns
    -------
    lw_vars : pd:Dataframe
        Longwave variables.
    sw_vars : pd.Dataframe
        Shortwave variables.
    lc_vars : pd.Dataframe
        Low cloud variables.
    """

    path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"
    with open(path + "mean_lw_vars.pkl", "rb") as f:
        lw_vars = pickle.load(f)
    with open(path + "mean_sw_vars.pkl", "rb") as f:
        sw_vars = pickle.load(f)
    with open(path + "mean_lc_vars.pkl", "rb") as f:
        lc_vars = pickle.load(f)
    return lw_vars, sw_vars, lc_vars


def load_parameters():
    """
    Load the parameters needed for the model.

    Returns
    -------
    hc_albedo : dict
        Parameters for the high cloud albedo.
    hc_emissivity : dict
        Parameters for the high cloud emissivity.
    alpha_t : dict
        Parameters for the LWP dependence of R_t.
    R_t : dict
        Parameters for the LWP dependence of alpha_t."""

    path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"
    with open(path + "hc_albedo_params.pkl", "rb") as f:
        hc_albedo = pickle.load(f)
    with open(path + "hc_emissivity_params.pkl", "rb") as f:
        hc_emissivity = pickle.load(f)
    with open(path + "alpha_t_params.pkl", "rb") as f:
        alpha_t = pickle.load(f)
    with open(path + "R_t_params.pkl", "rb") as f:
        R_t = pickle.load(f)
    with open(path + "water_vapor_dependence.pkl", "rb") as f:
        h2o_dependence = pickle.load(f)

    return {
        "alpha_hc": hc_albedo,
        "em_hc": hc_emissivity,
        "alpha_t": alpha_t,
        "R_t": R_t,
        "h2o_dependence": h2o_dependence,
    }

def load_average_lc_parameters():
    """
    Load the average lc parameters needed for the model.
    
    Returns
    -------
    average_lc_params : dict
        Average parameters for a_t and R_t."""

    path = "/work/bm1183/m301049/icon_arts_processed/derived_quantities/"
    with open(path + "average_a_t_r_t.pkl", "rb") as f:
        average_lc_params = pickle.load(f)

    return average_lc_params

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
