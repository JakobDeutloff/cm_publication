""" 
Functions to read data 
IMPORTANT: Update the path in the get_data_path function to the path where the data is stored.
"""

# %% import
import xarray as xr
import pickle


# %% load data

def get_data_path():
    """
    Get the path to the data.
    IMPORTANT: Update the path to your data pathr here.

    Returns
    -------
    str: Path to the data.
    """
    return "/work/bm1183/m301049/iwp_framework/publication/"

def load_icon_snapshot():
    """
    Load the ICON snapshot.

    Returns
    -------
    xarray.Dataset
        ICON snapshot.
    """
    path = get_data_path()
    return xr.open_dataset(path + "data/full_snapshot_proc.nc")

def load_atms_and_fluxes():
    """
    Load the data from the ICON-ARTS runs.

    Returns
    -------
    atms : xarray.Dataset
        Atmospheric variables.
    fluxes_allsky : xarray.Dataset
        Fluxes for all-sky conditions.
    fluxes_noice : xarray.Dataset
        Fluxes for no-ice conditions
    """
    path = get_data_path()
    fluxes_noice = xr.open_dataset(path + "data/fluxes_noice_proc.nc")
    fluxes_allsky = xr.open_dataset(path + "data/fluxes_allsky_proc.nc")
    atms = xr.open_dataset(path + "data/atms_proc.nc")
    
    return atms, fluxes_allsky, fluxes_noice


def load_cre():
    """
    Load the cloud radiative effect.

    Returns
    -------
    cre_binned : xarray.Dataset
        CRE binned by IWP and SW radiation.
    cre_average : xarray.Dataset
        CRE_interpolated averaged over SW radiation bins."""
    
    path = get_data_path()
    cre_binned = xr.open_dataset(path + "data/cre_binned.nc")
    cre_average = xr.open_dataset(path + "data/cre_mean.nc")
    return cre_binned, cre_average


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

    path = get_data_path()
    lw_vars = xr.open_dataset(path + "data/lw_vars.nc")
    sw_vars = xr.open_dataset(path + "data/sw_vars.nc")
    lc_vars = xr.open_dataset(path + "data/lower_trop_vars.nc")
    return lw_vars, sw_vars, lc_vars

def load_mean_derived_vars():
    """
    Load the mean derived variables.

    Returns
    -------
    lw_vars : xarray.Dataset
        Longwave variables.
    sw_vars : xarray.Dataset
        Shortwave variables.
    lc_vars : xarray.Dataset
        Low cloud variables.
    """

    path = get_data_path()
    with open(path + "data/lw_vars_mean.pkl", "rb") as f:
        lw_vars = pickle.load(f)
    with open(path + "data/sw_vars_mean.pkl", "rb") as f:
        sw_vars = pickle.load(f)
    with open(path + "data/lower_trop_vars_mean.pkl", "rb") as f:
        lc_vars = pickle.load(f)
    return lw_vars, sw_vars, lc_vars

def load_parameters():
    """
    Load the parameters needed for the model.

    Returns
    -------
    dict
        Dictionary containing the parameters.
    """

    path = get_data_path()
    with open(path + "parameters/hc_albedo_params.pkl", "rb") as f:
        hc_albedo = pickle.load(f)
    with open(path + "parameters/hc_emissivity_params.pkl", "rb") as f:
        hc_emissivity = pickle.load(f)
    with open(path + "parameters/C_h2o_params.pkl", "rb") as f:
        c_h2o = pickle.load(f)
    with open(path + "parameters/lower_trop_params.pkl", "rb") as f:
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

def load_model_output(name):
    """
    Load the model output.

    Parameters
    ----------
    name : str
        Name of the model output.

    Returns
    -------
    dict
        Model output.
    """

    path = get_data_path()
    with open(path + "model_output/" + name + ".pkl", "rb") as f:
        return pickle.load(f)

# %%
