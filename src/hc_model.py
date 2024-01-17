# %% import
import numpy as np
import pandas as pd


# define functions to calculate model quantities


def logistic(x, L, x0, k, j):
    """
    Logistic function.

    PARAMETERS:
    ---------------------------
    x: array-like
        Input data.
    L: float
        Maximum value.
    x0: float
        Midpoint.
    k: float
        Steepness.
    j: float
        Offset.

    RETURNS:
    ---------------------------
    y: array-like
        Output data.
    """
    return L / (1 + np.exp(-k * (x - x0))) + j

def calc_lc_fraction(LWP, threshold):
    lc_fraction = (LWP >= threshold) * 1
    return lc_fraction

def binning(IWP_bins, data, IWP):
    return data.groupby_bins(IWP, IWP_bins).mean()


def calc_hc_albedo(IWP, alpha_hc_params):
    """
    Calculates the high-cloud albedo.

    PARAMETERS:
    ---------------------------
    IWP: array-like
        Input data.
    alpha_hc_params: tuple
        Parameters for the logistic function.

    RETURNS:
    ---------------------------
    fitted_vals: array-like
        high cloud albedo.
    """
    fitted_vals = logistic(np.log10(IWP), *alpha_hc_params, 0)
    return fitted_vals


def calc_hc_emissivity(IWP, em_hc_params):
    """
    Calculates the high-cloud emissivity.

    PARAMETERS:
    ---------------------------
    IWP: array-like
        Input data.
    em_hc_params: tuple
        Parameters for the logistic function.

    RETURNS:
    ---------------------------
    fitted_vals: array-like
        high cloud emissivity.
    """
    fitted_vals = logistic(np.log10(IWP), *em_hc_params, 0)
    return fitted_vals


def calc_alpha_t(
    LWP,
    lc_fraction,
    albedo_cs,
    alpha_t_params,
    const_lc_quantities,
    prescribed_lc_quantities,
):
    """
    Calculates the albdedo below the high clouds.

    PARAMETERS:
    ---------------------------
    LWP: array-like
        Liquid water path.
    lc_fraction: array-like
        low cloud fraction.
    albedo_cs: float
        Clearsky albedo.
    alpha_t_params: tuple
        Parameters for the logistic function.
    const_lc_quantities: dict, optional
        Constant values for lc quantities if they should be used instead of logistic function.

    RETURNS:
    ---------------------------
    avg_value: array-like
        Albedo below high clouds.
    """
    if const_lc_quantities is not None:
        lc_value = const_lc_quantities["a_t"]
    else:
        lc_value = logistic(np.log10(LWP), *alpha_t_params)
    cs_value = albedo_cs
    avg_value = lc_fraction * lc_value + (1 - lc_fraction) * cs_value
    if prescribed_lc_quantities is not None:
        avg_value = prescribed_lc_quantities["a_t"]
    return avg_value


def calc_R_t(
    LWP, IWP, lc_fraction, R_t_cs, R_t_params, h20_params, const_lc_quantities, prescribed_lc_quantities
):
    """
    Calculates the LW radiation from below the high clouds.

    PARAMETERS:
    ---------------------------
    LWP: array-like
        Liquid water path.
    lc_fraction: array-like
        low cloud fraction.
    R_t_cs: float
        Clearsky R_t.
    R_t_params: tuple
        Parameters for the linear regression.
    const_lc_quantities: dict, optional
        Constant values for lc quantities if they should be used instead of linear regression.

    RETURNS:
    ---------------------------
    avg_value: array-like
        LW radiation from below high clouds.
    """
    if const_lc_quantities is not None:
        lc_value = const_lc_quantities["R_t"]
    else:
        lc_value = R_t_params.slope * np.log10(LWP) + R_t_params.intercept
        lc_value[lc_value < R_t_cs] = R_t_cs
    h2o_correction = h20_params.slope * np.log10(IWP) + h20_params.intercept
    avg_value = lc_fraction * lc_value + (1 - lc_fraction) * R_t_cs + h2o_correction
    if prescribed_lc_quantities is not None:
        avg_value = prescribed_lc_quantities["R_t"]
    return avg_value


def hc_sw_cre(alpha_hc, alpha_t, SW_in):
    """
    Calculates the SW CRE of the high clouds.

    PARAMETERS:
    ---------------------------
    alpha_hc: array-like
        High cloud albedo.
    alpha_t: array-like
        Albedo below high clouds.
    SW_in: float
        Daily average SW radiation at TOA.

    RETURNS:
    ---------------------------
    SW_cre: array-like
        SW CRE of the high clouds.
    """
    return -SW_in * (
        alpha_hc + (alpha_t * (1 - alpha_hc) ** 2) / (1 - alpha_t * alpha_hc) - alpha_t
    )


def hc_lw_cre(em_hc, T_h, R_t, sigma):
    """
    Calculates the LW CRE of the high clouds.

    PARAMETERS:
    ---------------------------
    em_hc: array-like
        High cloud emissivity.
    T_h: array-like
        High cloud temperature.
    R_t: array-like
        LW radiation from below high clouds.
    sigma: float
        Stefan-Boltzmann constant.

    RETURNS:
    ---------------------------
    LW_cre: array-like
        LW CRE of the high clouds.
    """
    return em_hc * ((-1 * sigma * T_h**4) - R_t)


# model function


def run_model(
    IWP_bins,
    albedo_cs,
    R_t_cs,
    SW_in,
    T_hc,
    LWP,
    IWP,
    parameters,
    const_lc_quantities=None,
    prescribed_lc_quantities=None,
):
    """
    Runs the HC Model with given input data and parameters.

    INPUT:
    ---------------------------
    IWP_bins: array-like
        Bins for IWP.
    fluxes_3d: xarray.Dataset
        3D fluxes data.
    atms: xarray.Dataset
        Atmosphere data.
    lw_vars: xarray.Dataset
        Longwave variables.
    parameters: dict
        Parameters for the model.
    const_lc_quantities: dict, optional
        Constant values for lc quantities.

    RETURNS:
    ---------------------------
    results: pd.DataFrame
        Results of the model."""

    IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2

    # calculate lc fraction
    lc_fraction = calc_lc_fraction(LWP, threshold=parameters["threshold_lc_fraction"])

    # bin the input data
    T_hc_binned = binning(IWP_bins, T_hc, IWP)
    LWP_binned = binning(
        IWP_bins, LWP.where(LWP > parameters["threshold_lc_fraction"]), IWP
    )
    lc_fraction_binned = binning(IWP_bins, lc_fraction, IWP)

    # calculate radiative properties below high clouds
    alpha_t = calc_alpha_t(
        LWP_binned,
        lc_fraction_binned,
        albedo_cs,
        parameters["alpha_t"],
        const_lc_quantities,
        prescribed_lc_quantities,
    )
    R_t = calc_R_t(
        LWP_binned,
        IWP_points,
        lc_fraction_binned,
        R_t_cs,
        parameters["R_t"],
        parameters["h2o_dependence"],
        const_lc_quantities,
        prescribed_lc_quantities,
    )

    # calculate radiative properties of high clouds
    alpha_hc = calc_hc_albedo(IWP_points, parameters["alpha_hc"])
    em_hc = calc_hc_emissivity(IWP_points, parameters["em_hc"])

    # calculate HCRE
    SW_cre = hc_sw_cre(alpha_hc, alpha_t, SW_in)
    LW_cre = hc_lw_cre(em_hc, T_hc_binned, R_t, sigma=5.67e-8)

    # build results df
    results = pd.DataFrame()
    results.index = IWP_points
    results.index.name = "IWP"
    results["T_hc"] = T_hc_binned
    results["LWP"] = LWP_binned
    results["lc_fraction"] = lc_fraction_binned
    results["alpha_t"] = alpha_t
    results["R_t"] = R_t
    results["alpha_hc"] = alpha_hc
    results["em_hc"] = em_hc
    results["SW_cre"] = SW_cre
    results["LW_cre"] = LW_cre

    return results
