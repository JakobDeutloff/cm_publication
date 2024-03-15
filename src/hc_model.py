# %% import
import numpy as np
import pandas as pd
import xarray as xr

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


def calc_lc_fraction(LWP, connected, fix_val, threshold=1e-4):
    """
    Calculates the low cloud fraction.

    PARAMETERS:
    ---------------------------
    LWP: array-like
        Liquid water path.
    threshold: float
        Threshold value for low cloud fraction.
    connected: array-like
        Connectedness data.

    RETURNS:
    ---------------------------
    lc_fraction: array-like
        Low cloud fraction.
    """
    if connected:
        lc_fraction = ((LWP >= threshold) & (connected != 1)) * 1
    else:
        lc_fraction = xr.DataArray(data=np.ones_like(LWP) * fix_val, dims=LWP.dims, coords=LWP.coords)
    return lc_fraction


def binning(IWP_bins, data, IWP):
    """
    Bins the data based on IWP.

    PARAMETERS:
    ---------------------------
    IWP_bins: array-like
        Bins for IWP.
    data: array-like
        Data to be binned.
    IWP: array-like
        Corresponding IWP.

    RETURNS:
    ---------------------------
    binned_data: array-like
        Binned data.
    """
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
        High cloud albedo.
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
        High cloud emissivity.
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
    Calculates the albedo below the high clouds.

    PARAMETERS:
    ---------------------------
    LWP: array-like
        Liquid water path.
    lc_fraction: array-like
        Low cloud fraction.
    albedo_cs: float
        Clearsky albedo.
    alpha_t_params: tuple
        Parameters for the logistic function.
    const_lc_quantities: dict, optional
        Constant values for lc quantities if they should be used instead of logistic function.
    prescribed_lc_quantities: dict, optional
        Prescribed values for lc quantities.

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
    LWP,
    IWP,
    lc_fraction,
    R_t_cs,
    R_t_params,
    h20_params,
    const_lc_quantities,
    prescribed_lc_quantities,
):
    """
    Calculates the LW radiation from below the high clouds.

    PARAMETERS:
    ---------------------------
    LWP: array-like
        Liquid water path.
    IWP: array-like
        Ice water path.
    lc_fraction: array-like
        Low cloud fraction.
    R_t_cs: float
        Clearsky R_t.
    R_t_params: tuple
        Parameters for the linear regression.
    h20_params: tuple
        Parameters for the linear regression.
    const_lc_quantities: dict, optional
        Constant values for lc quantities if they should be used instead of linear regression.
    prescribed_lc_quantities: dict, optional
        Prescribed values for lc quantities.

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
    return avg_value, h2o_correction


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

def ac_sw_cre(alpha_cs, alpha_t, alpha_hc, SW_in):
    """
    Calculates the SW CRE of the entire cloud population (hc + lc).

    PARAMETERS:
    ---------------------------
    alpha_cs: float
        Clearsky albedo.
    alpha_t: array-like
        Albedo below high clouds.
    alpha_hc: array-like
        High cloud albedo.
    SW_in: float
        Daily average SW radiation at TOA.

    RETURNS:
    ---------------------------
    SW_cre: array-like
        SW CRE of the high clouds.
    """
    return -SW_in * (alpha_hc + (alpha_t * (1 - alpha_hc) ** 2) / (1 - alpha_t * alpha_hc) - alpha_cs)

def ac_lw_cre(emm_hc, T_h, R_t, R_cs, h20_corr, sigma):
    """
    Calculates the LW CRE of the entire cloud population (hc + lc).

    PARAMETERS:
    ---------------------------
    em_hc: array-like
        High cloud emissivity.
    T_h: array-like
        High cloud temperature.
    R_t: array-like
        LW radiation from below high clouds.
    R_cs: float
        Clearsky LW radiation.
    h20_corr: array-like
        Correction for water vapor.
    sigma: float
        Stefan-Boltzmann constant.

    RETURNS:
    ---------------------------
    LW_cre: array-like
        LW CRE of the high clouds.
    """
    return  (1 - emm_hc) * R_t - emm_hc * sigma * T_h**4 - (R_cs + h20_corr)


def run_model(
    IWP_bins,
    albedo_cs,
    R_t_cs,
    SW_in,
    T_hc,
    LWP,
    IWP,
    connectedness,
    parameters,
    const_lc_quantities=None,
    prescribed_lc_quantities=None,
):
    """
    Runs the HC Model with given input data and parameters.

    PARAMETERS:
    ---------------------------
    IWP_bins: array-like
        Bins for IWP.
    albedo_cs: float
        Clearsky albedo.
    R_t_cs: float
        Clearsky R_t.
    SW_in: float
        Daily average SW radiation at TOA.
    T_hc: array-like
        High cloud temperature.
    LWP: array-like
        Liquid water path.
    IWP: array-like
        Ice water path.
    connectedness: array-like
        Connectedness data.
    parameters: dict
        Parameters for the model.
    const_lc_quantities: dict, optional
        Constant values for lc quantities.
    prescribed_lc_quantities: dict, optional
        Prescribed values for lc quantities.

    RETURNS:
    ---------------------------
    results: pd.DataFrame
        Results of the model.
    """

    IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2

    # calculate lc fraction
    lc_fraction = calc_lc_fraction(LWP, connected=connectedness, fix_val=parameters["lc_fraction"])

    # bin the input data
    T_hc_binned = binning(IWP_bins, T_hc, IWP)
    LWP_binned = binning(IWP_bins, LWP.where(LWP > 1e-4), IWP)
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
    R_t, h2o_corr = calc_R_t(
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

    # calculate CRE of entire cloud population
    SW_cre_ac = ac_sw_cre(albedo_cs, alpha_t, alpha_hc, SW_in)
    LW_cre_ac = ac_lw_cre(em_hc, T_hc_binned, R_t, R_t_cs, h2o_corr, sigma=5.67e-8)

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
    results["SW_cre_ac"] = SW_cre_ac
    results["LW_cre_ac"] = LW_cre_ac

    return results
