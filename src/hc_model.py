# %% import
import numpy as np
import pandas as pd


# define functions to calculate model quantities


def calc_constants(fluxes_3d):
    albedo_cs = fluxes_3d["albedo_clearsky"].sel(lat=slice(-30, 30)).mean()
    R_t_cs = (
        fluxes_3d["clearsky_lw_up"].isel(pressure=-1).sel(lat=slice(-30, 30)).mean()
    )
    SW_in = (
        fluxes_3d["clearsky_sw_down"]
        .isel(pressure=-1)
        .sel(lat=slice(-30, 30))
        .mean()
        .values
    )
    return albedo_cs, R_t_cs, SW_in


def logisic(x, L, x0, k, j):
    return L / (1 + np.exp(-k * (x - x0))) + j


def calc_hc_temperature(IWP_bins, lw_vars, atms):
    T_hc_binned = (
        lw_vars["h_cloud_temperature"]
        .sel(lat=slice(-30, 30))
        .groupby_bins(atms["IWP"].sel(lat=slice(-30, 30)), IWP_bins)
        .mean()
    )
    return T_hc_binned


def calc_LWP(IWP_bins, atms):
    LWP_binned = (
        atms["LWP"]
        .where(atms["LWP"] > 1e-6)
        .sel(lat=slice(-30, 30))
        .groupby_bins(atms["IWP"].sel(lat=slice(-30, 30)), IWP_bins)
        .mean()
    )
    return LWP_binned


def calc_lc_fraction(IWP_bins, atms):
    lc_fraction_binned = (
        atms["lc_fraction"]
        .sel(lat=slice(-30, 30))
        .groupby_bins(atms["IWP"].sel(lat=slice(-30, 30)), IWP_bins)
        .mean()
    )
    return lc_fraction_binned


def calc_hc_albedo(IWP, alpha_hc_params):
    fitted_vals = logisic(np.log10(IWP), *alpha_hc_params, 0)
    return fitted_vals


def calc_hc_emissivity(IWP, em_hc_params):
    fitted_vals = logisic(np.log10(IWP), *em_hc_params, 0)
    return fitted_vals


def calc_alpha_t(LWP, lc_fraction, albedo_cs, alpha_t_params):
    lc_value = logisic(np.log10(LWP), *alpha_t_params)
    cs_value = albedo_cs
    avg_value = lc_fraction * lc_value + (1 - lc_fraction) * cs_value
    return avg_value


def calc_R_t(LWP, lc_fraction, R_t_cs, R_t_params):
    lc_value = R_t_params.slope * LWP + R_t_params.intercept
    lc_value[lc_value < R_t_cs] = R_t_cs
    avg_value = lc_fraction * lc_value + (1 - lc_fraction) * R_t_cs
    return avg_value


def hc_sw_cre(alpha_hc, alpha_t, SW_in):
    return -SW_in * (
        alpha_hc + (alpha_t * (1 - alpha_hc) ** 2) / (1 - alpha_t * alpha_hc) - alpha_t
    )


def hc_lw_cre(em_hc, T_h, R_t, sigma):
    return em_hc * ((-1 * sigma * T_h**4) - R_t)


# model function


def run_model(IWP_bins, fluxes_3d, atms, lw_vars, parameters):
    IWP_points = (IWP_bins[1:] + IWP_bins[:-1]) / 2

    # calculate constants
    albedo_cs, R_t_cs, SW_in = calc_constants(fluxes_3d)

    # calculate model quantities
    T_hc = calc_hc_temperature(IWP_bins, lw_vars, atms)
    LWP = calc_LWP(IWP_bins, atms)
    lc_fraction = calc_lc_fraction(IWP_bins, atms)
    alpha_t = calc_alpha_t(LWP, lc_fraction, albedo_cs, parameters["alpha_t"])
    R_t = calc_R_t(LWP, lc_fraction, R_t_cs, parameters["R_t"])
    alpha_hc = calc_hc_albedo(IWP_points, parameters["alpha_hc"])
    em_hc = calc_hc_emissivity(IWP_points, parameters["em_hc"])

    # calculate HCRE
    SW_cre = hc_sw_cre(alpha_hc, alpha_t, SW_in)
    LW_cre = hc_lw_cre(em_hc, T_hc, R_t, sigma=5.67e-8)

    # build results df
    results = pd.DataFrame()
    results.index = IWP_points
    results.index.name = "IWP"
    results["T_hc"] = T_hc
    results["LWP"] = LWP
    results["lc_fraction"] = lc_fraction
    results["alpha_t"] = alpha_t
    results["R_t"] = R_t
    results["alpha_hc"] = alpha_hc
    results["em_hc"] = em_hc
    results["SW_cre"] = SW_cre
    results["LW_cre"] = LW_cre

    return results
