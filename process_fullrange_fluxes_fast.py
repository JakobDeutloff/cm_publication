#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:35:51 2023

@author: u242031
"""

# %% imports

import os
from glob import glob
import multiprocessing as mp
import json
import numpy as np
from pyarts import xml

# %% get info


def get_info(filelist_of_configs):
    """get info about simulations from config files

    Parameters
    ----------
    filelist_of_configs : list of str
        list of config files

    Returns
    -------
    info : dict
        dictionary with info about simulations
    lat : numpy array
        latitudes
    lon : numpy array
        longitudes

    """

    info = {}
    info["f_grid_lw"] = []
    info["f_grid_sw"] = []
    info["lw_chunks"] = []
    info["sw_chunks"] = []

    freqtype = ["lw", "sw"]

    for i, config_file in enumerate(filelist_of_configs):
        with open(config_file, "r") as read_file:
            config = json.load(read_file)

        # print(config_file)

        results_folder = os.path.join(
            config["results_path"], config["identifier"])

        if i == 0:
            lat = xml.load(os.path.join(
                config["results_path"], "matrix_of_Lat.xml"))
            lon = xml.load(os.path.join(
                config["results_path"], "matrix_of_Lon.xml"))

        for ftype in freqtype:
            # for stype in skytype:

            # data, flag = Load_raw_data(results_folder, ftype, 'allsky', load_data=False)
            filelist = glob(os.path.join(
                results_folder, f"*allsky_{ftype}.xml"))

            if len(filelist) > 0:
                info[f"f_grid_{ftype}"] += config["f_grid"]
                # info[f"{ftype}_chunks"].append(int(config["identifier"][6:11]))
                info[f"{ftype}_chunks"].append(results_folder)

    for ftype in freqtype:
        info[f"f_grid_{ftype}"] = np.array(info[f"f_grid_{ftype}"])

        # check that f_grid ascending
        temp = np.sum(np.diff(info[f"f_grid_{ftype}"]) <= 0)

        if temp:
            raise RuntimeError("f_grid not ascending")

    return info, lat, lon


# %% get atm data

def get_atm_data(filelist_of_configs):

    for i, config_file in enumerate(filelist_of_configs):
        with open(config_file, "r") as read_file:
            config = json.load(read_file)

        found = False
        try:
            atm_data_file = config["atmdatafile"]
            aux_data_file = config["auxdatafile"]

            atm = xml.load(atm_data_file)
            aux = xml.load(aux_data_file)

            found = True

            return atm, aux

        except:
            continue

        if found == False:
            print('no atmdata found')

    return []


# %%integrate spectral fluxes


def get_fluxes(info, lat, lon, ftype, stype, fast_coeffs, prefix="spectral_irradiance_", fid=".xml",
               edge_spectra=False, poi=[]):
    """integrate spectral fluxes

    Parameters
    ----------
    info : dict
        info about simulations
    lat : numpy array
        latitudes
    lon : numpy array
        longitudes
    ftype : str
        frequency type
    stype : str
        sky type
    fast_coeffs: dict
        Dictionary with optimized frequencies and weights
    prefix : str, optional
        prefix of flux files. The default is "spectral_irradiance_".
    fid : str, optional
        file extension. The default is '.xml'.
    edge_spectra : bool, optional
        flag indicating if toa and sfc spectra should be processed. The default is False.
    poi : list
        list of lat,lon pairs. if not empty profiles for these point are processed.
        The defalu is an empty list

    Returns
    -------
    fluxes : numpy array
        integrated fluxes
    longitude : numpy array
        longitudes
    latitude : numpy array
        latitudes

    """

    chunks = info[f"{ftype}_chunks"]
    f_grid = info[f"f_grid_{ftype}"]

    df_start_index = 0


    fast_frqs=fast_coeffs[(f'f_grid_{ftype}')]
    fast_weights=fast_coeffs[(f'quadrature_weights_{ftype}')]



    for i, chunk_i in enumerate(chunks):
        filename = os.path.join(chunk_i, f"{prefix}{stype}_{ftype}{fid}")

        print("...loading\n")
        print(f"{filename}\n")
        print(f"chunk no. {i}")

        spc_irr_chnk = np.array(xml.load(filename))

        # number of frequencies in chunk
        n_f = np.size(spc_irr_chnk, 1)

        if i == 0:
            # allocate
            fluxes = np.zeros(
                (
                    np.size(spc_irr_chnk, 0),
                    np.size(spc_irr_chnk, 2),
                    np.size(spc_irr_chnk, 5),
                )
            )

        for j, freq_j in enumerate(f_grid[df_start_index:df_start_index+n_f]):

            logic = freq_j == fast_frqs

            if sum(logic):
                fluxes+=spc_irr_chnk[:,j,:,0,0,:]*fast_weights[logic]

        df_start_index += n_f



    # get number of lon and lats
    dlon = np.diff(lon, axis=0)
    number_of_lon = int(np.where(dlon < 0)[0][0] + 1)
    number_of_lat = int(len(lon) / number_of_lon)

    fluxes = np.reshape(
        fluxes, (number_of_lat, number_of_lon,
                 np.size(fluxes, 1), np.size(fluxes, 2))
    )

    longitude = np.reshape(lon, (number_of_lat, number_of_lon))
    latitude = np.reshape(lat, (number_of_lat, number_of_lon))


    return fluxes, longitude, latitude


# %% process
def process_fluxes(flx_cfg):
    """process fluxes

    Parameters
    ----------
    flx_cfg : dict
        config for flux calculation

    Returns
    -------
    result : dict
        dictionary with fluxes, latitudes and longitudes
    """

    fluxes, longitude, latitude = get_fluxes(
        flx_cfg["info"],
        flx_cfg["lat"],
        flx_cfg["lon"],
        flx_cfg["ftype"],
        flx_cfg["stype"],
        flx_cfg["fast_coeffs"]
    )

    result = {}
    result[f'{flx_cfg["stype"]}_{flx_cfg["ftype"]}'] = fluxes
    result["longitude"] = longitude
    result["latitude"] = latitude


    return result


def multiprocess_fluxes(configs, N_cpu=None):
    """multiprocess flux calculation

    Parameters
    ----------
    configs : list of dict
        list of configs for flux calculation
    N_cpu : int, optional
        number of cpus to use. The default is None.

    Returns
    -------
    result : list of dict
        list of dictionaries with fluxes, latitudes and longitudes

    """

    if N_cpu == None:
        N_cpu = mp.cpu_count()

        if N_cpu > 32:
            N_cpu = 32

        if N_cpu > len(configs):
            N_cpu = len(configs)

    pool = mp.Pool(N_cpu)

    result = pool.map(process_fluxes, configs)

    return result



def get_data(config_folder, fast_coeffs, serial=False):
    '''


    Parameters
    ----------
    config_folder : string
        path of the config folder.
    poi : list of 2-vector
        Points of interest, where full spectral information is exported.
    serial : boolean, optional
        Flag if serial or multiprocessing should be used. The default is False.

    Returns
    -------
    result : list of dict
        list of dictionaries with fluxes, latitudes and longitudes


    '''

    # get configs of simulations
    filelist = glob(f"{config_folder}/*.json")
    filelist.sort()

    # get info about simulation
    info, lat, lon = get_info(filelist)

    # load atmospheric data
    atms, aux = get_atm_data(filelist)

    # prepare configs for flux calculation, so that we can use multiprocess
    flx_cfgs = []
    for ftype in freqtype:
        for stype in skytype:
            temp = {}
            temp["info"] = info
            temp["ftype"] = ftype
            temp["stype"] = stype
            temp["lat"] = lat
            temp["lon"] = lon
            temp["filelist"] = filelist
            temp['fast_coeffs'] = fast_coeffs
            flx_cfgs.append(temp)

    if serial:
        # serial processing
        results=[]
        for flx_cfg in flx_cfgs:
            result=process_fluxes(flx_cfg)
            results.append(result)
    else:
        # parallel processing
        results = multiprocess_fluxes(flx_cfgs, N_cpu=4)

    return results, info


# %% start with main
if __name__ == "__main__":
    # %% constants/paths

    setup_name = "fullrange_flux_mid1deg_fast"
    config_folder = f"../configs/{setup_name}"


    fast_freq_path='../../arts_dev/arts-xml-data/planets/Earth/Optimized-Flux-Frequencies/' # Todo change to correct path, put to ForJakob folder
    fast_freq_lw_name='LW-flux-optimized-f_grid.xml'
    fast_freq_sw_name='SW-flux-optimized-f_grid.xml'
    fast_LW_quadrature_weights_name='LW-flux-optimized-quadrature_weights.xml'
    fast_SW_quadrature_weights_name='SW-flux-optimized-quadrature_weights.xml'

    # longitude for meridional section
    lon_sect = -20

    # lat/lon of POI
    # set list with position of interest
    poi = [np.array([0.0, 0.0]),np.array([50.,-10.])]

    freqtype = ["lw", "sw"]
    skytype = ["allsky", "clearsky"]

    # load frequencies for fast computation
    fast_coeffs={}
    fast_coeffs['f_grid_lw']=xml.load(os.path.join(fast_freq_path,fast_freq_lw_name))[:]
    fast_coeffs['f_grid_sw']=xml.load(os.path.join(fast_freq_path,fast_freq_sw_name))[:]
    fast_coeffs['quadrature_weights_lw']=xml.load(os.path.join(fast_freq_path,fast_LW_quadrature_weights_name))[:]
    fast_coeffs['quadrature_weights_sw']=xml.load(os.path.join(fast_freq_path,fast_SW_quadrature_weights_name))[:]

    results, info =get_data(config_folder, fast_coeffs, serial=False)