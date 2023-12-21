# %% import 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.read_data import load_atms_and_fluxes

# %% load data
atms, fluxes_3d, fluxes_3d_noice = load_atms_and_fluxes()
cre_binned, cre_interpolated, cre_average = load_cre()


