# %%
import konrad 
import numpy as np
import matplotlib.pyplot as plt
from typhon import plots
from src.healpix_functions import attach_coords
import intake

# %% import nextgems data 
cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
ds = cat.ICON.ngc3028(time="PT3H", zoom=10, chunks="auto").to_dask().pipe(attach_coords)

# %% Run RRTMG with Konrad 
# pressure and half pressure levels 
plev, phlev = konrad.utils.get_pressure_grids(1000e2, 1, 201)

# atmosphere component 
atmosphere = konrad.atmosphere.Atmosphere(plev)

# It is possible to explicitly set different species (e.g. CO2).
atmosphere['CO2'][:] = 348e-6

# Create a surface component (T and p of the lowest atmosphere level are interpolated).
surface = konrad.surface.SlabOcean.from_atmosphere(atmosphere)

# Create cloud component (here clear-sky).
cloud = konrad.cloud.ClearSky.from_atmosphere(atmosphere)

# Setup the RRTMG radiation component (choose zenith angle and solar constant).
rrtmg = konrad.radiation.RRTMG(zenith_angle=47.88)

rrtmg.calc_radiation(atmosphere, surface, cloud)  # Actual RT simulation

# %% inspect output 
rrtmg.data_vars.keys()

# %%
fig, ax = plt.subplots()
plots.profile_p_log(atmosphere['phlev'], rrtmg['sw_flxu'][-1, :],
                    label='SW Up', color='skyblue', ls='solid')
plots.profile_p_log(atmosphere['phlev'], rrtmg['sw_flxd'][-1, :],
                    label='SW Down', color='skyblue', ls='dashed')
plots.profile_p_log(atmosphere['phlev'], rrtmg['lw_flxu'][-1, :],
                       label='LW Up', color='orangered', ls='solid')
plots.profile_p_log(atmosphere['phlev'], rrtmg['lw_flxd'][-1, :],
                    label='LW Down', color='orangered', ls='dashed')
ax.legend(loc='upper right')
ax.set_xlabel('Radiative flux [$\sf W/m^2$]')

# %%
