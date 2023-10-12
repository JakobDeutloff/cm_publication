# %%
import pickle
import xarray as xr 
import matplotlib.pyplot as plt
import pyarts

# %% load data 
path = '/work/bm1183/m301049/freddi_runs/'

atms = pickle.load(open(path + 'atms.pkl', 'rb'))
results = pickle.load(open(path + 'results.pkl', 'rb'))
aux = pickle.load(open(path + 'aux.pkl', 'rb'))
info = pickle.load(open(path + 'info.pkl', 'rb'))

# %% convert arts arrays to xarray 
atms = atms.to_dict()
aux = aux.to_dict()

# %% build results xarray
lat = results[0]['latitude']
lon = results[0]['longitude']
pressure_levels = atms['grid2'] 
height = atms['Data']





