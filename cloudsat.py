#%%
import rioxarray as rxr
import gdal

# Path to the CloudSat file
path = '/work/bm1183/m301049/cloudsat/'

# %% open with gdal 
f1 = gdal.Open(path + 'ice/2014090182520_42155_CS_2C-ICE_GRANULE_P1_R05_E06_F00.hdf')

# %% open with rasterio - data seems to be missiong 
f1 = rxr.open_rasterio(filename=path + 'ice/2014090182520_42155_CS_2C-ICE_GRANULE_P1_R05_E06_F00.hdf')


