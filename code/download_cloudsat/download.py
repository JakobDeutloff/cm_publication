# %%
from typhon.files import FileSet, CloudSat
import xarray as xr
import sys 

# %% Path where to save the files 
path = "/work/bm1183/m301049/cloudsat/"

# get time delta 
time_start = sys.argv[1]
time_end = sys.argv[2]

# %%
def collect_cloudsat(start, end):
    cloudsat_files = FileSet(
        name="2C-ICE",
        path="/work/um0878/data/cloudsat/2C-ICE.P1_R05/{year}/{doy}/"
             "{year}{doy}{hour}{minute}{second}_*.hdf",
        handler=CloudSat(),
        # Each file of CloudSat covers exactly 5933 seconds. Since we state it
        # here, the indexing of files is much faster
        time_coverage="5933 seconds",
        # Load only the fields that we need:
        read_args={
            "fields": ["ice_water_path"],
        },
        max_threads=15,
    )

    print("Collect 2C-ICE...")
    data = xr.concat(
        cloudsat_files[start:end],

        dim="scnline"
    )

    return data
# %% read data
data = collect_cloudsat(time_start, time_end)

# %% save data 
data.to_netcdf(path + time_start + '_' + time_end + '_fwp.nc')
