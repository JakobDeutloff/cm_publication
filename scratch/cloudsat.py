# %%
from typhon.files import FileSet, CloudSat
import xarray as xr

# %%Path to the CloudSat file
path = "/work/bm1183/m301049/cloudsat/"

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
# %%
data = collect_cloudsat("2019-07-01", "2019-07-02")
# start dataset: 2006-07-01 
# end dataset: 2019-07-01
# %%
data.to_netcdf('one_day_fwp.nc')
# %%
