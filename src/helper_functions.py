from src.healpix_functions import sel_region    

def hor_mean(data, mode="icon"):
    if mode == "icon":
        return data.mean(["cell", "time"])
    else:
        return data.sel(lat=slice(-30, 30)).mean(["lat", "lon"])