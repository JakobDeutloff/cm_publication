# %%
import os
import numpy as np

# %%
start = "2006-07-01"
end = "2019-07-01"
time_increments = [f"200{i}-07-01" for i in range(6, 10)] + [f"20{i}-07-01" for i in range(10, 20)]
path = os.getcwd()

# %%
for i in range(len(time_increments)-1):
    start_time = time_increments[i]
    end_time = time_increments[i+1]
    print("-----node line -----")
    os.system(
        f"sbatch {path}/download_cloudsat/submitter.sh {start_time} {end_time}"
    )