# README

This repository contains the code for the paper "Insights on Tropical High-Cloud Radiative Effect from
a New Conceptual Mode" by Deutloff et al. (2024). The data is also made available under the DOI given in the paper. 

## Get stared 
The required packages are listed in the requirements.txt file. After installing them, preferably in a new environment, you should be able to execute the code. To correctly access the data, you have to specify the path to the data in the function get_data_path() defined in src/read_data.py. 

## Structure 
The repository is structured as follows:
- scripts: Contains the scripts used for the analysis and the plotting.
    - process_data: Contains the scripts used to process the data
    - derive_model_relations: Contains the scripts used to derive parameters of the conceptual model
    - model_runs: Contains the scripts used to run the conceptual model
    - plot: Contains the scripts used to plot the results
- src: Contains function definitions used in the scripts
