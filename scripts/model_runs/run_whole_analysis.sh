#!/bin/bash

# Set pythonpath
export PYTHONPATH="${PYTHONPATH}:/home/m/m301049/HcModel/"

# execute python script in respective environment 
/home/m/m301049/miniconda3/envs/defaultenv/bin/python /home/m/m301049/HcModel/scripts/process_data/process_atms_and_fluxes_mons.py
/home/m/m301049/miniconda3/envs/defaultenv/bin/python /home/m/m301049/HcModel/scripts/derive_model_relations/calculate_cre.py
/home/m/m301049/miniconda3/envs/defaultenv/bin/python /home/m/m301049/HcModel/scripts/derive_model_relations/calculate_emissivity.py
/home/m/m301049/miniconda3/envs/defaultenv/bin/python /home/m/m301049/HcModel/scripts/derive_model_relations/calculate_albedo.py
/home/m/m301049/miniconda3/envs/defaultenv/bin/python /home/m/m301049/HcModel/scripts/derive_model_relations/calculate_albedo_t_and_R_t.py
/home/m/m301049/miniconda3/envs/defaultenv/bin/python /home/m/m301049/HcModel/scripts/model_runs/icon_mons.py