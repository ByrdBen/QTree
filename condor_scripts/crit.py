import os
import pickle
import numpy as np
import re
from qtree.pruning import *

# Path to where you've saved data
load_folder = r""
# Path to where you'd like processed data to be saved
save_folder = r""
folders = os.listdir(load_folder)

# Defining Hilbert space dimensions for proper indexing
# These should be loaded automatically somehow in the future
qdim = 30
rdim = 60
cdim = 6

# Initialize results dictionary
results = {}

# Each flux sweep lives in it's own folder. Assuming they're
# batched together:
for folder in folders:
    path_full = os.path.join(load_folder, folder)
    # Find and sort all files in subfolder
    filenames = sorted(f for f in os.listdir(path_full) if f.endswith(".pkl"))
    # Initialize empty dict for given folder (flux sweep)
    results[folder] = {}
    # Initialize list to store names of broken datafiles
    skipped_files = []

    for name in filenames:
        try:
            # Load pickled datafile (will need to swap to HDF5 in the future)
            with open(os.path.join(path_full, name), "rb") as f:
                d = pickle.load(f)
        #Place broken filenames in skipped_files and continue
        except (pickle.UnpicklingError, EOFError):
            skipped_files.append(name)
            continue
        # Find flux point for indexing results
        flux = float(re.search(r"res_flux_([\d.]+)\.pkl", name).group(1))
        # Extract relevant info
        PCA_list = d['PCA_list'][0]
        map_list = d['map_list'][0]
        # Store data tuple, indexed by folder/flux 
        results[folder][flux] = get_transitions(PCA_list, map_list, qdim, rdim, cdim)
        del d
    # Condor output to see how processing is progressing
    if skipped_files:
        print(f"{folder}: skipped {skipped_files}")
        
# Save all filtered data in save_folder
with open(os.path.join(save_folder, "transitions.pkl"), "wb") as f:
    pickle.dump(results, f)

        