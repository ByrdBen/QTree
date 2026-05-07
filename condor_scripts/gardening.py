import os
import pickle
import numpy as np
import re
"""
This function has quickly become a memory bottleneck. Some quick digging 
suggests pre-allocating the memory, and populating the output dict 
one entry at a time is more effecient than loading and manipulating
all the data at once. Hoping this will fix it.
"""

# Folder where you saved the data
load_folder = r""
# Folder where you'd like results to be saved
save_folder = r""
folders = os.listdir(load_folder)

for folder in folders:
    path_full = os.path.join(load_folder, folder)
    res_list_loaded = []
    # Grab list of all files in subfolder
    filenames = sorted(
        f for f in os.listdir(path_full)
        if f.endswith(".pkl")
    )

    # List of relevant substrings (computational subspace)
    string_list = ["q0_", "q1_"]
    n_files = len(filenames)
    # Initialize empty list to contain skipped/corrupted datafiles
    skipped_files = []
    # Check first data file to create template to save on memory allocation
    first_dict = None
    for name in filenames:
        try:
            with open(os.path.join(path_full, name), "rb") as f:
                first_dict = pickle.load(f)
            print(f"Using {name} as template")
            break
        except (pickle.UnpicklingError, EOFError):
            skipped_files.append(name)
    
    # Grab useful keys
    keys = [
        k for k in first_dict.keys()
        if any(p in k for p in string_list) and re.search(r"_c\d+", k)]
    
    # Define shape
    array_shape = first_dict[keys[0]][0].shape
    
    # Pre-allocate space in output dict 
    output = {
        k: np.zeros((n_files, *array_shape), dtype=first_dict[k][0].dtype)
        for k in keys
    }
    
    # Open and store each individual data set into output, then delete from memory
    for i, name in enumerate(filenames):
        filename = os.path.join(path_full, name)
        
        try:
            with open(filename, "rb") as f:
                d = pickle.load(f)
                
        except (pickle.UnpicklingError, EOFError) as e:
            skipped_files.append(name)
            continue
            
        if all(key in d for key in keys):
            for k in keys:
                output[k][i] = d[k][0]
        else:
            skipped_files.append(name)
        del d
        
    dat1 = {}
    dat0 = {}
    
    # Sort data by qubit branch
    for key in output.keys():
        if 'q0_' in key:
            dat0[key] = output[key]
        if 'q1_' in key:
            dat1[key] = output[key]
            
    filename = os.path.join(save_folder, f"{folder}_0.pkl")
    with open(filename, "wb") as f:
        pickle.dump(dat0, f)
    
    filename = os.path.join(save_folder, f"{folder}_1.pkl")
    with open(filename, "wb") as f:
        pickle.dump(dat1, f)
    
    # Write to text file
    filename = os.path.join(save_folder, f"{folder}_skipped_points.txt")
    with open(filename, "w") as f:
        for name in skipped_files:
            line = name
            f.write(line + "\n")
    print(str(folder) + ' has been tended to!')