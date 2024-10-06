import numpy as np
import glob
import re
import os
import tables
import pandas as pd


def check_files_exist(global_variables):
    array_datasteps_files = [1]
    if global_variables["path_dl2"] != None:
        array_datasteps_files.append(2)
    if global_variables["path_dl3"] != None:
        array_datasteps_files.append(3)
        
    for file in [f"path_dl{i}" for i in array_datasteps_files]:
        # Check that files exist
        if not (os.path.isfile(global_variables[file])):
            raise FileNotFoundError(f"Input DL{file[-1:]} file ({global_variables[file]}) not found.")
    
def open_files(global_variables, key_dl1, key_dl2):
    # Checking if all files exist
    check_files_exist(global_variables=global_variables) 
    
    # Opening the tables for DL1 and DL2 as well as for the datacheck
    table_dl1_datacheck = tables.open_file(global_variables["path_dl1_dcheck"]).root.dl1datacheck

    table_dl1 = pd.read_hdf(global_variables[f"path_dl1"], key_dl1)
    
    if global_variables[f"path_dl2"] != None:
        table_dl2 = pd.read_hdf(global_variables[f"path_dl2"], key_dl2)
    else:
        table_dl2 = None
        
    return table_dl1_datacheck, table_dl1, table_dl2
    
def find_dl1_fname(run_number, dchecking=False, version_string="v*", return_version=False, print_details=True):
    str_dchecks = "" if not dchecking else "datacheck_"

    # Root location in IT cluster for DL1 data and the filename
    root_dl1 = f"/fefs/aswg/data/real/DL1/*/{version_string}/tailcut84/"
    fname_dl1_runwise = f"{str_dchecks}dl1_LST-1.Run{run_number:05}.h5"
    # Finding all DL1 files corresponding to the provided run number
    files_dl1a_runwise = np.sort(glob.glob(root_dl1 + "*/" + fname_dl1_runwise) + glob.glob(root_dl1 + fname_dl1_runwise))

    # Checking runs we have, not, or we have duplicated
    if len(files_dl1a_runwise) == 0:
        raise ValueError(f"Run {run_number:5} not found in {root_dl1}")
    
    elif len(files_dl1a_runwise) > 1:
        print(f"DL1: Run {run_number:5} presented {len(files_dl1a_runwise)} different versions:") if print_details else None
        
        str_versions, versions, lengths_versions = [], [], []
        for i, runfile in enumerate(files_dl1a_runwise):
            
            str_version = runfile.split("/")[7][1:]   # Getting the version string e.g. "0.10.1_test3"
            str_parts = re.split("\.|_", str_version) # Splitting in parts e.g. ["0", "10", "1", "test3"]
            # Then we extract as float, whenever there is only digits
            str_parts_float = [float(part) for part in str_parts if part.isdigit()]

            # Then we construct a float number associated to each version
            for ii, part in enumerate(str_parts_float):
                final_float_str = f"{part:04.0f}." if ii == 0 else final_float_str + f"{part:04.0f}"

            str_versions.append(f"v{str_version}")
            versions.append(float(final_float_str)) 
            lengths_versions.append(len(str_version))
    
        version_index = 0
        for i in range(1, len(versions)):
            condition_larger_float = versions[i] > versions[version_index]
            condition_get_shorter  = (versions[i] == versions[version_index] and lengths_versions[i] < lengths_versions[version_index])
            if condition_larger_float or condition_get_shorter:
                version_index = i
        
        for i, runfile in enumerate(files_dl1a_runwise):
            str_selected = "<-- (SELECTED)" if i == version_index else ""
            print(f"* {str_versions[i]} {str_selected}") if print_details else None
        
        final_fname   = files_dl1a_runwise[version_index]
        final_version = str_versions[version_index]
    
    else:
        final_fname   = files_dl1a_runwise[0]
        final_version = final_fname.split("/")[7]
        print(f"DL1 file version: {final_version}") if print_details else None

    if return_version:
        return final_fname, final_version
    else:
        return final_fname
    
    
def find_r0_fname(run_number, srun_number, index_writer=1, version_string="v*", return_version=False, print_details=True):

    # Root location in IT cluster for R0 data and the filename
    root_r0 = f"/fefs/aswg/data/real/R0/*/"
    fname_r0_srunwise = f"LST-1.{index_writer}.Run{run_number:05}.{srun_number:04}.fits.fz"
    # Finding all R0 files corresponding to the provided run number
    files_r0_srunwise = np.sort(glob.glob(root_r0 + "*/" + fname_r0_srunwise) + glob.glob(root_r0 + fname_r0_srunwise))

    # Checking runs we have, not, or we have duplicated
    if len(files_r0_srunwise) == 0:
        raise ValueError(f"Run {run_number:5} not found in {root_r0}")
    
    final_fname   = files_r0_srunwise[0]
    
    return final_fname

    
    
def straight_line(x, intercept, slope):
    """
    Straight line function.

    Parameters:
    x (float): The input value.
    intercept (float): The intercept.
    slope (float): The slope.

    Returns:
    float: The calculated value of the straight line function.
    """
    return intercept + slope * x

def expfunc(x, a, b):
    """
    Exponential function.

    Parameters:
    x (float): The input value.
    a (float): The amplitude.
    b (float): The exponential index.

    Returns:
    float: The calculated value of the exponential function.
    """    
    return a * np.exp(b * x)

def powerlaw(x, norm, pindex):
    """
    Power-law function.

    Parameters:
    x (float): The input value.
    norm (float): The amplitude.
    pindex (float): The power-law exponent.

    Returns:
    float: The calculated value of the power-law function.
    """
    return norm * (x) ** pindex