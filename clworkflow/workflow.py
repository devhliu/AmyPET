"""
Centiloid Workflow Implementation

This module provides the main functions for running the centiloid pipeline
on paired PET and MRI images.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from amypet.centiloid import run as centiloid_run
from amypet.backend_centiloid import sort_input
from amypet.utils import get_clref, check_urs, check_cls, save_cl_anchors

def check_image_pairs(fpets, fmris):
    """
    Visual check of PET and MRI image pairs before running the CL pipeline.
    
    Args:
        fpets: List of PET image file paths
        fmris: List of MRI image file paths
    """
    from amypet import im_check_pairs
    return im_check_pairs(fpets, fmris)

def run_centiloid_pipeline(pet_files, mri_files, atlas_dir, 
                          outpath=None, tracer='pib', flip_pet=None):
    """
    Run the centiloid pipeline on paired PET and MRI images.
    
    Args:
        pet_files: List of PET image file paths or directory containing PET images
        mri_files: List of MRI image file paths or directory containing MRI images
        atlas_dir: Directory containing the atlas files
        outpath: Output directory path
        tracer: Tracer type ('pib', 'fbb', 'fbp', 'flute', or 'new')
        flip_pet: Optional list of flip parameters for PET images
        
    Returns:
        Dictionary containing the centiloid calculation results
    """
    # Ensure paths are Path objects
    atlas_dir = Path(atlas_dir)
    if outpath is not None:
        outpath = Path(outpath)
    
    # Sort and validate input files
    pet_mr_list, flips = sort_input(pet_files, mri_files, flip_pet=flip_pet)
    fpets, fmris = pet_mr_list
    
    # Run centiloid calculation
    results = centiloid_run(fpets, fmris, atlas_dir, tracer=tracer, 
                           flip_pet=flips, outpath=outpath)
    
    # Save results if outpath is provided
    if outpath is not None:
        output_file = outpath / f"output_{tracer}.pkl"
        with open(str(output_file), 'wb') as f:
            pickle.dump(results, f)
    
    return results

def calibrate_tracer(pib_results, new_tracer_results):
    """
    Calibrate a new tracer against PiB reference.
    
    Args:
        pib_results: Results from PiB tracer
        new_tracer_results: Results from the new tracer
        
    Returns:
        Calibration data
    """
    from amypet import calib_tracer
    return calib_tracer(pib_results, new_tracer_results)

def save_calibration(calibration_data, tracer_name, outpath=None):
    """
    Save tracer calibration data and create UR to PiB conversion.
    
    Args:
        calibration_data: Calibration data from calibrate_tracer
        tracer_name: Name of the tracer
        outpath: Output directory path
        
    Returns:
        UR to PiB conversion data
    """
    from amypet import save_ur2pib
    
    # Save calibration data
    if outpath is not None:
        outpath = Path(outpath)
        with open(outpath / f"cal_{tracer_name}.pkl", 'wb') as f:
            pickle.dump(calibration_data, f)
    
    # Save UR to PiB conversion
    ur_conversion = save_ur2pib(calibration_data, tracer_name)
    
    return ur_conversion

def compare_with_reference(calibration_data, reference_file, region='wc'):
    """
    Compare calculated centiloid values with reference values.
    
    Args:
        calibration_data: Calibration data from calibrate_tracer
        reference_file: Excel file with reference centiloid values
        region: Region of interest ('wc', 'cg', 'wcb', 'pns')
        
    Returns:
        Figure with comparison plot and R² value
    """
    import openpyxl as xl
    
    # Load reference data
    info = xl.load_workbook(reference_file)
    
    # Extract sheet and data based on file name
    if 'FBB' in str(reference_file):
        dat = info['18F-FBB']
        cl_o = np.array([i.value for i in dat['H'][4:29]] + [i.value for i in dat['H'][30:]])
        pid = [i.value for i in dat['A'][4:29]] + [int(i.value[1:]) for i in dat['A'][30:]]
    elif 'flute' in str(reference_file).lower():
        dat = info['GE PiB & Flutemetamol']
        cl_o = np.array([i.value for i in dat['T'][3:53]] + [i.value for i in dat['T'][54:78]])
        pid = [int(i.value[3:]) for i in dat['C'][3:53]] + [int(i.value[4:]) for i in dat['C'][54:78]]
    elif 'Avid' in str(reference_file):
        dat = info['Sheet1']
        cl_o = np.array([i.value for i in dat['I'][2:]])
        pid = [i.value for i in dat['A'][2:]]
    else:
        raise ValueError("Unrecognized reference file format")
    
    # Match subjects
    idxs = [pid.index(int(i)) for i in calibration_data[region]['sbj']]
    cl_os = cl_o[idxs]
    
    # Get calculated CLs
    cl_amy = calibration_data[region]['calib']['cl_std_fbb']
    
    # Create comparison plot
    fig, ax = plt.subplots()
    ax.scatter(cl_os, cl_amy, c='black')
    from .utils import identity_line
    identity_line(ax=ax, ls='--', c='b')
    ax.set_xlabel('Original F18 CLs')
    ax.set_ylabel('AmyPET F18 CLs')
    ax.grid('on')
    
    # Calculate R²
    m, a, r, p, stderr = linregress(cl_os, cl_amy)
    r2 = r**2
    ax.text(0, 125, f'$R^2={r2:.4f}$', fontsize=12)
    
    return fig, r2