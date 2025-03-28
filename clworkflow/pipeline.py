"""
Centiloid calculation pipeline.

This module provides the main pipeline for calculating centiloid scores
from paired PET and MRI images.
"""
import os
import pickle
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from .core import sort_input, process_single_subject

log = logging.getLogger(__name__)

def run_centiloid_pipeline(pet_files, mri_files, atlas_dir, 
                          outpath=None, tracer='pib', flip_pet=None,
                          parallel=True, max_workers=None):
    """
    Run the centiloid pipeline on paired PET and MRI images.
    
    Args:
        pet_files: List of PET image file paths or directory containing PET images
        mri_files: List of MRI image file paths or directory containing MRI images
        atlas_dir: Directory containing the atlas files
        outpath: Output directory path
        tracer: Tracer type ('pib', 'fbb', 'fbp', 'flute', or 'new')
        flip_pet: Optional list of flip parameters for PET images
        parallel: Whether to process subjects in parallel
        max_workers: Maximum number of parallel workers (None = auto)
        
    Returns:
        Dictionary containing the centiloid calculation results
    """
    # Ensure paths are Path objects
    atlas_dir = Path(atlas_dir)
    if outpath is not None:
        outpath = Path(outpath)
        os.makedirs(outpath, exist_ok=True)
    
    # Sort and validate input files
    pet_mr_list, flips = sort_input(pet_files, mri_files, flip_pet=flip_pet)
    fpets, fmris = pet_mr_list
    
    log.info(f"Processing {len(fpets)} subjects with tracer: {tracer}")
    
    # Process each subject
    results = {}
    
    if parallel and len(fpets) > 1:
        # Process subjects in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, (fpet, fmri) in enumerate(zip(fpets, fmris)):
                subject_outpath = outpath / f"subject_{i}" if outpath else None
                flip = flips[i]
                
                futures.append(
                    executor.submit(
                        process_single_subject,
                        fpet, fmri, atlas_dir, tracer, flip, subject_outpath
                    )
                )
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results[f"subject_{i}"] = result
                except Exception as e:
                    log.error(f"Error processing subject {i}: {e}")
    else:
        # Process subjects sequentially
        for i, (fpet, fmri) in enumerate(zip(fpets, fmris)):
            try:
                subject_outpath = outpath / f"subject_{i}" if outpath else None
                flip = flips[i]
                
                result = process_single_subject(
                    fpet, fmri, atlas_dir, tracer, flip, subject_outpath
                )
                results[f"subject_{i}"] = result
            except Exception as e:
                log.error(f"Error processing subject {i}: {e}")
    
    # Save results if outpath is provided
    if outpath is not None:
        output_file = outpath / f"output_{tracer}.pkl"
        with open(str(output_file), 'wb') as f:
            pickle.dump(results, f)
    
    return results

def calculate_group_statistics(results, group_key=None):
    """
    Calculate group statistics from centiloid results.
    
    Args:
        results: Dictionary of centiloid results
        group_key: Optional key to filter subjects by group
        
    Returns:
        Dictionary with group statistics
    """
    import numpy as np
    
    # Filter subjects by group if needed
    subjects = results.values()
    if group_key:
        subjects = [s for s in subjects if group_key in s['subject'].lower()]
    
    if not subjects:
        return {"error": "No subjects found"}
    
    # Extract centiloid values for each reference region
    regions = list(subjects[0]['cl'].keys())
    cl_values = {region: [] for region in regions}
    
    for subject in subjects:
        for region in regions:
            cl_values[region].append(subject['cl'][region])
    
    # Calculate statistics
    stats = {}
    for region in regions:
        values = np.array(cl_values[region])
        stats[region] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'n': len(values)
        }
    
    return stats

def compare_tracers(pib_results, new_tracer_results, reference_region='wc'):
    """
    Compare a new tracer with PiB reference.
    
    Args:
        pib_results: Results from PiB tracer
        new_tracer_results: Results from the new tracer
        reference_region: Reference region to use for comparison
        
    Returns:
        Dictionary with comparison results
    """
    import numpy as np
    from scipy.stats import linregress
    
    # Extract subject IDs
    pib_subjects = list(pib_results.keys())
    new_subjects = list(new_tracer_results.keys())
    
    # Find common subjects
    common_subjects = set(pib_subjects).intersection(set(new_subjects))
    
    if not common_subjects:
        return {"error": "No common subjects found"}
    
    # Extract centiloid values for common subjects
    pib_values = []
    new_values = []
    
    for subject in common_subjects:
        pib_values.append(pib_results[subject]['cl'][reference_region])
        new_values.append(new_tracer_results[subject]['cl'][reference_region])
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = linregress(pib_values, new_values)
    
    # Calculate conversion factors
    conversion = {
        'm': slope,
        'a': intercept,
        'r': r_value,
        'r2': r_value**2,
        'p': p_value,
        'stderr': std_err
    }
    
    return {
        'conversion': conversion,
        'pib_values': pib_values,
        'new_values': new_values,
        'reference_region': reference_region
    }