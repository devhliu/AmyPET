"""
Example script for running the centiloid workflow.
"""
import pickle
from pathlib import Path

from clworkflow import (
    run_centiloid_pipeline, 
    calculate_group_statistics,
    compare_tracers,
    plot_centiloid_comparison,
    get_clref,
    check_urs,
    check_cls,
    save_cl_anchors
)

def run_pib_calibration():
    """
    Run the PiB calibration workflow.
    """
    # Define paths
    data_dir = Path('/data/AMYPET')
    atlas_dir = data_dir / 'Atlas' / 'CL_2mm'
    output_dir = data_dir / 'CL' / 'PiB'
    
    # Process AD group
    pet_dir = data_dir / 'CL' / 'PiB' / 'AD-100_PET_5070' / 'nifti'
    mri_dir = data_dir / 'CL' / 'PiB' / 'AD-100_MR' / 'nifti'
    pet_files = sorted(pet_dir.glob("*.*"))
    mri_files = sorted(mri_dir.glob("*.*"))
    
    # Define flip parameters for specific subjects if needed
    flip_pet = len(pet_files) * [(1, 1, 1)]
    flip_pet[9] = (1, -1, 1)
    flip_pet[22] = (1, -1, 1)
    flip_pet[26] = (1, -1, 1)
    flip_pet[34] = (1, -1, 1)
    flip_pet[36] = (1, -1, 1)
    flip_pet[40] = (1, -1, 1)
    flip_pet[44] = (1, -1, 1)
    
    # Run centiloid pipeline for AD group
    out_ad = run_centiloid_pipeline(
        pet_files, 
        mri_files, 
        atlas_dir, 
        flip_pet=flip_pet, 
        outpath=output_dir / 'output_pib_ad'
    )
    
    # Process YC group
    pet_dir = data_dir / 'CL' / 'PiB' / 'YC-0_PET_5070' / 'nifti'
    mri_dir = data_dir / 'CL' / 'PiB' / 'YC-0_MR' / 'nifti'
    pet_files = sorted(pet_dir.glob("*.*"))
    mri_files = sorted(mri_dir.glob("*.*"))
    
    # Run centiloid pipeline for YC group
    out_yc = run_centiloid_pipeline(
        pet_files, 
        mri_files, 
        atlas_dir, 
        outpath=output_dir / 'output_pib_yc'
    )
    
    # Load results if needed
    with open(str(output_dir / 'output_pib_yc.pkl'), 'rb') as f:
        out_yc = pickle.load(f)
    
    with open(str(output_dir / 'output_pib_ad.pkl'), 'rb') as f:
        out_ad = pickle.load(f)
    
    # Get reference values
    refs = get_clref(output_dir / 'SupplementaryTable1.xlsx')
    
    # Check uptake ratios and centiloid values
    diff = check_urs(out_yc, out_ad, refs)
    diff = check_cls(out_yc, out_ad, diff, refs)
    
    # Save centiloid anchors
    cla = save_cl_anchors(diff, outpath=output_dir)
    
    # Calculate group statistics
    yc_stats = calculate_group_statistics(out_yc, group_key='YC')
    ad_stats = calculate_group_statistics(out_ad, group_key='AD')
    
    print("PiB calibration completed successfully!")
    print("\nYoung Controls Statistics:")
    print(yc_stats)
    print("\nAD Patients Statistics:")
    print(ad_stats)
    
    return out_yc, out_ad, diff, cla

def run_new_tracer_calibration(tracer='fbb'):
    """
    Run calibration for a new tracer.
    
    Args:
        tracer: Tracer name ('fbb', 'fbp', 'flute')
    """
    # Define paths
    data_dir = Path('/data/AMYPET')
    atlas_dir = data_dir / 'Atlas' / 'CL_2mm'
    output_dir = data_dir / 'CL' / tracer.upper()
    
    # Process AD group
    pet_dir = data_dir / 'CL' / tracer.upper() / f'AD-100_PET_5070'
    mri_dir = data_dir / 'CL' / tracer.upper() / f'AD-100_MR'
    pet_files = sorted(pet_dir.glob("*.*"))
    mri_files = sorted(mri_dir.glob("*.*"))
    
    # Run centiloid pipeline for AD group
    out_ad = run_centiloid_pipeline(
        pet_files, 
        mri_files, 
        atlas_dir, 
        tracer=tracer,
        outpath=output_dir / f'output_{tracer}_ad'
    )
    
    # Process YC group
    pet_dir = data_dir / 'CL' / tracer.upper() / f'YC-0_PET_5070'
    mri_dir = data_dir / 'CL' / tracer.upper() / f'YC-0_MR'
    pet_files = sorted(pet_dir.glob("*.*"))
    mri_files = sorted(mri_dir.glob("*.*"))
    
    # Run centiloid pipeline for YC group
    out_yc = run_centiloid_pipeline(
        pet_files, 
        mri_files, 
        atlas_dir, 
        tracer=tracer,
        outpath=output_dir / f'output_{tracer}_yc'
    )
    
    # Load PiB results for comparison
    pib_dir = data_dir / 'CL' / 'PiB'
    with open(str(pib_dir / 'output_pib_yc.pkl'), 'rb') as f:
        pib_yc = pickle.load(f)
    
    with open(str(pib_dir / 'output_pib_ad.pkl'), 'rb') as f:
        pib_ad = pickle.load(f)
    
    # Compare with PiB
    comparison = compare_tracers(pib_ad, out_ad, reference_region='wc')
    
    # Plot comparison
    plot_centiloid_comparison(
        comparison['pib_values'],
        comparison['new_values'],
        tracer.upper(),
        reference_region='wc',
        outpath=output_dir
    )
    
    print(f"{tracer.upper()} calibration completed successfully!")
    print("\nComparison with PiB:")
    print(f"Slope: {comparison['conversion']['m']:.4f}")
    print(f"Intercept: {comparison['conversion']['a']:.4f}")
    print(f"R²: {comparison['conversion']['r2']:.4f}")
    
    return out_yc, out_ad, comparison

if __name__ == "__main__":
    # Run PiB calibration
    pib_yc, pib_ad, diff, cla = run_pib_calibration()
    
    # Run new tracer calibration (e.g., FBB)
    fbb_yc, fbb_ad, fbb_comparison = run_new_tracer_calibration(tracer='fbb')