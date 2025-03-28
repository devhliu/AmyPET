# Centiloid Workflow

This package provides a streamlined workflow for calculating centiloid scores from paired PET and MRI images in NIfTI format.

## Overview

The centiloid scale is a standardized quantification method for amyloid PET imaging that allows comparison of results across different tracers and analysis methods. This workflow implements the centiloid calculation pipeline according to the standard methodology.

## Main Components

1. **run_centiloid_pipeline**: Main function to process paired PET and MRI images and calculate centiloid scores
2. **calculate_group_statistics**: Function to calculate statistics for groups of subjects
3. **compare_tracers**: Function to compare a new tracer with PiB reference
4. **plot_centiloid_comparison**: Function to visualize tracer comparisons
5. **plot_group_comparison**: Function to visualize group comparisons

## Workflow Steps

The centiloid calculation workflow consists of the following steps:

1. **Image Preprocessing**:
   - Reorientation of MRI and PET images to standard space
   - Coregistration of PET to MRI
   - Segmentation of MRI
   - Spatial normalization to MNI space

2. **Uptake Ratio Calculation**:
   - Extraction of mean values from target and reference regions
   - Calculation of uptake ratios (SUVr) for different reference regions:
     - Grey cerebellum (cg)
     - Whole cerebellum (wc)
     - Whole cerebellum + brainstem (wcb)
     - Pons (pns)

3. **Centiloid Conversion**:
   - Conversion of uptake ratios to centiloid scale
   - For PiB: Direct conversion using standard equations
   - For other tracers: Conversion to PiB-equivalent values first

4. **Calibration and Validation**:
   - Comparison with reference values
   - Calculation of conversion factors for new tracers
   - Statistical analysis and visualization

## Usage Example

```python
from pathlib import Path
from clworkflow import run_centiloid_pipeline

# Define paths
data_dir = Path('/path/to/data')
atlas_dir = data_dir / 'Atlas' / 'CL_2mm'
output_dir = data_dir / 'output'

# Define input files
pet_files = sorted(data_dir / 'pet_images'.glob("*.nii*"))
mri_files = sorted(data_dir / 'mri_images'.glob("*.nii*"))

# Run centiloid pipeline
results = run_centiloid_pipeline(
    pet_files, 
    mri_files, 
    atlas_dir, 
    outpath=output_dir
)

print("Centiloid calculation completed!")