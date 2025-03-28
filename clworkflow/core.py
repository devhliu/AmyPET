"""
Core functionality for centiloid score calculation.

This module provides the essential functions for calculating centiloid scores
from PET and MRI images.
"""
import os
import logging
import numpy as np
from pathlib import Path

from amypet.antspy_adapter import nimpa, spm12
from miutil.fdio import hasext, nsort

log = logging.getLogger(__name__)

def load_masks(mskpath, voxsz=2):
    """
    Load the Centiloid PET masks for calculating uptake ratio (UR/SUVr).
    
    Args:
        mskpath: Path to the directory containing mask files
        voxsz: Voxel size (1 or 2 mm)
        
    Returns:
        Tuple of (mask file paths, mask data)
    """
    voxsz = int(voxsz)
    
    if voxsz not in [1, 2]:
        raise ValueError('Incorrect voxel size - only 1 and 2 are accepted.')
    
    log.info('Loading CL masks...')
    fmasks = {
        'cg': mskpath / f'voi_CerebGry_{voxsz}mm.nii', 
        'wc': mskpath / f'voi_WhlCbl_{voxsz}mm.nii',
        'wcb': mskpath / f'voi_WhlCblBrnStm_{voxsz}mm.nii',
        'pns': mskpath / f'voi_Pons_{voxsz}mm.nii', 
        'ctx': mskpath / f'voi_ctx_{voxsz}mm.nii'
    }
    masks = {fmsk: nimpa.getnii(fmasks[fmsk]) for fmsk in fmasks}
    
    return fmasks, masks

def sort_input(fpets, fmris, flip_pet=None):
    """
    Classify input data of PET and MRI and optionally flip PET if needed.
    
    Args:
        fpets: List or string or Path to PET image(s)
        fmris: List or string or Path to MRI image(s)
        flip_pet: List of flips (3D tuples) for PET images
        
    Returns:
        Tuple of (pet_mr_list, flips)
    """
    if isinstance(fpets, (str, Path)) and isinstance(fmris, (str, Path)):
        # When single PET and MR files are given
        if os.path.isfile(fpets) and os.path.isfile(fmris):
            pet_mr_list = [[fpets], [fmris]]
        # When folder paths are given for PET and MRI files
        elif os.path.isdir(fpets) and os.path.isdir(fmris):
            fp = nsort(
                os.path.join(fpets, f) for f in os.listdir(fpets) 
                if hasext(f, ('nii', 'nii.gz')))
            fm = nsort(
                os.path.join(fmris, f) for f in os.listdir(fmris) 
                if hasext(f, ('nii', 'nii.gz')))
            pet_mr_list = [fp, fm]
        else:
            raise ValueError('Unrecognized or unmatched input for PET and MRI image data')
    elif isinstance(fpets, list) and isinstance(fmris, list):
        if len(fpets) != len(fmris):
            raise ValueError('The number of PET and MRI files must match!')
        # Check if all files exist
        if not all(os.path.isfile(f) and hasext(f, ('nii', 'nii.gz')) for f in fpets) or not all(
                os.path.isfile(f) and hasext(f, ('nii', 'nii.gz')) for f in fmris):
            raise ValueError('Some paired files do not exist or have incorrect format')
        pet_mr_list = [fpets, fmris]
    else:
        raise ValueError('Unrecognized input image data')

    # Process flip parameters
    if flip_pet is not None:
        if isinstance(flip_pet, tuple) and len(pet_mr_list[0]) == 1:
            flips = [flip_pet]
        elif isinstance(flip_pet, list) and len(flip_pet) == len(pet_mr_list[0]):
            flips = flip_pet
        else:
            log.warning('The flip definition is not compatible with the list of PET images')
            flips = [None] * len(pet_mr_list[0])
    else:
        flips = [None] * len(pet_mr_list[0])

    return pet_mr_list, flips

def calculate_uptake_ratios(wpet, masks):
    """
    Calculate uptake ratios (UR/SUVr) for different reference regions.
    
    Args:
        wpet: Warped PET image data
        masks: Dictionary of mask data
        
    Returns:
        Dictionary of uptake ratios
    """
    # Extract masks
    ctx = masks['ctx']
    cg = masks['cg']
    wc = masks['wc']
    wcb = masks['wcb']
    pns = masks['pns']
    
    # Calculate mean values in target and reference regions
    ctx_mean = np.mean(wpet[ctx > 0])
    cg_mean = np.mean(wpet[cg > 0])
    wc_mean = np.mean(wpet[wc > 0])
    wcb_mean = np.mean(wpet[wcb > 0])
    pns_mean = np.mean(wpet[pns > 0])
    
    # Calculate uptake ratios
    ur = {
        'cg': ctx_mean / cg_mean,
        'wc': ctx_mean / wc_mean,
        'wcb': ctx_mean / wcb_mean,
        'pns': ctx_mean / pns_mean
    }
    
    return ur

def convert_to_centiloid(ur, tracer='pib'):
    """
    Convert uptake ratios to centiloid values.
    
    Args:
        ur: Dictionary of uptake ratios
        tracer: Tracer type ('pib', 'fbb', 'fbp', 'flute')
        
    Returns:
        Dictionary of centiloid values
    """
    # Standard conversion factors for PiB
    if tracer.lower() == 'pib':
        cl = {
            'cg': (ur['cg'] - 1.009) / (2.088 - 1.009) * 100,
            'wc': (ur['wc'] - 1.007) / (1.932 - 1.007) * 100,
            'wcb': (ur['wcb'] - 1.005) / (1.869 - 1.005) * 100,
            'pns': (ur['pns'] - 0.991) / (1.839 - 0.991) * 100
        }
    # For other tracers, use tracer-specific conversion if available
    else:
        # Import conversion factors from amypet
        from amypet.utils import get_ur2pib
        
        try:
            conversion = get_ur2pib(tracer)
            
            # Apply conversion to PiB scale first, then to centiloid
            pib_ur = {}
            for region in ur:
                if region in conversion:
                    slope = conversion[region]['m']
                    intercept = conversion[region]['a']
                    pib_ur[region] = slope * ur[region] + intercept
                else:
                    pib_ur[region] = ur[region]
                    log.warning(f"No conversion factor found for {region} with tracer {tracer}")
            
            # Convert PiB-equivalent URs to centiloid
            cl = {
                'cg': (pib_ur['cg'] - 1.009) / (2.088 - 1.009) * 100,
                'wc': (pib_ur['wc'] - 1.007) / (1.932 - 1.007) * 100,
                'wcb': (pib_ur['wcb'] - 1.005) / (1.869 - 1.005) * 100,
                'pns': (pib_ur['pns'] - 0.991) / (1.839 - 0.991) * 100
            }
        except Exception as e:
            log.error(f"Error converting {tracer} to centiloid: {e}")
            # Fallback to PiB conversion
            cl = {
                'cg': (ur['cg'] - 1.009) / (2.088 - 1.009) * 100,
                'wc': (ur['wc'] - 1.007) / (1.932 - 1.007) * 100,
                'wcb': (ur['wcb'] - 1.005) / (1.869 - 1.005) * 100,
                'pns': (ur['pns'] - 0.991) / (1.839 - 0.991) * 100
            }
    
    return cl

def process_single_subject(fpet, fmri, atlas_dir, tracer='pib', flip_pet=None, outpath=None):
    """
    Process a single subject's PET and MRI images to calculate centiloid scores.
    
    Args:
        fpet: Path to PET image
        fmri: Path to MRI image
        atlas_dir: Path to atlas directory
        tracer: Tracer type
        flip_pet: Flip parameters for PET image
        outpath: Output directory
        
    Returns:
        Dictionary with processing results
    """
    # Create output directory if needed
    if outpath:
        outpath = Path(outpath)
        os.makedirs(outpath, exist_ok=True)
    
    # Load masks
    fmasks, masks = load_masks(atlas_dir)
    
    # Process PET image (flip if needed)
    if flip_pet:
        pet_data = nimpa.getnii(fpet, output='all')
        flipped_data = np.flip(pet_data['im'], axis=tuple(i for i, f in enumerate(flip_pet) if f < 0))
        flipped_file = str(outpath / f"flipped_{Path(fpet).name}") if outpath else None
        nimpa.array2nii(flipped_data, pet_data['affine'], flipped_file)
        fpet = flipped_file
    
    # Perform spatial normalization using SPM
    # 1. Reorient MRI to standard space
    spm12.reorient(fmri)
    
    # 2. Reorient PET to standard space
    spm12.reorient(fpet)
    
    # 3. Coregister PET to MRI
    spm12.coreg(fmri, fpet)
    
    # 4. Segment MRI
    seg_files = spm12.segment(fmri)
    
    # 5. Normalize MRI and PET to MNI space
    wpet = spm12.normalize(seg_files['forward_deformation'], [fpet], voxsz=2)[0]
    
    # Calculate uptake ratios
    wpet_data = nimpa.getnii(wpet)
    ur = calculate_uptake_ratios(wpet_data, masks)
    
    # Convert to centiloid
    cl = convert_to_centiloid(ur, tracer)
    
    # Create quality control image if output path is provided
    qc_image = None
    if outpath:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # Create figure for quality control
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(2, 3, figure=fig)
            
            # Plot MRI
            mri_data = nimpa.getnii(fmri)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(np.rot90(mri_data[:, mri_data.shape[1]//2, :]), cmap='gray')
            ax1.set_title('MRI (Coronal)')
            ax1.axis('off')
            
            # Plot PET
            pet_data = nimpa.getnii(fpet)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(np.rot90(pet_data[:, pet_data.shape[1]//2, :]), cmap='hot')
            ax2.set_title('PET (Coronal)')
            ax2.axis('off')
            
            # Plot warped PET
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(np.rot90(wpet_data[:, wpet_data.shape[1]//2, :]), cmap='hot')
            ax3.set_title('Warped PET (Coronal)')
            ax3.axis('off')
            
            # Plot centiloid values
            ax4 = fig.add_subplot(gs[1, :])
            regions = list(cl.keys())
            values = [cl[r] for r in regions]
            ax4.bar(regions, values)
            ax4.set_title('Centiloid Values by Reference Region')
            ax4.set_ylim(0, max(values) * 1.2)
            for i, v in enumerate(values):
                ax4.text(i, v + 2, f"{v:.1f}", ha='center')
            
            # Save figure
            qc_image = str(outpath / f"qc_{Path(fpet).stem}.png")
            plt.tight_layout()
            plt.savefig(qc_image)
            plt.close()
        except Exception as e:
            log.error(f"Error creating QC image: {e}")
    
    # Prepare results
    results = {
        'subject': Path(fpet).stem,
        'pet': str(fpet),
        'mri': str(fmri),
        'wpet': wpet,
        'ur': ur,
        'cl': cl,
        'qc_image': qc_image
    }
    
    return results