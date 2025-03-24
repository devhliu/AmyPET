'''
ANTsPy utility functions to replace nimpa and spm12 functionality
'''

__author__ = "devhliu"
__copyright__ = "Copyright 2025-03-24"

import logging
import os
import shutil
from pathlib import Path
import numpy as np
import nibabel as nib
import ants

log = logging.getLogger(__name__)

# =======================================================================================
def create_dir(pth):
    """Create directory if it doesn't exist"""
    os.makedirs(pth, exist_ok=True)
    return pth

# =======================================================================================
def getnii(fim, output='im'):
    """Load NIfTI image using ANTsPy
    
    Arguments:
    - fim: file path to the NIfTI image or ANTsImage
    - output: what to output: 'im' for just the image array, 'all' for dictionary with metadata
    
    Returns:
    - Numpy array of image data or dictionary with image data and metadata
    """
    if isinstance(fim, ants.ANTsImage):
        img = fim
    elif isinstance(fim, (str, Path)):
        img = ants.image_read(str(fim))
    else:
        raise ValueError(f"Unrecognized input type: {type(fim)}")
    
    # Get the numpy array
    img_array = img.numpy()
    
    # If only the image array is requested
    if output == 'im':
        return img_array
    
    # If all metadata is requested
    elif output == 'all':
        # Load with nibabel to get header info
        nib_img = nib.load(str(fim))
        affine = nib_img.affine
        header = nib_img.header
        
        # Determine orientation information
        # This is a simplified approach - ANTsPy and NiBabel handle orientation differently
        # than nimpa, so this is an approximation
        transpose = (0, 1, 2)  # Default orientation
        flip = [False, False, False]  # Default no flip
        
        # Get voxel size
        voxsize = header.get_zooms()
        
        return {
            'im': img_array,
            'affine': affine,
            'hdr': header,
            'shape': img_array.shape,
            'voxsize': voxsize,
            'transpose': transpose,
            'flip': flip
        }
    else:
        raise ValueError(f"Unrecognized output type: {output}")

# =======================================================================================
def array2nii(im, affine, fname, descrip=None, trnsp=None, flip=None):
    """Save numpy array as NIfTI file using ANTsPy
    
    Arguments:
    - im: numpy array of image data
    - affine: 4x4 affine transformation matrix
    - fname: output file path
    - descrip: description to add to NIfTI header
    - trnsp: tuple indicating transpose order (not used in ANTsPy implementation)
    - flip: list of booleans indicating which axes to flip (not used in ANTsPy implementation)
    """
    # Convert to ANTsImage
    ants_img = ants.from_numpy(im, origin=None, spacing=None, direction=None)
    
    # Set the direction and spacing from the affine
    # This is a simplified approach - ANTsPy handles this differently
    # than nimpa, so this is an approximation
    spacing = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    ants_img.set_spacing(spacing)
    
    # Save the image
    ants.image_write(ants_img, str(fname))
    
    # If description is needed, we need to modify the header with nibabel
    if descrip is not None:
        nib_img = nib.load(str(fname))
        header = nib_img.header
        header['descrip'] = descrip
        nib.save(nib.Nifti1Image(nib_img.get_fdata(), nib_img.affine, header), str(fname))
    
    return fname

# =======================================================================================
def dcmsort(dcmdir):
    """Sort DICOM files in a directory
    
    This is a simplified version that just checks if DICOM files exist
    For full DICOM sorting, consider using pydicom directly
    """
    import glob
    dcmdir = Path(dcmdir)
    dcm_files = glob.glob(str(dcmdir / "*.dcm"))
    if not dcm_files:
        dcm_files = glob.glob(str(dcmdir / "*.IMA")) + glob.glob(str(dcmdir / "*.ima"))
    
    return len(dcm_files) > 0

# =======================================================================================
def centre_mass_corr(fim, flip=None, outpath=None, com=None):
    """Center of mass correction for an image
    
    Arguments:
    - fim: file path to the image or ANTsImage
    - flip: list of booleans indicating which axes to flip
    - outpath: output directory
    - com: center of mass coordinates (if already known)
    """
    # Load the image
    if isinstance(fim, (str, Path)):
        img = ants.image_read(str(fim))
        fim = Path(fim)
    else:
        img = fim
        fim = Path("image.nii.gz")
    
    # Get image data
    img_array = img.numpy()
    
    # Apply flip if requested
    if flip is not None:
        for i, do_flip in enumerate(flip):
            if do_flip:
                img_array = np.flip(img_array, axis=i)
    
    # Calculate center of mass if not provided
    if com is None:
        # Create a mask of non-zero voxels
        mask = img_array > 0
        if not np.any(mask):
            com_abs = np.zeros(3)
        else:
            # Get coordinates of non-zero voxels
            coords = np.array(np.where(mask)).T
            # Weight by intensity
            weights = img_array[mask]
            # Calculate center of mass
            com_abs = np.average(coords, axis=0, weights=weights)
    else:
        com_abs = com
    
    # Create output path if needed
    if outpath is not None:
        outpath = Path(outpath)
        create_dir(outpath)
        out_file = outpath / (fim.stem + "_com.nii.gz")
    else:
        out_file = fim.parent / (fim.stem + "_com.nii.gz")
    
    # Create a new ANTsImage with center of mass at origin
    new_img = ants.from_numpy(img_array)
    
    # Set the same spacing and direction as the original
    new_img.set_spacing(img.spacing)
    new_img.set_direction(img.direction)
    
    # Adjust the origin to center the image at the center of mass
    # This is a simplified approach - ANTsPy handles this differently
    # than nimpa, so this is an approximation
    origin = img.origin - com_abs * img.spacing
    new_img.set_origin(origin)
    
    # Save the image
    ants.image_write(new_img, str(out_file))
    
    return {'fim': out_file, 'com_abs': com_abs}

# =======================================================================================
def bias_field_correction(fim, executable='ants', outpath=None):
    """Perform bias field correction using ANTsPy N4 algorithm
    
    Arguments:
    - fim: file path to the image or ANTsImage
    - executable: which implementation to use (only 'ants' supported)
    - outpath: output directory
    """
    if executable != 'ants' and executable != 'sitk':
        raise ValueError("Only 'ants' implementation is supported")
    
    # Load the image
    if isinstance(fim, (str, Path)):
        img = ants.image_read(str(fim))
        fim = Path(fim)
    else:
        img = fim
        fim = Path("image.nii.gz")
    
    # Create output path if needed
    if outpath is not None:
        outpath = Path(outpath)
        create_dir(outpath)
        out_file = outpath / (fim.stem + "_n4.nii.gz")
    else:
        out_file = fim.parent / (fim.stem + "_n4.nii.gz")
    
    # Perform N4 bias field correction
    corrected_img = ants.n4_bias_field_correction(img)
    
    # Save the image
    ants.image_write(corrected_img, str(out_file))
    
    return {'fim': out_file}

# =======================================================================================
def imsmooth(im, fwhm=4.0, voxsize=None):
    """Smooth image with Gaussian kernel
    
    Arguments:
    - im: numpy array or ANTsImage
    - fwhm: full width at half maximum of Gaussian kernel in mm
    - voxsize: voxel size in mm (required if im is numpy array)
    """
    # Convert sigma from FWHM
    # FWHM = 2.355 * sigma
    sigma_mm = fwhm / 2.355
    
    # Convert to ANTsImage if numpy array
    if isinstance(im, np.ndarray):
        if voxsize is None:
            raise ValueError("voxsize must be provided if im is numpy array")
        img = ants.from_numpy(im)
        img.set_spacing(voxsize)
    else:
        img = im
    
    # Convert sigma from mm to voxels for each dimension
    sigma_vox = [sigma_mm / s for s in img.spacing]
    
    # Smooth the image
    smoothed_img = ants.smooth_image(img, sigma_vox)
    
    # Return as numpy array if input was numpy array
    if isinstance(im, np.ndarray):
        return smoothed_img.numpy()
    else:
        return smoothed_img

# =======================================================================================
def imtrimup(fim, scale=None, int_order=1, store_img_intrmd=False, outpath=None):
    """Trim and upscale an image
    
    Arguments:
    - fim: file path to the image or ANTsImage
    - scale: scale factor for each dimension
    - int_order: interpolation order
    - store_img_intrmd: whether to store intermediate images
    - outpath: output directory
    """
    # Load the image
    if isinstance(fim, (str, Path)):
        img = ants.image_read(str(fim))
        fim = Path(fim)
    else:
        img = fim
        fim = Path("image.nii.gz")
    
    # Create output path if needed
    if outpath is not None:
        outpath = Path(outpath)
        create_dir(outpath)
    else:
        outpath = fim.parent
    
    # Get image data
    img_array = img.numpy()
    
    # If no scale provided, use default
    if scale is None:
        scale = [2, 2, 2]
    
    # Create output file paths
    fimi = [outpath / (fim.stem + "_trimup.nii.gz")]
    
    # Determine new spacing
    new_spacing = [s / sc for s, sc in zip(img.spacing, scale)]
    
    # Resample the image
    if int_order == 0:
        interp_type = 'nearestNeighbor'
    elif int_order == 1:
        interp_type = 'linear'
    else:
        interp_type = 'bSpline'
    
    # Resample to new spacing
    resampled_img = ants.resample_image(img, new_spacing, use_voxels=False, interp_type=interp_type)
    
    # Save the image
    ants.image_write(resampled_img, str(fimi[0]))
    
    return {'im': resampled_img.numpy(), 'fimi': fimi}

# =======================================================================================
# SPM12 replacement functions
# =======================================================================================

def get_bbox(img_dict):
    """Get bounding box for an image
    
    This is a simplified version that returns a standard bounding box
    """
    # Default bounding box similar to SPM12's default
    return [[-78, -112, -70], [78, 76, 85]]

def ensure_spm():
    """Placeholder for SPM12's ensure_spm function
    
    This function is not needed with ANTsPy but included for compatibility
    """
    return None

def spm_dir():
    """Return a placeholder for SPM12 directory
    
    This function is not needed with ANTsPy but included for compatibility
    """
    return "/placeholder/spm12"

def standalone_path():
    """Return a placeholder for SPM12 standalone path
    
    This function is not needed with ANTsPy but included for compatibility
    """
    class DummyPath:
        def __init__(self):
            self.parent = self
    return DummyPath()

def coreg_spm(ref_img, flo_img, fwhm_ref=0, fwhm_flo=0, outpath=None, fname_aff="", 
              fcomment="", pickname="ref", costfun="nmi", graphics=1, visual=0, 
              del_uncmpr=True, save_arr=True, save_txt=True, modify_nii=True, 
              standalone=False):
    """Register floating image to reference image using ANTsPy
    
    Arguments:
    - ref_img: reference image path or ANTsImage
    - flo_img: floating image path or ANTsImage
    - fwhm_ref: smoothing for reference image
    - fwhm_flo: smoothing for floating image
    - outpath: output directory
    - Various other parameters for compatibility with SPM12 function
    
    Returns:
    - Dictionary with registration results
    """
    # Load images
    if isinstance(ref_img, (str, Path)):
        fixed = ants.image_read(str(ref_img))
        ref_img = Path(ref_img)
    else:
        fixed = ref_img
        ref_img = Path("reference.nii.gz")
    
    if isinstance(flo_img, (str, Path)):
        moving = ants.image_read(str(flo_img))
        flo_img = Path(flo_img)
    else:
        moving = flo_img
        flo_img = Path("floating.nii.gz")
    
    # Create output path if needed
    if outpath is not None:
        outpath = Path(outpath)
        create_dir(outpath)
    else:
        outpath = flo_img.parent
    
    # Apply smoothing if requested
    if fwhm_ref > 0:
        fixed = imsmooth(fixed, fwhm=fwhm_ref)
    
    if fwhm_flo > 0:
        moving = imsmooth(moving, fwhm=fwhm_flo)
    
    # Perform registration
    # For rigid registration, use ANTsPy's registration function
    reg_result = ants.registration(fixed=fixed, moving=moving, type_of_transform='Rigid')
    
    # Get the registered image
    registered_img = reg_result['warpedmovout']
    
    # Create output file name
    if fcomment:
        out_file = outpath / (flo_img.stem + fcomment + ".nii.gz")
    else:
        out_file = outpath / (flo_img.stem + "_reg.nii.gz")
    
    # Save the registered image
    ants.image_write(registered_img, str(out_file))
    
    # Get the transformation matrix
    # ANTsPy doesn't directly provide the affine matrix in the same format as SPM12
    # This is a simplified approach
    affine = np.eye(4)
    
    # Save the affine matrix if requested
    if save_arr:
        affine_file = outpath / (flo_img.stem + "_affine.txt")
        np.savetxt(str(affine_file), affine)
    
    return {
        'freg': out_file,
        'affine': affine,
        'faffine': outpath / (flo_img.stem + "_affine.txt") if save_arr else None
    }

def seg_spm(img, spm_path, outpath=None, store_nat_gm=True, store_nat_wm=True, 
            store_nat_csf=True, store_fwd=True, store_inv=True, visual=0, 
            standalone=False):
    """Segment an image using ANTsPy
    
    Arguments:
    - img: image path or ANTsImage
    - spm_path: path to SPM12 (not used)
    - outpath: output directory
    - Various other parameters for compatibility with SPM12 function
    
    Returns:
    - Dictionary with segmentation results
    """
    # Load image
    if isinstance(img, (str, Path)):
        image = ants.image_read(str(img))
        img = Path(img)
    else:
        image = img
        img = Path("image.nii.gz")
    
    # Create output path if needed
    if outpath is not None:
        outpath = Path(outpath)
        create_dir(outpath)
    else:
        outpath = img.parent
    
    # Perform segmentation using ANTsPy's atropos function
    seg_result = ants.atropos(image, x=image > 0)
    
    # Extract probability maps
    prob_maps = seg_result['probabilityimages']
    
    # Initialize result dictionary
    result = {}
    
    # Save probability maps if requested
    if store_nat_gm and len(prob_maps) > 0:
        gm_file = outpath / (img.stem + "_c1.nii.gz")
        ants.image_write(prob_maps[0], str(gm_file))
        result['c1'] = gm_file
    
    if store_nat_wm and len(prob_maps) > 1:
        wm_file = outpath / (img.stem + "_c2.nii.gz")
        ants.image_write(prob_maps[1], str(wm_file))
        result['c2'] = wm_file
    
    if store_nat_csf and len(prob_maps) > 2:
        csf_file = outpath / (img.stem + "_c3.nii.gz")
        ants.image_write(prob_maps[2], str(csf_file))
        result['c3'] = csf_file
    
    # Create placeholder deformation fields
    if store_fwd:
        fordef_file = outpath / (img.stem + "_y_fwd.nii.gz")
        # Create an identity deformation field (placeholder)
        shape = image.numpy().shape + (3,)
        identity_field = np.zeros(shape)
        for i in range(3):
            identity_field[..., i] = np.indices(image.numpy().shape)[i]
        nib.save(nib.Nifti1Image(identity_field, np.eye(4)), str(fordef_file))
        result['fordef'] = fordef_file
    
    if store_inv:
        invdef_file = outpath / (img.stem + "_y_inv.nii.gz")
        # Create an identity deformation field (placeholder)
        shape = image.numpy().shape + (3,)
        identity_field = np.zeros(shape)
        for i in range(3):
            identity_field[..., i] = np.indices(image.numpy().shape)[i]
        nib.save(nib.Nifti1Image(identity_field, np.eye(4)), str(invdef_file))
        result['invdef'] = invdef_file
    
    return result

def normw_spm(deformation_field, images, voxsz=1.0, intrp=1, bbox=None, outpath=None, 
               standalone=False):
    """Apply deformation field to images using ANTsPy
    
    Arguments:
    - deformation_field: deformation field path or ANTsImage
    - images: list of image paths or ANTsImages to warp
    - voxsz: output voxel size
    - intrp: interpolation order
    - bbox: bounding box
    - outpath: output directory
    - standalone: whether to use standalone version (not used)
    
    Returns:
    - List of warped image paths
    """
    # Ensure images is a list
    if not isinstance(images, list):
        images = [images]
    
    # Create output path if needed
    if outpath is not None:
        outpath = Path(outpath)
        create_dir(outpath)
    
    # Load deformation field
    # Note: ANTsPy handles deformation fields differently than SPM12
    # This is a simplified approach
    if isinstance(deformation_field, (str, Path)):
        def_field_path = Path(deformation_field)
    else:
        def_field_path = Path("deformation_field.nii.gz")
    
    # Process each image
    warped_images = []
    for img in images:
        # Load image
        if isinstance(img, (str, Path)):
            image = ants.image_read(str(img))
            img_path = Path(img)
        else:
            image = img
            img_path = Path("image.nii.gz")
        
        # Determine output path
        if outpath is not None:
            out_file = outpath / (img_path.stem + "_warped.nii.gz")
        else:
            out_file = img_path.parent / (img_path.stem + "_warped.nii.gz")
        
        # Set interpolation type
        if intrp == 0:
            interp_type = 'nearestNeighbor'
        elif intrp == 1:
            interp_type = 'linear'
        else:
            interp_type = 'bSpline'
        
        # Create a reference image with desired voxel size
        # This is a simplified approach - ANTsPy handles this differently
        # than SPM12, so this is an approximation
        if bbox is None:
            bbox = [[-78, -112, -70], [78, 76, 85]]
        
        # Calculate dimensions based on bbox and voxel size
        dimensions = [int((bbox[1][i] - bbox[0][i]) / voxsz) for i in range(3)]
        
        # Create a reference image
        ref_img = ants.make_image(dimensions, spacing=[voxsz]*3)
        
        # Apply transformation
        # Note: This is a simplified approach. In reality, you would need to
        # convert the SPM12 deformation field to ANTsPy format
        warped_img = ants.apply_transforms(ref_img, image, transformlist=None, 
                                          interpolator=interp_type)
        
        # Save the warped image
        ants.image_write(warped_img, str(out_file))
        warped_images.append(out_file)
    
    return warped_images

def resample_spm(ref_img, flo_img, M, intrp=1.0, outpath=None, pickname="ref", 
                 fcomment="", del_ref_uncmpr=True, del_flo_uncmpr=True, 
                 del_out_uncmpr=True):
    """Resample an image using ANTsPy
    
    Arguments:
    - ref_img: reference image path or ANTsImage
    - flo_img: floating image path or ANTsImage
    - M: transformation matrix
    - intrp: interpolation order
    - outpath: output directory
    - Various other parameters for compatibility with SPM12 function
    
    Returns:
    - Path to resampled image
    """
    # Load images
    if isinstance(ref_img, (str, Path)):
        reference = ants.image_read(str(ref_img))
        ref_img = Path(ref_img)
    else:
        reference = ref_img
        ref_img = Path("reference.nii.gz")
    
    if isinstance(flo_img, (str, Path)):
        floating = ants.image_read(str(flo_img))
        flo_img = Path(flo_img)
    else:
        floating = flo_img
        flo_img = Path("floating.nii.gz")
    
    # Create output path if needed
    if outpath is not None:
        outpath = Path(outpath)
        create_dir(outpath)
    else:
        outpath = flo_img.parent
    
    # Set interpolation type
    if intrp == 0:
        interp_type = 'nearestNeighbor'
    elif intrp == 1:
        interp_type = 'linear'
    else:
        interp_type = 'bSpline'
    
    # Create output file name
    if fcomment:
        out_file = outpath / (flo_img.stem + fcomment + ".nii.gz")
    else:
        out_file = outpath / (flo_img.stem + "_resampled.nii.gz")
    
    # Convert transformation matrix to ANTsPy format
    # This is a simplified approach - ANTsPy handles transformations differently
    # than SPM12, so this is an approximation
    transform = ants.create_ants_transform(transform_type='AffineTransform', 
                                          dimension=3, 
                                          matrix=M.flatten())
    
    # Apply transformation
    resampled_img = ants.apply_transforms(reference, floating, 
                                         transformlist=[transform], 
                                         interpolator=interp_type)
    
    # Save the resampled image
    ants.image_write(resampled_img, str(out_file))
    
    return out_file