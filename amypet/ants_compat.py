'''
Compatibility layer for transitioning from nimpa and spm12 to ANTsPy
'''

__author__ = "devhliu"
__copyright__ = "Copyright 2025-03-24"

import logging
from pathlib import Path

# Import the ANTsPy utility functions
from .ants_utils import *

log = logging.getLogger(__name__)

# Create aliases for nimpa functions to maintain compatibility
class nimpa:
    """Compatibility class to replace nimpa with ANTsPy equivalents"""
    # Core functions
    getnii = getnii
    array2nii = array2nii
    create_dir = create_dir
    dcmsort = dcmsort
    centre_mass_corr = centre_mass_corr
    bias_field_correction = bias_field_correction
    imsmooth = imsmooth
    imtrimup = imtrimup
    
    @staticmethod
    def nii_ugzip(fim, outpath=None):
        """Unzip a gzipped NIfTI file if needed
        
        Arguments:
        - fim: file path to the NIfTI image
        - outpath: output directory
        
        Returns:
        - Path to the unzipped NIfTI file
        """
        import gzip
        import shutil
        
        fim = Path(fim)
        if outpath is not None:
            outpath = Path(outpath)
            create_dir(outpath)
        else:
            outpath = fim.parent
        
        # If already unzipped, just return the path
        if fim.suffix != '.gz':
            return str(fim)
        
        # Create output file path
        out_file = outpath / fim.stem
        
        # Unzip the file
        with gzip.open(fim, 'rb') as f_in:
            with open(out_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return str(out_file)
    
    @staticmethod
    def rem_chars(s):
        """Remove special characters from a string"""
        import re
        return re.sub(r'[^\w\s]', '', s)

# Create aliases for spm12 functions to maintain compatibility
class spm12:
    """Compatibility class to replace spm12 with ANTsPy equivalents"""
    # Core functions
    get_bbox = get_bbox
    ensure_spm = ensure_spm
    spm_dir = spm_dir
    standalone_path = standalone_path
    coreg_spm = coreg_spm
    seg_spm = seg_spm
    normw_spm = normw_spm
    resample_spm = resample_spm