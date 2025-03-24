'''
Adapter module that provides ANTsPy implementations of nimpa and spm12 functionality
'''

__author__ = "devhliu"
__copyright__ = "Copyright 2025-03-24"

import logging

log = logging.getLogger(__name__)

log.info("Using ANTsPy implementation for image processing")
from .ants_compat import nimpa, spm12

# Export the modules for use in other parts of the codebase
__all__ = ['nimpa', 'spm12']