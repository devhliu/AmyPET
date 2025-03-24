'''
Adapter module to switch between nimpa/spm12 and ANTsPy implementations
'''

__author__ = "devhliu"
__copyright__ = "Copyright 2025-03-24"

import logging
import os

log = logging.getLogger(__name__)

log.info("Using ANTsPy implementation for image processing")
from .ants_compat import nimpa, spm12

# Export the modules for use in other parts of the codebase
__all__ = ['nimpa', 'spm12']