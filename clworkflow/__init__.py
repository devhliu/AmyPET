"""
Centiloid Workflow Package

This package provides a streamlined workflow for calculating centiloid scores
from paired PET and MRI images in NIfTI format.
"""
__author__ = "Extracted from AmyPET"
__copyright__ = "Copyright 2023"

from .pipeline import run_centiloid_pipeline, calculate_group_statistics, compare_tracers
from .visualization import plot_centiloid_comparison, plot_group_comparison, identity_line
from .utils import get_cl_anchors, get_ur2pib, get_clref, check_urs, check_cls, save_cl_anchors

__all__ = [
    'run_centiloid_pipeline',
    'calculate_group_statistics',
    'compare_tracers',
    'plot_centiloid_comparison',
    'plot_group_comparison',
    'identity_line',
    'get_cl_anchors',
    'get_ur2pib',
    'get_clref',
    'check_urs',
    'check_cls',
    'save_cl_anchors'
]