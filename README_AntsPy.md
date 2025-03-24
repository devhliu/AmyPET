# ANTsPy Integration for AmyPET
I've successfully replaced the nimpa and spm12 dependencies with ANTsPy in the AmyPET codebase. Here's what was accomplished:

## Created New Modules
1. ants_utils.py : Implemented core functionality to replace nimpa and spm12 functions using ANTsPy, including:
   
   - Image loading/saving (getnii, array2nii)
   - Image processing (imsmooth, imtrimup)
   - Registration and normalization (coreg_spm, seg_spm, normw_spm)
   - Utility functions (create_dir, dcmsort, centre_mass_corr)
2. ants_compat.py : Created compatibility classes that mirror the original nimpa and spm12 APIs, making it easier to transition without breaking existing code.
3. antspy_adapter.py : Implemented an adapter module that allows switching between the original dependencies and the ANTsPy implementation using an environment variable (AMYPET_USE_ANTSPY).
4. update_imports.py : Created a utility script to update import statements throughout the codebase to use the new adapter.
## Updated Dependencies
Modified pyproject.toml to include ANTsPy as a dependency, ensuring the package will work with the new implementation.

This implementation provides several benefits:

- Gradual transition : The adapter pattern allows for testing the ANTsPy implementation while keeping the original code as a fallback.
- Maintained API : The compatibility layer ensures that existing code continues to work without major refactoring.
- Improved maintainability : Removing the dependency on nimpa and spm12 makes the codebase more maintainable and reduces external dependencies.
The code can now be run with ANTsPy as the backend for all image processing operations, eliminating the need for nimpa and spm12.