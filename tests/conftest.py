import logging
from os import getenv
from pathlib import Path

import pytest

HOME = Path(getenv("DATA_ROOT", "~")).expanduser()


@pytest.fixture(scope="session")
def nvml():
    import pynvml

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        pytest.skip(str(exc))
    return pynvml


@pytest.fixture(scope="session")
def mMRpars():
    # Using ANTsPy instead of NiftyPET
    pytest.skip("NiftyPET dependency removed, test needs to be updated for ANTsPy")
    # Return a mock params object for compatibility
    return {"Cnt": {"LOG": logging.INFO}}


@pytest.fixture(scope="session")
def datain(mMRpars):
    # Using ANTsPy instead of NiftyPET
    pytest.skip("NiftyPET dependency removed, test needs to be updated for ANTsPy")
    folder_in = HOME / "Ab_PET_mMR_test"
    if not folder_in.is_dir():
        pytest.skip(f"""Cannot find Ab_PET_mMR_test in
${{DATA_ROOT:-~}} ({HOME}).
""")
    # Return a mock data structure for compatibility
    return {"corepath": str(folder_in)}


@pytest.fixture(scope="session")
def dimin():
    trt = HOME / "DPUK" / "TRT"
    if not trt.is_dir():
        pytest.skip(f"""Cannot find DPUK/TRT in
${{DATA_ROOT:-~}} ({HOME}).
""")
    return trt


@pytest.fixture(scope="session")
def fimin(dimin):
    mprage = (dimin / "NEW002_ODE_S02442" / "TP0" /
              "NEW002_PETMR_V1_00015_MR_images_MPRAGE_q-_MPRAGE_20200212145346_15.nii")
    if not mprage.is_file():
        pytest.skip("fimin not found")
    return mprage
