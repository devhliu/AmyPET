from pathlib import Path

import pytest


@pytest.fixture
def dyndir(datain, mMRpars):
    # Using ANTsPy instead of NiftyPET
    pytest.skip("NiftyPET dependency removed, test needs to be updated for ANTsPy")
    # output path
    opth = str(Path(datain["corepath"]).parent / "amypad" / "dyndir")

    res = Path(opth) / "PET" / "multiple-frames"
    if res.is_dir() and len(list(res.glob('*.nii*'))) > 1:
        return res
    
    # This test needs to be rewritten to use ANTsPy instead of NiftyPET
    # The following code is kept as reference but will be skipped

    # object mu-map with alignment
    mupdct = nipet.align_mumap(
        datain,
        mMRpars,
        outpath=opth,
        store=True,
        use_stored=True,
        hst=hst,
        itr=2,
        petopt="ac",
        fcomment="_mu",
        musrc="pct",
    )
    # object mu-map without alignment--straight from DICOM resampled to PET
    # muodct = nipet.obj_mumap(datain, mMRpars, outpath=opth, store=True)

    nipet.mmrchain(
        datain,
        mMRpars,
        frames=frm_timings["timings"],
        mu_h=muhdct,
        mu_o=mupdct,
        itr=5,
        fwhm=0.0,
        outpath=opth,
        fcomment="_dyn",
        store_img=True,
        store_img_intrmd=True,
    )
    return Path(opth) / "PET" / "multiple-frames"


@pytest.mark.timeout(30 * 60) # 30m
def test_imtrimup(dyndir):
    imtrimup = pytest.importorskip("amypet.imtrimup")
    imtrimup.run(dyndir, glob='*_frm?_t*.nii*')
