'''
Static frames processing tools for AmyPET
'''

__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2022-3"

import logging
import os
from pathlib import Path
from subprocess import run

import dcm2niix
import numpy as np
from matplotlib import pyplot as plt
from miutil.fdio import hasext
from niftypet import nimpa

from .preproc import r_trimup
from .proc import extract_vois

log = logging.getLogger(__name__)

nifti_ext = 'nii', 'nii.gz'
dicom_ext = 'dcm', 'img', 'ima'


# ========================================================================================
def preproc_ur(pet_path, frames=None, outpath=None, fname=None, com_correction=True, fwhm=0.,
               force=True):
    ''' Prepare the PET image for UR (aka SUVr) analysis.
        Arguments:
        - pet_path: path to the folder of DICOM images, or to the NIfTI file
        - outpath:  output folder path; if not given will assume the parent
                    folder of the input image
        - fname:    core name of the static (UR) NIfTI file
        - frames:   list of frames to be used for UR processing
        - com_correction: centre-of-mass correction - moves the coordinate
                    system to the centre of the spatial image intensity
                    distribution.
        - fwhm:     smoothing parameter in mm (FWHM) for the Gaussian kernel
        - force:    forces the generation of the UR image even if it
                    exists.

    '''

    if not os.path.exists(pet_path):
        raise ValueError('The provided path does not exist')

    # > convert the path to Path object
    pet_path = Path(pet_path)

    # --------------------------------------
    # > sort out the output folder
    if outpath is None:
        petout = pet_path.parent
    else:
        petout = Path(outpath)

    nimpa.create_dir(petout)

    if fname is None:
        fname = nimpa.rem_chars(pet_path.name.split('.')[0]) + '_static.nii.gz'
    elif not hasext(fname, nifti_ext[1]):
        fname += '.nii.gz'
    # --------------------------------------

    # > NIfTI case
    if pet_path.is_file() and hasext(pet_path, nifti_ext):
        log.info('PET path exists and it is a NIfTI file')

        fpet_nii = pet_path

    # > DICOM case (if any file inside the folder is DICOM)
    elif pet_path.is_dir() and any(hasext(f, dicom_ext) for f in pet_path.glob('*')):

        # > get the NIfTi images from previous processing
        fpet_nii = list(petout.glob(pet_path.name + '*.nii*'))

        if not fpet_nii:
            run([dcm2niix.bin, '-i', 'y', '-v', 'n', '-o', petout, 'f', '%f_%s', pet_path])

        fpet_nii = list(petout.glob(pet_path.name + '*.nii*'))

        # > if cannot find a file it might be due to spaces in folder/file names
        if not fpet_nii:
            fpet_nii = list(petout.glob(pet_path.name.replace(' ', '_') + '*.nii*'))

        if not fpet_nii:
            raise ValueError('No static UR NIfTI files found')
        elif len(fpet_nii) > 1:
            raise ValueError('Too many static UR NIfTI files found')
        else:
            fpet_nii = fpet_nii[0]

    # > read the dynamic image
    imdct = nimpa.getnii(fpet_nii, output='all')

    # > number of dynamic frames
    nfrm = imdct['hdr']['dim'][4]

    # > ensure that the frames exist in part of full dynamic image data
    if frames and nfrm < max(frames):
        raise ValueError('The selected frames do not exist')
    elif not frames:
        frames = np.arange(nfrm)

    log.info(f'{nfrm} frames have been found in the dynamic image.')

    # ------------------------------------------
    # > static image file path
    fstat = petout / fname

    outdct = {'fpet_framed': fpet_nii}

    # > check if the static (for UR) file already exists
    if not fstat.is_file() or force:

        if nfrm > 1:
            imstat = np.sum(imdct['im'][frames, ...], axis=0)
        else:
            imstat = np.squeeze(imdct['im'])

        # > apply smoothing when requested
        if fwhm > 0:
            fwhmstr = str(fwhm).replace('.', '-')
            fur_smo = petout / (fname.split('.nii')[0] + f'_smo-{fwhmstr}.nii.gz')
            imsmo = nimpa.imsmooth(imstat, fwhm=fwhm, voxsize=imdct['voxsize'])

            nimpa.array2nii(
                imsmo, imdct['affine'], fur_smo,
                trnsp=(imdct['transpose'].index(0), imdct['transpose'].index(1),
                       imdct['transpose'].index(2)), flip=imdct['flip'])

            outdct['fur_smo'] = fur_smo

        nimpa.array2nii(
            imstat, imdct['affine'], fstat,
            trnsp=(imdct['transpose'].index(0), imdct['transpose'].index(1),
                   imdct['transpose'].index(2)), flip=imdct['flip'])

        outdct['fur'] = fstat

        log.info(f'Saved uptake ratio (UR) file image to: {fstat}')

        if com_correction:
            fur_com = nimpa.centre_mass_corr(fstat, outpath=petout)
            log.info(
                f'Centre-of-mass corrected uptake ratio (UR) image has been saved to: {fur_com}')
            outdct['fcom'] = fur_com['fim']
            outdct['com'] = fur_com['com_abs']

            # > the same for the smoothed
            if fwhm > 0:
                fur_smo_com = nimpa.centre_mass_corr(fur_smo, outpath=petout,
                                                     com=fur_com['com_abs'])
                outdct['fcom_smo'] = fur_smo_com['fim']

    # ------------------------------------------

    return outdct


# ========================================================================================
# Extract VOI values for uptake ratio (UR) analysis
# ========================================================================================


def voi_process(petpth, lblpth, t1wpth, voi_dct=None, ref_voi=None, voi_mask=None, frames=None, fname=None,
                pet_int_order=0, t1_bias_corr=True, outpath=None, output_masks=True, save_voi_masks=False,
                qc_plot=True, reg_fwhm_pet=0, reg_fwhm_mri=0, reg_costfun='nmi', reg_fresh=True,
                com_correction=True):
    ''' Process PET image for VOI extraction using MR-based parcellations.
        The T1w image and the labels which are based on the image must be
        in the same image space.

        Arguments:
        - petpth:   path to the PET NIfTI image
        - lblpth:   path to the label NIfTI image (parcellations)
        - t1wpth:   path to the T1w MRI NIfTI image for registration
        - voi_dct:  dictionary of VOI definitions
        - ref_voi:  if given and in `voi_dct` it is used as reference region
                    for calculating UR;
        - voi_mask: an additional mask on top of VOIs, e.g., to refine the GM
                    or to get rid of lesions;
        - frames:   select the frames if multi-frame image given;
                    by default selects all frames
        - fname:    the core file name for resulting images
        - t1_bias_corr: it True, performs bias field correction of the T1w image
        - outpath:  folder path to the output images, including intermediate
                    images
        - output_masks: if True, output VOI sampling masks in the output
                    dictionary
        - save_voi_masks: if True, saves all the VOI masks to the `masks` folder
        - qc_plot:  plots the PET images and overlay sampling, and saves it to
                    a PNG file; requires `output_masks` to be True.
        - reg_fwhm: FWHMs of the Gaussian filter applied to PET or MRI images
                    by default 0 mm;
        - reg_costfun: cost function used in image registration
        - reg_fresh:runs fresh registration if True, otherwise uses an existing
                    one if found.
        - com_correction: correction for centre of mass (image)

    '''

    # > output dictionary
    out = {}

    # > make sure the paths are Path objects
    petpth = Path(petpth)
    t1wpth = Path(t1wpth)
    lblpth = Path(lblpth)

    if outpath is None:
        outpath = petpth.parent
    else:
        outpath = Path(outpath)

    out['input'] = {'fpet': petpth, 'ft1w': t1wpth, 'flbl': lblpth}

    if not (petpth.exists() and t1wpth.is_file() and lblpth.is_file()):
        raise ValueError('One of the three paths to PET, T1w or label image is incorrect.')

    # > if dictionary is not given, the VOI values will be calculated for each unique
    # > VOI in the label/parcellation image
    if voi_dct is None:
        lbl = nimpa.getnii(lblpth)
        voi_dct = {int(lab): [int(lab)] for lab in np.unique(lbl)}

    if ref_voi is not None and not all(r in voi_dct for r in ref_voi):
        raise ValueError('Not all VOIs listed as reference are in the VOI definition dictionary.')

    # > static (UR) image preprocessing
    ur_preproc = preproc_ur(petpth, frames=frames,
                            outpath=outpath/'UR',
                            fname=fname,
                            com_correction=com_correction)

    out.update(ur_preproc)

    if t1_bias_corr:
        out['n4'] = nimpa.bias_field_correction(t1wpth, executable='sitk',
                                                outpath=ur_preproc['fur'].parent.parent)
        fmri = out['n4']['fim']
    else:
        fmri = t1wpth

    # --------------------------------------------------
    # TRIMMING / UPSCALING
    # > derive the scale of upscaling/trimming using the current
    # > image/voxel sizes
    trmout = r_trimup(ur_preproc['fur'], lblpth, store_img_intrmd=True, int_order=pet_int_order)

    # > trimmed folder
    trmdir = trmout['trmdir']

    # > trimmed and upsampled PET file
    out['ftrm'] = trmout['ftrm']
    out['trim_scale'] = trmout['trim_scale']
    # --------------------------------------------------

    # > - - - - - - - - - - - - - - - - - - - - - - - -
    # > parcellations in PET space
    fplbl = trmdir / '{}_Parcellation_in-upsampled-PET.nii.gz'.format(
        ur_preproc['fur'].name.split('.nii')[0])

    if not fplbl.is_file() or reg_fresh:

        log.info(f'registration with smoothing of {reg_fwhm_pet}, {reg_fwhm_mri} mm'
                 ' for reference and floating images respectively')

        spm_res = nimpa.coreg_spm(trmout['ftrm'], fmri, fwhm_ref=reg_fwhm_pet,
                                  fwhm_flo=reg_fwhm_mri, fwhm=[7, 7], costfun=reg_costfun,
                                  fcomment='', outpath=trmdir, visual=0, save_arr=False,
                                  del_uncmpr=True)

        flbl_pet = nimpa.resample_spm(
            trmout['ftrm'],
            lblpth,
            spm_res['faff'],
            outpath=trmdir,
            intrp=0.,
            fimout=fplbl,
            del_ref_uncmpr=True,
            del_flo_uncmpr=True,
            del_out_uncmpr=True,
        )

    out['flbl'] = fplbl
    out['faff'] = spm_res['faff']
    # > - - - - - - - - - - - - - - - - - - - - - - - -

    # > get the label image in PET space
    plbl_dct = nimpa.getnii(fplbl, output='all')

    # > get the sampling output
    if save_voi_masks:
        mask_dir = trmdir / 'masks'
    else:
        mask_dir = None
    voival = extract_vois(trmout['im'], plbl_dct, voi_dct,
                atlas_mask=voi_mask,
                outpath=mask_dir, output_masks=output_masks)

    # > calculate UR if reference regions is given
    urtxt = None
    if ref_voi is not None:

        ur = {}

        urtxt = ' '
        for rvoi in ref_voi:
            ref = voival[rvoi]['avg']
            ur[rvoi] = {}
            for voi in voi_dct:
                ur[rvoi][voi] = voival[voi]['avg'] / ref

            # > get the static trimmed image:
            imur = nimpa.getnii(out['ftrm'], output='all')

            fur = trmdir / 'UR_ref-{}_{}'.format(rvoi, ur_preproc['fur'].name)
            # > save UR image
            nimpa.array2nii(
                imur['im'] / ref, imur['affine'], fur,
                trnsp=(imur['transpose'].index(0), imur['transpose'].index(1),
                       imur['transpose'].index(2)), flip=imur['flip'])

            ur[rvoi]['fur'] = fur

            if 'ur' in voi_dct:
                urval = ur[rvoi]['ur']
                urtxt += f'$UR_\\mathrm{{{rvoi}}}=${urval:.3f}; '

        out['ur'] = ur

    out['vois'] = voival

    # -----------------------------------------
    # > QC plot
    if qc_plot and output_masks:
        showpet = nimpa.imsmooth(trmout['im'].astype(np.float32), voxsize=plbl_dct['voxsize'],
                                 fwhm=3.)

        def axrange(prf, thrshld, parts):
            zs = next(x for x, val in enumerate(prf) if val > thrshld)
            ze = len(prf) - next(x for x, val in enumerate(prf[::-1]) if val > thrshld)
            # divide the range in parts
            p = int((ze-zs) / parts)
            zn = []
            for k in range(1, parts):
                zn.append(zs + k*p)
            return zn

        # z-profile
        zn = []
        thrshld = 100
        zprf = np.sum(voival['neocx']['roimsk'], axis=(1, 2))
        zn += axrange(zprf, thrshld, 3)

        zprf = np.sum(voival['cblgm']['roimsk'], axis=(1, 2))
        zn += axrange(zprf, thrshld, 2)

        mskshow = voival['neocx']['roimsk'] + voival['cblgm']['roimsk']

        xn = []
        xprf = np.sum(mskshow, axis=(0, 1))
        xn += axrange(xprf, thrshld, 4)

        fig, ax = plt.subplots(2, 3, figsize=(16, 16))

        for ai, zidx in enumerate(zn):
            msk = mskshow[zidx, ...]
            impet = showpet[zidx, ...]
            ax[0][ai].imshow(impet, cmap='magma', vmax=0.9 * impet.max())
            ax[0][ai].imshow(msk, cmap='gray_r', alpha=0.25)
            ax[0][ai].xaxis.set_visible(False)
            ax[0][ai].yaxis.set_visible(False)

        for ai, xidx in enumerate(xn):
            msk = mskshow[..., xidx]
            impet = showpet[..., xidx]
            ax[1][ai].imshow(impet, cmap='magma', vmax=0.9 * impet.max())
            ax[1][ai].imshow(msk, cmap='gray_r', alpha=0.25)
            ax[1][ai].xaxis.set_visible(False)
            ax[1][ai].yaxis.set_visible(False)

        ax[0, 1].text(0, trmout['im'].shape[1] + 10, urtxt, fontsize=12)

        plt.tight_layout()

        fqc = trmdir / f'QC_{petpth.name}_Parcellation-over-upsampled-PET.png'
        plt.savefig(fqc, dpi=300)
        plt.close('all')
        out['fqc'] = fqc
    # -----------------------------------------

    return out
