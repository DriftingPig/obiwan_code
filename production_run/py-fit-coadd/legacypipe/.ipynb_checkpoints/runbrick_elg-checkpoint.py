'''
Main "pipeline" script for the Legacy Survey (DECaLS, MzLS, BASS)
data reductions.

For calling from other scripts, see:

- :py:func:`run_brick`

Or for much more fine-grained control, see the individual stages:

- :py:func:`stage_tims`
- :py:func:`stage_refs`
- :py:func:`stage_outliers`
- :py:func:`stage_halos`
- :py:func:`stage_fit_on_coadds [optional]`
- :py:func:`stage_image_coadds`
- :py:func:`stage_srcs`
- :py:func:`stage_fitblobs`
- :py:func:`stage_coadds`
- :py:func:`stage_wise_forced`
- :py:func:`stage_galex_forced` [optional]
- :py:func:`stage_writecat`

To see the code we run on each "blob" of pixels, see "oneblob.py".

- :py:func:`one_blob`

'''
from __future__ import print_function
import sys
import os

import numpy as np

import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.ttime import Time

from legacypipe.survey import get_rgb, imsave_jpeg
from legacypipe.bits import DQ_BITS, MASKBITS, FITBITS
from legacypipe.utils import RunbrickError, NothingToDoError, iterwrapper, find_unique_pixels
from legacypipe.coadds import make_coadds, write_coadd_images, quick_coadds

from legacypipe.fit_on_coadds import stage_fit_on_coadds
from legacypipe.galex import stage_galex_forced
from legacypipe.survey import *
from tractor import *
from tractor.sfd import SFDMap
from tractor.galaxy import DevGalaxy, ExpGalaxy
from tractor.sersic import SersicGalaxy
from legacypipe.survey import LegacySersicIndexSim
import tractor.ellipses as ellipses
import tractor
from tractor.basics import RaDecPos
import galsim
import logging
logger = logging.getLogger('legacypipe.runbrick')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

def runbrick_global_init():
    from tractor.galaxy import disable_galaxy_cache
    info('Starting process', os.getpid(), Time()-Time())
    disable_galaxy_cache()

def stage_tims(W=3600, H=3600, pixscale=0.262, brickname=None,
               survey=None,
               survey_blob_mask=None,
               ra=None, dec=None,
               release=None,
               plots=False, ps=None,
               target_extent=None, program_name='runbrick.py',
               bands=None,
               do_calibs=True,
               old_calibs_ok=True,
               splinesky=True,
               subsky=True,
               gaussPsf=False, pixPsf=False, hybridPsf=False,
               normalizePsf=False,
               apodize=False,
               constant_invvar=False,
               read_image_pixels = True,
               min_mjd=None, max_mjd=None,
               gaia_stars=True,
               mp=None,
               record_event=None,
               unwise_dir=None,
               unwise_tr_dir=None,
               unwise_modelsky_dir=None,
               galex_dir=None,
               command_line=None,
               read_parallel=True,
               **kwargs):
    '''
    This is the first stage in the pipeline.  It
    determines which CCD images overlap the brick or region of
    interest, runs calibrations for those images if necessary, and
    then reads the images, creating `tractor.Image` ("tractor image"
    or "tim") objects for them.

    PSF options:

    - *gaussPsf*: boolean.  Single-component circular Gaussian, with
      width set from the header FWHM value.  Useful for quick
      debugging.

    - *pixPsf*: boolean.  Pixelized PsfEx model.

    - *hybridPsf*: boolean.  Hybrid Pixelized PsfEx / Gaussian approx model.

    Sky:

    - *splinesky*: boolean.  If we have to create sky calibs, create SplineSky model rather than ConstantSky?
    - *subsky*: boolean.  Subtract sky model from tims?

    '''
    from legacypipe.survey import (
        get_git_version, get_version_header, get_dependency_versions,
        wcs_for_brick, read_one_tim)
    from astrometry.util.starutil_numpy import ra2hmsstring, dec2dmsstring

    tlast = Time()
    record_event and record_event('stage_tims: starting')

    assert(survey is not None)

    if bands is None:
        bands = ['g','r','z']

    # Get brick object
    custom_brick = (ra is not None)
    if custom_brick:
        from legacypipe.survey import BrickDuck
        # Custom brick; create a fake 'brick' object
        brick = BrickDuck(ra, dec, brickname)
    else:
        brick = survey.get_brick_by_name(brickname)
        if brick is None:
            raise RunbrickError('No such brick: "%s"' % brickname)
    brickid = brick.brickid
    brickname = brick.brickname

    # Get WCS object describing brick
    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    if target_extent is not None:
        (x0,x1,y0,y1) = target_extent
        W = x1-x0
        H = y1-y0
        targetwcs = targetwcs.get_subimage(x0, y0, W, H)
    pixscale = targetwcs.pixel_scale()
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])
    # custom brick -- set RA,Dec bounds
    if custom_brick:
        brick.ra1,_  = targetwcs.pixelxy2radec(W, H/2)
        brick.ra2,_  = targetwcs.pixelxy2radec(1, H/2)
        _, brick.dec1 = targetwcs.pixelxy2radec(W/2, 1)
        _, brick.dec2 = targetwcs.pixelxy2radec(W/2, H)

    # Create FITS header with version strings
    gitver = get_git_version()

    version_header = get_version_header(program_name, survey.survey_dir, release,
                                        git_version=gitver)

    deps = get_dependency_versions(unwise_dir, unwise_tr_dir, unwise_modelsky_dir, galex_dir)
    for name,value,comment in deps:
        version_header.add_record(dict(name=name, value=value, comment=comment))
    if command_line is not None:
        version_header.add_record(dict(name='CMDLINE', value=command_line,
                                       comment='runbrick command-line'))
    version_header.add_record(dict(name='BRICK', value=brickname,
                                comment='LegacySurveys brick RRRr[pm]DDd'))
    version_header.add_record(dict(name='BRICKID' , value=brickid,
                                comment='LegacySurveys brick id'))
    version_header.add_record(dict(name='RAMIN'   , value=brick.ra1,
                                comment='Brick RA min (deg)'))
    version_header.add_record(dict(name='RAMAX'   , value=brick.ra2,
                                comment='Brick RA max (deg)'))
    version_header.add_record(dict(name='DECMIN'  , value=brick.dec1,
                                comment='Brick Dec min (deg)'))
    version_header.add_record(dict(name='DECMAX'  , value=brick.dec2,
                                comment='Brick Dec max (deg)'))
    # Add NOAO-requested headers
    version_header.add_record(dict(
        name='RA', value=ra2hmsstring(brick.ra, separator=':'), comment='Brick center RA (hms)'))
    version_header.add_record(dict(
        name='DEC', value=dec2dmsstring(brick.dec, separator=':'), comment='Brick center DEC (dms)'))
    version_header.add_record(dict(
        name='CENTRA', value=brick.ra, comment='Brick center RA (deg)'))
    version_header.add_record(dict(
        name='CENTDEC', value=brick.dec, comment='Brick center Dec (deg)'))
    for i,(r,d) in enumerate(targetrd[:4]):
        version_header.add_record(dict(
            name='CORN%iRA' %(i+1), value=r, comment='Brick corner RA (deg)'))
        version_header.add_record(dict(
            name='CORN%iDEC'%(i+1), value=d, comment='Brick corner Dec (deg)'))

    # Find CCDs
    ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)
    if ccds is None:
        raise NothingToDoError('No CCDs touching brick')
    debug(len(ccds), 'CCDs touching target WCS')

    if 'ccd_cuts' in ccds.get_columns():
        ccds.cut(ccds.ccd_cuts == 0)
        debug(len(ccds), 'CCDs survive cuts')
    else:
        print('WARNING: not applying CCD cuts')

    # Cut on bands to be used
    ccds.cut(np.array([b in bands for b in ccds.filter]))
    debug('Cut to', len(ccds), 'CCDs in bands', ','.join(bands))

    debug('Cutting on CCDs to be used for fitting...')
    I = survey.ccds_for_fitting(brick, ccds)
    if I is not None:
        debug('Cutting to', len(I), 'of', len(ccds), 'CCDs for fitting.')
        ccds.cut(I)

    if min_mjd is not None:
        ccds.cut(ccds.mjd_obs >= min_mjd)
        debug('Cut to', len(ccds), 'after MJD', min_mjd)
    if max_mjd is not None:
        ccds.cut(ccds.mjd_obs <= max_mjd)
        debug('Cut to', len(ccds), 'before MJD', max_mjd)

    # Create Image objects for each CCD
    ims = []
    info('Keeping', len(ccds), 'CCDs:')
    for ccd in ccds:
        im = survey.get_image_object(ccd)
        if survey.cache_dir is not None:
            im.check_for_cached_files(survey)
        ims.append(im)
        info(' ', im, im.band, 'expnum', im.expnum, 'exptime', im.exptime, 'propid', ccd.propid,
              'seeing %.2f' % (ccd.fwhm*im.pixscale), 'MJD %.3f' % ccd.mjd_obs,
              'object', getattr(ccd, 'object', '').strip(), '\n   ', im.print_imgpath)

    tnow = Time()
    debug('Finding images touching brick:', tnow-tlast)
    tlast = tnow

    if do_calibs:
        from legacypipe.survey import run_calibs
        record_event and record_event('stage_tims: starting calibs')
        kwa = dict(git_version=gitver, survey=survey,
                   old_calibs_ok=old_calibs_ok,
                   survey_blob_mask=survey_blob_mask)
        if gaussPsf:
            kwa.update(psfex=False)
        if splinesky:
            kwa.update(splinesky=True)
        if not gaia_stars:
            kwa.update(gaia=False)

        # Run calibrations
        args = [(im, kwa) for im in ims]
        mp.map(run_calibs, args)
        tnow = Time()
        debug('Calibrations:', tnow-tlast)
        tlast = tnow

    # Read Tractor images
    args = [(im, targetrd, dict(gaussPsf=gaussPsf, pixPsf=pixPsf,
                                hybridPsf=hybridPsf, normalizePsf=normalizePsf,
                                subsky=subsky,
                                apodize=apodize,
                                constant_invvar=constant_invvar,
                                pixels=read_image_pixels,
                                old_calibs_ok=old_calibs_ok))
                                for im in ims]
    record_event and record_event('stage_tims: starting read_tims')
    if read_parallel:
        tims = list(mp.map(read_one_tim, args))
    else:
        tims = list(map(read_one_tim, args))
    record_event and record_event('stage_tims: done read_tims')

    tnow = Time()
    debug('Read', len(ccds), 'images:', tnow-tlast)
    tlast = tnow

    # Cut the table of CCDs to match the 'tims' list
    I = np.array([i for i,tim in enumerate(tims) if tim is not None])
    ccds.cut(I)
    tims = [tim for tim in tims if tim is not None]
    assert(len(ccds) == len(tims))
    if len(tims) == 0:
        raise NothingToDoError('No photometric CCDs touching brick.')

    # Check calibration product versions
    for tim in tims:
        for cal,ver in [('sky', tim.skyver), ('psf', tim.psfver)]:
            if tim.plver.strip() != ver[1].strip():
                print(('Warning: image "%s" PLVER is "%s" but %s calib was run'
                      +' on PLVER "%s"') % (str(tim), tim.plver, cal, ver[1]))

    # Add additional columns to the CCDs table.
    ccds.ccd_x0 = np.array([tim.x0 for tim in tims]).astype(np.int16)
    ccds.ccd_y0 = np.array([tim.y0 for tim in tims]).astype(np.int16)
    ccds.ccd_x1 = np.array([tim.x0 + tim.shape[1]
                            for tim in tims]).astype(np.int16)
    ccds.ccd_y1 = np.array([tim.y0 + tim.shape[0]
                            for tim in tims]).astype(np.int16)
    rd = np.array([[tim.subwcs.pixelxy2radec(1, 1)[-2:],
                    tim.subwcs.pixelxy2radec(1, y1-y0)[-2:],
                    tim.subwcs.pixelxy2radec(x1-x0, 1)[-2:],
                    tim.subwcs.pixelxy2radec(x1-x0, y1-y0)[-2:]]
                    for tim,x0,y0,x1,y1 in
                    zip(tims, ccds.ccd_x0+1, ccds.ccd_y0+1,
                        ccds.ccd_x1, ccds.ccd_y1)])
    _,x,y = targetwcs.radec2pixelxy(rd[:,:,0], rd[:,:,1])
    ccds.brick_x0 = np.floor(np.min(x, axis=1)).astype(np.int16)
    ccds.brick_x1 = np.ceil (np.max(x, axis=1)).astype(np.int16)
    ccds.brick_y0 = np.floor(np.min(y, axis=1)).astype(np.int16)
    ccds.brick_y1 = np.ceil (np.max(y, axis=1)).astype(np.int16)
    ccds.psfnorm = np.array([tim.psfnorm for tim in tims])
    ccds.galnorm = np.array([tim.galnorm for tim in tims])
    ccds.propid = np.array([tim.propid for tim in tims])
    ccds.plver  = np.array([tim.plver for tim in tims])
    ccds.skyver = np.array([tim.skyver[0] for tim in tims])
    ccds.psfver = np.array([tim.psfver[0] for tim in tims])
    ccds.skyplver = np.array([tim.skyver[1] for tim in tims])
    ccds.psfplver = np.array([tim.psfver[1] for tim in tims])

    # Cut "bands" down to just the bands for which we have images.
    timbands = [tim.band for tim in tims]
    bands = [b for b in bands if b in timbands]
    debug('Cut bands to', bands)

    if plots:
        from legacypipe.runbrick_plots import tim_plots
        tim_plots(tims, bands, ps)

    # Add header cards about which bands and cameras are involved.
    for band in 'grz':
        hasit = band in bands
        version_header.add_record(dict(
            name='BRICK_%s' % band.upper(), value=hasit,
            comment='Does band %s touch this brick?' % band))

        cams = np.unique([tim.imobj.camera for tim in tims
                          if tim.band == band])
        version_header.add_record(dict(
            name='CAMS_%s' % band.upper(), value=' '.join(cams),
            comment='Cameras contributing band %s' % band))
    version_header.add_record(dict(name='BANDS', value=''.join(bands),
                                   comment='Bands touching this brick'))
    version_header.add_record(dict(name='NBANDS', value=len(bands),
                                   comment='Number of bands in this catalog'))
    for i,band in enumerate(bands):
        version_header.add_record(dict(name='BAND%i' % i, value=band,
                                       comment='Band name in this catalog'))

    _add_stage_version(version_header, 'TIMS', 'tims')
    keys = ['version_header', 'targetrd', 'pixscale', 'targetwcs', 'W','H',
            'tims', 'ps', 'brickid', 'brickname', 'brick', 'custom_brick',
            'target_extent', 'ccds', 'bands', 'survey']
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def _add_stage_version(version_header, short, stagename):
    from legacypipe.survey import get_git_version
    version_header.add_record(dict(name='VER_%s'%short, value=get_git_version(),
                                   help='legacypipe version for stage_%s'%stagename))

def stage_refs(survey=None,
               brick=None,
               brickname=None,
               brickid=None,
               pixscale=None,
               targetwcs=None,
               bands=None,
               version_header=None,
               tycho_stars=True,
               gaia_stars=True,
               large_galaxies=True,
               star_clusters=True,
               plots=False, ps=None,
               record_event=None,
               **kwargs):
    
    record_event and record_event('stage_refs: starting')
    _add_stage_version(version_header, 'REFS', 'refs')

    refstars=None
    refcat=None
    T_dup = None
    T_clusters = None

    keys = ['refstars', 'gaia_stars', 'T_dup', 'T_clusters', 'version_header',
            'refcat']
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def stage_outliers(tims=None, targetwcs=None, W=None, H=None, bands=None,
                   mp=None, nsigma=None, plots=None, ps=None, record_event=None,
                   survey=None, brickname=None, version_header=None,
                   refstars=None, outlier_mask_file=None,
                   outliers=True, cache_outliers=False,
                   **kwargs):
    '''This pipeline stage tries to detect artifacts in the individual
    exposures, by blurring all images in the same band to the same PSF size,
    then searching for outliers.

    *cache_outliers*: bool: if the outliers-mask*.fits.fz file exists
    (from a previous run), use it.  We turn this off in production
    because we still want to create the JPEGs and the checksum entry
    for the outliers file.
    '''
    outliers=False 
    from legacypipe.outliers import patch_from_coadd, mask_outlier_pixels, read_outlier_mask_file

    record_event and record_event('stage_outliers: starting')
    _add_stage_version(version_header, 'OUTL', 'outliers')

    version_header.add_record(dict(name='OUTLIER',
                                   value=outliers,
                                   help='Are we applying outlier rejection?'))

    return dict(tims=tims, version_header=version_header)

def stage_halos(pixscale=None, targetwcs=None,
                W=None,H=None,
                bands=None, ps=None, tims=None,
                plots=False, plots2=False,
                brickname=None,
                version_header=None,
                mp=None, nsigma=None,
                survey=None, brick=None,
                refstars=None,
                star_halos=True,
                record_event=None,
                **kwargs):
    record_event and record_event('stage_halos: starting')
    _add_stage_version(version_header, 'HALO', 'halos')

    return dict(tims=tims, version_header=version_header)

def stage_image_coadds(survey=None, targetwcs=None, bands=None, tims=None,
                       brickname=None, version_header=None,
                       plots=False, ps=None, coadd_bw=False, W=None, H=None,
                       brick=None, blobmap=None, lanczos=True, ccds=None,
                       write_metrics=True,
                       mp=None, record_event=None,
                       co_sky=None,
                       custom_brick=False,
                       refstars=None,
                       T_clusters=None,
                       saturated_pix=None,
                       less_masking=False,
                       **kwargs):
    record_event and record_event('stage_image_coadds: starting')
    '''
    Immediately after reading the images, we can create coadds of just
    the image products.  Later, full coadds including the models will
    be created (in `stage_coadds`).  But it's handy to have the coadds
    early on, to diagnose problems or just to look at the data.
    '''
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=version_header)

    C = make_coadds(tims, bands, targetwcs,
                    detmaps=True, ngood=True, lanczos=lanczos,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, tims,
                                   targetwcs, co_sky),
                    mp=mp, plots=plots, ps=ps)

    # interim maskbits
    from legacypipe.utils import copy_header_with_wcs
    from legacypipe.bits import IN_BLOB
    refmap = get_blobiter_ref_map(refstars, T_clusters, less_masking, targetwcs)
    # Construct a mask bits map
    maskbits = np.zeros((H,W), np.int16)
    # !PRIMARY
    if not custom_brick:
        U = find_unique_pixels(targetwcs, W, H, None,
                               brick.ra1, brick.ra2, brick.dec1, brick.dec2)
        maskbits |= MASKBITS['NPRIMARY'] * np.logical_not(U).astype(np.int16)
        del U
    # BRIGHT
    if refmap is not None:
        maskbits |= MASKBITS['BRIGHT']  * ((refmap & IN_BLOB['BRIGHT'] ) > 0)
        maskbits |= MASKBITS['MEDIUM']  * ((refmap & IN_BLOB['MEDIUM'] ) > 0)
        maskbits |= MASKBITS['GALAXY']  * ((refmap & IN_BLOB['GALAXY'] ) > 0)
        maskbits |= MASKBITS['CLUSTER'] * ((refmap & IN_BLOB['CLUSTER']) > 0)
        del refmap
    # SATUR
    if saturated_pix is not None:
        for b, sat in zip(bands, saturated_pix):
            maskbits |= (MASKBITS['SATUR_' + b.upper()] * sat).astype(np.int16)
    # ALLMASK_{g,r,z}
    for b,allmask in zip(bands, C.allmasks):
        maskbits |= (MASKBITS['ALLMASK_' + b.upper()] * (allmask > 0))
    # omitting maskbits header cards, bailout, & WISE
    hdr = copy_header_with_wcs(version_header, targetwcs)
    with survey.write_output('maskbits', brick=brickname, shape=maskbits.shape) as out:
        out.fits.write(maskbits, header=hdr, extname='MASKBITS')

    # Sims: coadds of galaxy sims only, image only
    if hasattr(tims[0], 'sims_image'):
        sims_coadd,_ = quick_coadds(
            tims, bands, targetwcs, images=[tim.sims_image for tim in tims])
    

    # obiwan  
    # IDs of simulated galaxies touching ccds
    if hasattr(tims[0], 'ids_added'):
        ids_added= set()
        for tim in tims:
            print(tim.ids_added)
            ids_added= ids_added.union(set(tim.ids_added))
            a=fits_table()
            a.set('id',np.array(list(ids_added)))
            a_fn= survey.find_file('image-jpeg',brick=brickname,output=True)
            a_fn= a_fn.replace('legacysurvey-%s-image.jpg' % brickname,
                    'sim_ids_added.fits')
            a.writeto(a_fn)
            print('Wrote %s' % a_fn)

    D = _depth_histogram(brick, targetwcs, bands, C.psfdetivs, C.galdetivs)
    with survey.write_output('depth-table', brick=brickname) as out:
        D.writeto(None, fits_object=out.fits)
    del D

    # Write per-brick CCDs table
    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='ccdinfo',
                            comment='NOAO data product type'))
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=primhdr)

    coadd_list= [('image', C.coimgs)]
    if hasattr(tims[0], 'sims_image'):
        coadd_list.append(('simscoadd', sims_coadd))

    for name,ims in coadd_list:
        rgb = get_rgb(ims, bands)
        kwa = {}
        if coadd_bw and len(bands) == 1:
            rgb = rgb.sum(axis=2)
            kwa = dict(cmap='gray')

        with survey.write_output(name + '-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
            debug('Wrote', out.fn)

        # Blob-outlined version
        if blobmap is not None:
            from scipy.ndimage.morphology import binary_dilation
            outline = np.logical_xor(
                binary_dilation(blobmap >= 0, structure=np.ones((3,3))),
                (blobmap >= 0))
            # coadd_bw
            if len(rgb.shape) == 2:
                rgb = np.repeat(rgb[:,:,np.newaxis], 3, axis=2)
            # Outline in green
            rgb[:,:,0][outline] = 0
            rgb[:,:,1][outline] = 1
            rgb[:,:,2][outline] = 0

            with survey.write_output(name+'blob-jpeg', brick=brickname) as out:
                imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
                debug('Wrote', out.fn)

            # write out blob map
            if write_metrics:
                hdr = copy_header_with_wcs(version_header, targetwcs)
                hdr.add_record(dict(name='IMTYPE', value='blobmap',
                                    comment='LegacySurveys image type'))
                with survey.write_output('blobmap', brick=brickname,
                                         shape=blobmap.shape) as out:
                    out.fits.write(blobmap, header=hdr)
        del rgb
    return None

def stage_srcs(pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               brickname=None,
               version_header=None,
               mp=None, nsigma=None,
               saddle_fraction=None,
               saddle_min=None,
               survey=None, brick=None,
               refcat=None, refstars=None,
               T_clusters=None,
               ccds=None,
               record_event=None,
               large_galaxies=True,
               gaia_stars=True,
               **kwargs):
    '''
    In this stage we run SED-matched detection to find objects in the
    images.  For each object detected, a `tractor` source object is
    created, initially a `tractor.PointSource`.  In this stage, the
    sources are also split into "blobs" of overlapping pixels.  Each
    of these blobs will be processed independently.
    '''
    from functools import reduce
    from tractor import Catalog
    from legacypipe.detection import (detection_maps,
                        run_sed_matched_filters, segment_and_group_sources)
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.measurements import label

    record_event and record_event('stage_srcs: starting')
    _add_stage_version(version_header, 'SRCS', 'srcs')
    print("running elgs")
    
    for tim in tims:
        tim.invvar_old = tim.getInvvar().copy()
    import os
    random_fn = os.environ["RANDOMS_FROM_FITS"]
    print("%s"%random_fn)
    samp = fits_table(random_fn)
    from runbrick_sim_elg import build_simcat
    cat,_ = build_simcat(samp,brickname)
    survey.simcat = cat
    simcat_new = survey.simcat.copy() 
    simcat_new.chi2_old = np.zeros_like(simcat_new.ra)
    simcat_new.chi2_new = np.zeros_like(simcat_new.ra)
    #the fitted result
    simcat_new.current_n = -np.ones_like(simcat_new.ra)
    simcat_new.current_e1 = -np.ones_like(simcat_new.ra)
    simcat_new.current_e2 = -np.ones_like(simcat_new.ra)
    simcat_new.current_rhalf = -np.ones_like(simcat_new.ra)
    #the goal
    simcat_new.goal_n = np.zeros_like(simcat_new.ra)
    simcat_new.goal_e1 = np.zeros_like(simcat_new.ra)
    simcat_new.goal_e2 = np.zeros_like(simcat_new.ra)
    simcat_new.goal_rhalf = np.zeros_like(simcat_new.ra)
    #original fitted result
    simcat_new.orign_n = survey.simcat.n.copy()
    simcat_new.origin_e1 = survey.simcat.e1.copy() 
    simcat_new.origin_e2 = survey.simcat.e2.copy() 
    simcat_new.origin_rhalf = survey.simcat.rhalf.copy() 
    simcat_new.fitted = np.zeros_like(simcat_new.ra,dtype=np.bool)
    simcat_new.same_model = np.zeros_like(simcat_new.ra,dtype=np.bool)
    #result with no refit
    simcat_new.current0_n = -np.ones_like(simcat_new.ra)
    simcat_new.current0_e1 = -np.ones_like(simcat_new.ra)
    simcat_new.current0_e2 = -np.ones_like(simcat_new.ra)
    simcat_new.current0_rhalf = -np.ones_like(simcat_new.ra)
    simcat_new.n0 = -np.ones_like(simcat_new.ra)
    simcat_new.e10 = -np.ones_like(simcat_new.ra)
    simcat_new.e20 = -np.ones_like(simcat_new.ra)
    simcat_new.rhalf0 = -np.ones_like(simcat_new.ra)
    force_model = None
    N_tot = len(survey.simcat)
    i=0
    re_fit = False
    while i<N_tot:
        if re_fit is True:
            print("refitting")
            i = i-1
        obj_origin = survey.simcat[i].copy()
        if obj_origin.n<0.5:
            i += 1
            continue
        simcat_new.fitted[i] = True
        obj = obj_origin.copy()
        obj_origin.e = np.hypot(survey.simcat[i].e1.copy(), survey.simcat[i].e2.copy())
        N_fail=0
        chi2_old = None 
        maxiter=0
        print("idx:: %d"%i) 

        sersic_n_current = -1
        e1_current = -1 
        e2_current = -1 
        rhalf_current = -1 

        while N_fail<3 and maxiter<60:
            maxiter+=1
            obj_old = obj.copy()
            for tim in tims:
                sim_img = SimImageNew(survey)
                sim_img.set_tim_img(tim,obj)
                hot = np.zeros((3600,3600),dtype=np.bool)
                idx = np.where(tim.data>0)
                hot[idx] = True
                #add gaussian sim noise (or do this with invvar?)
                #TODO
                err = 1./np.sqrt(tim.invvar_old)
                err[np.where(tim.invvar_old == 0)] = 0
                tim.data += np.random.normal(size=tim.shape)*err
                #tim.data += np.random.normal(size=tim.shape)*tim.sig1
  
            T = fits_table()
            T.ra = np.array([obj.ra],dtype=np.float64)
            T.dec = np.array([obj.dec],dtype=np.float64)
            T.ibx = np.array([int(obj.x+0.5)])
            T.iby = np.array([int(obj.y+0.5)])
            T.ref_cat = np.array(['  '])
            T.ref_id  = np.array([0])
    
            brightness =  tractor.NanoMaggies(g=np.float64(obj.gflux), r=np.float64(obj.rflux),z=np.float64(obj.zflux), order=['g','r','z'])
            cat = Catalog(*[PointSource(RaDecPos(T.ra[0], T.dec[0]), brightness)])
            cat.freezeAllParams()
       
            #blobmap,blobsrcs,blobslices = segment_and_group_sources(hot, T, name=brickname,
            #                                                 ps=ps, plots=plots)
 
            blobmap = np.int32(np.zeros((3600,3600)))
            blobmap[~hot] = -1
            blobsrcs = [np.array([0])]
            x,y = np.where(hot)
            blobslices = [(slice(x.min(),x.max()+1,None),slice(y.min(),y.max()+1,None))]
            del hot

            reoptimize=False
            iterative=True
            use_ceres=False
            refmap = np.zeros((3600,3600),dtype=np.bool)
            large_galaxies_force_pointsource=False
            less_masking=False 
            frozen_galaxies={}
            max_blobsize=None 
            custom_brick=False 
            skipblobs = []
            R = []
            #add gaussian noise to image here?
            blobiter = _blob_iter(brickname, blobslices, blobsrcs, blobmap, targetwcs, tims,
                          cat, bands, plots, ps, reoptimize, iterative, use_ceres,
                          refmap, large_galaxies_force_pointsource, less_masking, brick,
                          frozen_galaxies,
                          skipblobs=skipblobs,
                          single_thread=(mp is None or mp.pool is None),
                          max_blobsize=max_blobsize, custom_brick=custom_brick, force_model = force_model)

            blobiter = iterwrapper(blobiter, len(blobsrcs))


            R.extend(mp.map(_bounce_one_blob, blobiter))
            assert(len(R) == len(blobsrcs))
            # drop brickname,iblob
            R = [r['result'] for r in R]
            # Drop now-empty blobs.
            R = [r for r in R if r is not None and len(r)]
            #R[0].Isrcs; R[0].sources
            if len(R) == 0:
                simcat_new.n[i]=-999
                break
                raise NothingToDoError('No sources passed significance tests.')
            # Merge results R into one big table
            BB = merge_tables(R)
            del R
            # Pull out the source indices...
            II = BB.Isrcs
            newcat = BB.sources
            try:
                idx = np.where(II==0)[0][0]
            except:
                simcat_new.n[i]=-999
                break
            source_name = BB.sources[idx].getSourceType()
            if source_name != 'PointSource':
                rhalf = BB.sources[idx].shape.re
                e1 = BB.sources[idx].shape.e1
                e2 = BB.sources[idx].shape.e2
                e = np.hypot(e1,e2)
            else:
                rhalf = 0
                e1 = 0
                e2 = 0
                e = 0
            if source_name == 'RexGalaxy' or source_name == 'ExpGalaxy':
                sersic_n = 1
            if source_name == 'DevGalaxy':
                sersic_n = 4
            if source_name == 'PointSource':
                sersic_n = 0
            if source_name == 'SersicGalaxy':
                sersic_n = BB.sources[idx].sersicindex.val

            chi2_new = (sersic_n - obj_origin.n)**2/5.5 + (e - obj_origin.e)**2+(rhalf - obj_origin.rhalf)**2
         
            if chi2_old is None:
                chi2_old = chi2_new
                simcat_new.chi2_old[i] = chi2_old
                sersic_n_current = sersic_n 
                e1_current = e1 
                e2_current = e2 
                rhalf_current = rhalf 

                simcat_new.orign_n[i] = sersic_n
                simcat_new.origin_e1[i] = e1
                simcat_new.origin_e2[i] = e2
                simcat_new.origin_rhalf[i] = rhalf

            if chi2_new<1e-4:
                if chi2_new<chi2_old:
                    #ACCEPT
                    obj_old = obj
                    chi2_old = chi2_new
                    sersic_n_current = sersic_n 
                    e1_current = e1 
                    e2_current = e2 
                    rhalf_current = rhalf 
                break
            else:
                if chi2_new<=chi2_old:
                    N_fail=0
                    #ACCEPT
                    obj_old = obj
                    chi2_old = chi2_new
                    sersic_n_current = sersic_n 
                    e1_current = e1 
                    e2_current = e2 
                    rhalf_current = rhalf 
                else:
                    #DECLINE
                    N_fail+=1
            #construct new obj based on the current result
            print("old n: %f e1: %f e2:%f rhalf:%f"%(obj.n, obj.e1, obj.e2, obj.rhalf))
            if N_fail == 0:
                step = 0.2
                if chi2_new<1e-1:
                    step = 1
            elif N_fail == 1:
                step = 0.1
            else:
                step = 0.05
            obj.n -= (sersic_n - obj_origin.n)*step 
            old_e = np.hypot(obj.e1, obj.e2) 
            if obj_origin.e < 0.0001:
                #if input e is small, keep it small
                obj.e1 = obj_origin.e1
                obj.e2 = obj_origin.e2
            else:
                
                sign_e1 = obj_origin.e1/np.abs(obj_origin.e1)
                sign_e2 = obj_origin.e2/np.abs(obj_origin.e2)
                ratio_e1 = sign_e1*np.sqrt(obj_origin.e1**2/obj_origin.e**2)
                ratio_e2 = sign_e2*np.sqrt(obj_origin.e2**2/obj_origin.e**2)
                new_e = np.clip(0,0.999,np.hypot(obj.e1, obj.e2) - (e - obj_origin.e)*step)
                obj.e1 = new_e*ratio_e1
                obj.e2 = new_e*ratio_e2
                
            obj.rhalf = obj.rhalf - (rhalf - obj_origin.rhalf)*step
            print("new n: %f e1: %f e2:%f rhalf:%f"%(obj.n, obj.e1, obj.e2, obj.rhalf))
            print("current n: %f e1: %f e2:%f rhalf:%f"%(sersic_n, e1, e2, rhalf))
            print("goal n: %f e1: %f e2:%f rhalf:%f"%(obj_origin.n, obj_origin.e1, obj_origin.e2, obj_origin.rhalf))
        if simcat_new.n[i]==-999:
            #not getting detected, must be a noisy region, not using it
            print("bad source!")
            i+=1
            continue
        simcat_new.n[i] = obj_old.n 
        simcat_new.rhalf[i] = obj_old.rhalf 
        simcat_new.e1[i] = obj_old.e1 
        simcat_new.e2[i] = obj_old.e2 
        simcat_new.chi2_new[i] = chi2_old 
        simcat_new.current_n[i] = sersic_n_current  
        simcat_new.current_e1[i] = e1_current 
        simcat_new.current_e2[i] = e2_current 
        simcat_new.current_rhalf[i] = rhalf_current  
        simcat_new.goal_n[i] = obj_origin.n
        simcat_new.goal_e1[i] =  obj_origin.e1
        simcat_new.goal_e2[i] =  obj_origin.e2
        simcat_new.goal_rhalf[i] =  obj_origin.rhalf
        if re_fit is False:
            simcat_new.current0_n[i] = sersic_n_current  
            simcat_new.current0_e1[i] = e1_current 
            simcat_new.current0_e2[i] = e2_current 
            simcat_new.current0_rhalf[i] = rhalf_current  
            simcat_new.n0[i] = obj_old.n 
            simcat_new.e10[i] = obj_old.e1 
            simcat_new.e20[i] = obj_old.e2 
            simcat_new.rhalf0[i] = obj_old.rhalf 
        if re_fit is False and (simcat_new.goal_n[i]-simcat_new.current_n[i]==0):
            simcat_new.same_model[i] = True
        else:
            if re_fit is True:
                re_fit = False
                force_model = None
            else:
                re_fit = True
                if simcat_new.goal_n[i]==0:
                    force_model = 'psf'
                elif simcat_new.goal_n[i]==4:
                    force_model = 'dev'
                elif simcat_new.goal_n[i]==1 and simcat_new.goal_e1[i]==0 and simcat_new.goal_e2[i]==0:
                    force_model = 'rex'
                elif simcat_new.goal_n[i]==1:
                     force_model = 'exp'
                else:
                    force_model = 'ser'
        i += 1

    with survey.write_output('galaxy-sims', brick=brickname) as out:
            simcat_new.writeto(None, fits_object=out.fits)


    return None 

def stage_fitblobs(T=None,
                   T_clusters=None,
                   T_dup=None,
                   brickname=None,
                   brickid=None,
                   brick=None,
                   version_header=None,
                   blobsrcs=None, blobslices=None, blobmap=None,
                   cat=None,
                   targetwcs=None,
                   W=None,H=None,
                   bands=None, ps=None, tims=None,
                   survey=None,
                   plots=False, plots2=False,
                   nblobs=None, blob0=None, blobxy=None,
                   blobradec=None, blobid=None,
                   max_blobsize=None,
                   reoptimize=False,
                   iterative=False,
                   large_galaxies_force_pointsource=True,
                   less_masking=False,
                   use_ceres=True, mp=None,
                   checkpoint_filename=None,
                   checkpoint_period=600,
                   write_pickle_filename=None,
                   write_metrics=True,
                   get_all_models=False,
                   refstars=None,
                   bailout=False,
                   record_event=None,
                   custom_brick=False,
                   **kwargs):
    pass

    return None 


class SimImageNew(object):
    def __init__(self, survey):
        self.survey=survey
        brickname = survey.brick
        brick = survey.get_brick_by_name(brickname)
        brickid = brick.brickid
        brickname = brick.brickname
        brickwcs = wcs_for_brick(brick)
        W, H, pixscale = brickwcs.get_width(), brickwcs.get_height(), brickwcs.pixel_scale()
        targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
        self.targetwcs = targetwcs

    def set_tim_img(self, tim, obj):
        objstamp = BuildStamp(tim, targetwcs = self.targetwcs, camera = 'decam')
        
        #reset tim=0, make ivar super big
        tim.data = np.zeros_like(tim.data)
        #tim.setInvvar(np.ones_like(tim.data)*np.median(tim.getInvvar())*5000)
        tim.setInvvar(tim.invvar_old.copy())
        tim_image = galsim.Image(tim.getImage())
        tim_invvar = galsim.Image(tim.getInvvar())#invvar_old
        # Also store galaxy sims and sims invvar
        sims_image = tim_image.copy()
        sims_image.fill(0.0)
        sims_ivar = sims_image.copy()
       
        if True:
            strin= 'Drawing 1 galaxy: n=%.2f, rhalf=%.2f, e1=%.2f, e2=%.2f' % \
                        (obj.n,obj.rhalf,obj.e1,obj.e2)
            print(strin)
            stamp = objstamp.galaxy(obj)
            stamp_nonoise= stamp.copy()
            if self.survey.add_sim_noise:
                stamp += noise_for_galaxy(stamp,objstamp.nano2e)
            ivarstamp= ivar_for_galaxy(stamp,objstamp.nano2e)
            # Add source if EVEN 1 pix falls on the CCD
            overlap = stamp.bounds & tim_image.bounds
            if overlap.area() > 0:
                print('Stamp overlaps tim')
                stamp = stamp[overlap]
                ivarstamp = ivarstamp[overlap]
                stamp_nonoise= stamp_nonoise[overlap]

                # Zero out invvar where bad pixel mask is flagged (> 0)
                # Add stamp to image
                back= tim_image[overlap].copy()
                tim_image[overlap] += stamp #= back.copy() + stamp.copy()
                # Add variances
                back_ivar= tim_invvar[overlap].copy()
                tot_ivar= get_srcimg_invvar(ivarstamp, back_ivar)
                tim_invvar[overlap] = tot_ivar.copy()

        tim.data = tim_image.array
        tim.inverr = np.sqrt(tim_invvar.array)
        return tim

class BuildStamp():
    def __init__(self, tim, targetwcs=None, camera='decam'):
        self.nano2e = tim.nano2e
        self.targetwcs = targetwcs
        self.tim = tim
        self.band = tim.band

    # local image that the src resides in 
    def set_local(self, obj):
        ra, dec = obj.ra, obj.dec
        flag, target_x, target_y = self.targetwcs.radec2pixelxy(ra, dec)
        #intentionally setting source center in the center pixel on the coadd image
        #target_x, target_y = int(target_x+0.5), int(target_y+0.5)
        #ra_new, dec_new = self.targetwcs.pixelxy2radec(target_x, target_y)[-2:]
        
        x=int(obj.get('x')+0.5)
        y=int(obj.get('y')+0.5)
        self.target_x=x
        self.target_y=y
        
        self.ra, self.dec = ra, dec
        flag, xx, yy = self.tim.subwcs.radec2pixelxy(*(self.targetwcs.pixelxy2radec(self.target_x+1, self.target_y+1)[-2:]))
        x_cen = xx-1
        y_cen = yy-1
        self.wcs=self.tim.getWcs() 
        x_cen_int,y_cen_int = round(x_cen),round(y_cen)
        #self.sx0,self.sx1,self.sy0,self.sy1 = x_cen_int-32,x_cen_int+31,y_cen_int-32,y_cen_int+31
        self.sx0,self.sx1,self.sy0,self.sy1 = x_cen_int-64,x_cen_int+63,y_cen_int-64,y_cen_int+63
        (h,w) = self.tim.shape
        self.sx0 = np.clip(int(self.sx0), 0, w-1)
        self.sx1 = np.clip(int(self.sx1), 0, w-1) + 1
        self.sy0 = np.clip(int(self.sy0), 0, h-1)
        self.sy1 = np.clip(int(self.sy1), 0, h-1) + 1
        subslc = slice(self.sy0,self.sy1),slice(self.sx0,self.sx1)
        subimg = self.tim.getImage ()[subslc]
        subie  = self.tim.getInvError()[subslc]
        subwcs = self.tim.getWcs().shifted(self.sx0, self.sy0)
        subsky = self.tim.getSky().shifted(self.sx0, self.sy0)
    
        subpsf = self.tim.psf.constantPsfAt((self.sx0+self.sx1)/2., (self.sy0+self.sy1)/2.)
        new_tim = tractor.Image(data=subimg, inverr=subie, wcs=subwcs,psf=subpsf, photocal=self.tim.getPhotoCal(), sky=subsky, name=self.tim.name)
        new_tim.band = self.tim.band
        return new_tim
    def galaxy(self, obj):
        new_tim = self.set_local(obj)
        n,r_half,e1,e2,flux = float(obj.get('n')),float(obj.get('rhalf')),float(obj.get('e1')),float(obj.get('e2')),float(obj.get(self.band+'flux'))
        assert(self.band in ['g','r','z'])
        if self.band == 'g':
               brightness =  tractor.NanoMaggies(g=flux, order=['g'])
        if self.band == 'r':
                brightness =  tractor.NanoMaggies(r=flux, order=['r'])
        if self.band == 'z':
                brightness =  tractor.NanoMaggies(z=flux, order=['z'])
        shape = ellipses.EllipseE(r_half,e1,e2)
        if n==1:
              new_gal = ExpGalaxy(RaDecPos(self.ra, self.dec), brightness, shape)
        elif n==4:
              new_gal = DevGalaxy(RaDecPos(self.ra, self.dec), brightness, shape)
        elif n==0:
              new_gal = PointSource(RaDecPos(self.ra, self.dec), brightness)
        else:
              new_gal = SersicGalaxy(RaDecPos(self.ra, self.dec), brightness, shape, LegacySersicIndexSim(n))

        new_tractor = Tractor([new_tim], [new_gal])

        mod0 = new_tractor.getModelImage(0)
        galsim_img = galsim.Image(mod0)
        galsim_img.bounds.xmin=self.sx0+1
        galsim_img.bounds.xmax=self.sx1-1
        galsim_img.bounds.ymin=self.sy0+1
        galsim_img.bounds.ymax=self.sy1-1
        return galsim_img

def noise_for_galaxy(gal,nano2e):
    """Returns numpy array of noise in Img count units for gal in image cnt units"""
    # Noise model + no negative image vals when compute noise
    one_std_per_pix= gal.array.copy() # nanomaggies
    one_std_per_pix[one_std_per_pix < 0]=0
    # rescale
    one_std_per_pix *= nano2e # e-
    one_std_per_pix= np.sqrt(one_std_per_pix)
    num_stds= np.random.randn(one_std_per_pix.shape[0],one_std_per_pix.shape[1])
    #one_std_per_pix.shape, num_stds.shape
    noise= one_std_per_pix * num_stds
    # rescale
    noise /= nano2e #nanomaggies
    return noise

def ivar_for_galaxy(gal,nano2e):
    """Adds gaussian noise to perfect source

    Args:
        gal: galsim.Image() for source, UNITS: nanomags
        nano2e: factor to convert to e- (gal * nano2e has units e-)

    Returns:
        galsim.Image() of invvar for the source, UNITS: nanomags
    """
    var= gal.copy() * nano2e #e^2
    var.applyNonlinearity(np.abs)
    var /= nano2e**2 #nanomag^2
    var.invertSelf()
    return var

def get_srcimg_invvar(stamp_ivar,img_ivar):
    """stamp_ivar, img_ivar -- galsim Image objects"""
    # Use img_ivar when stamp_ivar == 0, both otherwise
    return img_ivar
    use_img_ivar= np.ones(img_ivar.array.shape).astype(bool)
    use_img_ivar[ stamp_ivar.array > 0 ] = False
    # First compute using both
    ivar= np.power(stamp_ivar.array.copy(), -1) + np.power(img_ivar.array.copy(), -1)
    ivar= np.power(ivar,-1)
    keep= np.ones(ivar.shape).astype(bool)
    keep[ (stamp_ivar.array > 0)*\
          (img_ivar.array > 0) ] = False
    ivar[keep] = 0.
    # Now use img_ivar only where need to
    ivar[ use_img_ivar ] = img_ivar.array.copy()[ use_img_ivar ]
    # return
    obj_ivar = stamp_ivar.copy()
    obj_ivar.fill(0.)
    obj_ivar+= ivar
    return obj_ivar

def _write_checkpoint(R, checkpoint_filename):
    from astrometry.util.file import pickle_to_file, trymakedirs
    d = os.path.dirname(checkpoint_filename)
    if len(d) and not os.path.exists(d):
        trymakedirs(d)
    fn = checkpoint_filename + '.tmp'
    pickle_to_file(R, fn)
    os.rename(fn, checkpoint_filename)
    debug('Wrote checkpoint to', checkpoint_filename)

def _check_checkpoints(R, blobslices, brickname):
    # Check that checkpointed blobids match our current set of blobs,
    # based on blob bounding-box.  This can fail if the code changes
    # between writing & reading the checkpoint, resulting in a
    # different set of detected sources.
    keepR = []
    for ri in R:
        brick = ri['brickname']
        iblob = ri['iblob']
        r = ri['result']

        if brick != brickname:
            print('Checkpoint brick mismatch:', brick, brickname)
            continue

        if r is None:
            pass
        else:
            if r.iblob != iblob:
                print('Checkpoint iblob mismatch:', r.iblob, iblob)
                continue
            if iblob >= len(blobslices):
                print('Checkpointed iblob', iblob, 'is too large! (>= %i)' % len(blobslices))
                continue
            if len(r) == 0:
                pass
            else:
                # expected bbox:
                sy,sx = blobslices[iblob]
                by0,by1,bx0,bx1 = sy.start, sy.stop, sx.start, sx.stop
                # check bbox
                rx0,ry0 = r.blob_x0[0], r.blob_y0[0]
                rx1,ry1 = rx0 + r.blob_width[0], ry0 + r.blob_height[0]
                if rx0 != bx0 or ry0 != by0 or rx1 != bx1 or ry1 != by1:
                    print('Checkpointed blob bbox', [rx0,rx1,ry0,ry1],
                          'does not match expected', [bx0,bx1,by0,by1], 'for iblob', iblob)
                    continue
        keepR.append(ri)
    return keepR

def _blob_iter(brickname, blobslices, blobsrcs, blobmap, targetwcs, tims, cat, bands,
               plots, ps, reoptimize, iterative, use_ceres, refmap,
               large_galaxies_force_pointsource, less_masking,
               brick, frozen_galaxies, single_thread=False,
               skipblobs=None, max_blobsize=None, custom_brick=False, force_model=None):
    '''
    *blobmap*: map, with -1 indicating no-blob, other values indexing *blobslices*,*blobsrcs*.
    '''
    from collections import Counter

    if skipblobs is None:
        skipblobs = []

    # sort blobs by size so that larger ones start running first
    blobvals = Counter(blobmap[blobmap>=0])
    blob_order = np.array([b for b,npix in blobvals.most_common()])
    del blobvals

    if custom_brick:
        U = None
    else:
        H,W = targetwcs.shape
        U = find_unique_pixels(targetwcs, W, H, None,
                               brick.ra1, brick.ra2, brick.dec1, brick.dec2)

    for nblob,iblob in enumerate(blob_order):
        if iblob in skipblobs:
            info('Skipping blob', iblob)
            continue

        bslc  = blobslices[iblob]
        Isrcs = blobsrcs  [iblob]
        assert(len(Isrcs) > 0)

        # blob bbox in target coords
        sy,sx = bslc
        by0,by1 = sy.start, sy.stop
        bx0,bx1 = sx.start, sx.stop
        blobh,blobw = by1 - by0, bx1 - bx0

        # Here we assume the "blobmap" array has been remapped so that
        # -1 means "no blob", while 0 and up label the blobs, thus
        # iblob equals the value in the "blobmap" map.
        blobmask = (blobmap[bslc] == iblob)
        # at least one pixel should be set!
        assert(np.any(blobmask))

        if U is not None:
            # If the blob is solely outside the unique region of this brick,
            # skip it!
            if np.all(U[bslc][blobmask] == False):
                info('Blob', nblob+1, 'is completely outside the unique region of this brick -- skipping')
                yield (brickname, iblob, None)
                continue

        # find one pixel within the blob, for debugging purposes
        onex = oney = None
        for y in range(by0, by1):
            ii = np.flatnonzero(blobmask[y-by0,:])
            if len(ii) == 0:
                continue
            onex = bx0 + ii[0]
            oney = y
            break

        npix = np.sum(blobmask)
        info(('Blob %i of %i, id: %i, sources: %i, size: %ix%i, npix %i, brick X: %i,%i, ' +
               'Y: %i,%i, one pixel: %i %i') %
              (nblob+1, len(blobslices), iblob, len(Isrcs), blobw, blobh, npix,
               bx0,bx1,by0,by1, onex,oney))

        if max_blobsize is not None and npix > max_blobsize:
            info('Number of pixels in blob,', npix, ', exceeds max blobsize', max_blobsize)
            yield (brickname, iblob, None)
            continue

        # Here we cut out subimages for the blob...
        rr,dd = targetwcs.pixelxy2radec([bx0,bx0,bx1,bx1],[by0,by1,by1,by0])
        subtimargs = []
        for tim in tims:
            h,w = tim.shape
            _,x,y = tim.subwcs.radec2pixelxy(rr,dd)
            sx0,sx1 = x.min(), x.max()
            sy0,sy1 = y.min(), y.max()
            #print('blob extent in pixel space of', tim.name, ': x',
            # (sx0,sx1), 'y', (sy0,sy1), 'tim shape', (h,w))
            if sx1 < 0 or sy1 < 0 or sx0 > w or sy0 > h:
                continue
            sx0 = int(np.clip(int(np.floor(sx0)), 0, w-1))
            sx1 = int(np.clip(int(np.ceil (sx1)), 0, w-1)) + 1
            sy0 = int(np.clip(int(np.floor(sy0)), 0, h-1))
            sy1 = int(np.clip(int(np.ceil (sy1)), 0, h-1)) + 1
            subslc = slice(sy0,sy1),slice(sx0,sx1)
            subimg = tim.getImage   ()[subslc]
            subie  = tim.getInvError()[subslc]
            if tim.dq is None:
                subdq = None
            else:
                subdq  = tim.dq[subslc]
            subwcs = tim.getWcs().shifted(sx0, sy0)
            subsky = tim.getSky().shifted(sx0, sy0)
            subpsf = tim.getPsf().getShifted(sx0, sy0)
            subwcsobj = tim.subwcs.get_subimage(sx0, sy0, sx1-sx0, sy1-sy0)
            tim.imobj.psfnorm = tim.psfnorm
            tim.imobj.galnorm = tim.galnorm
            # FIXME -- maybe the cache is worth sending?
            if hasattr(tim.psf, 'clear_cache'):
                tim.psf.clear_cache()
            # Yuck!  If we not running with --threads AND oneblob.py modifies the data,
            # bad things happen!
            if single_thread:
                subimg = subimg.copy()
                subie = subie.copy()
                subdq = subdq.copy()
            subtimargs.append((subimg, subie, subdq, subwcs, subwcsobj,
                               tim.getPhotoCal(),
                               subsky, subpsf, tim.name, tim.band, tim.sig1, tim.imobj))

        yield (brickname, iblob, 
               (nblob, iblob, Isrcs, targetwcs, bx0, by0, blobw, blobh,
                blobmask, subtimargs, [cat[i] for i in Isrcs], bands, plots, ps,
                reoptimize, iterative, use_ceres, refmap[bslc],
                large_galaxies_force_pointsource, less_masking,
                frozen_galaxies.get(iblob, []), force_model))

def _bounce_one_blob(X):
    ''' This just wraps the one_blob function, for debugging &
    multiprocessing purposes.
    '''
    from legacypipe.oneblob import one_blob
    (brickname, iblob, X) = X
    try:
        result = one_blob(X)
        ### This defines the format of the results in the checkpoints files
        return dict(brickname=brickname, iblob=iblob, result=result)
    except:
        import traceback
        print('Exception in one_blob: brick %s, iblob %i' % (brickname, iblob))
        traceback.print_exc()
        raise

def _get_mod(X):
    from tractor import Tractor
    (tim, srcs) = X
    t0 = Time()
    tractor = Tractor([tim], srcs)
    mod = tractor.getModelImage(0)
    debug('Getting model for', tim, ':', Time()-t0)
    if hasattr(tim.psf, 'clear_cache'):
        tim.psf.clear_cache()
    return mod

def _get_both_mods(X):
    from astrometry.util.resample import resample_with_wcs, OverlapError
    from astrometry.util.miscutils import get_overlapping_region
    (tim, srcs, srcblobs, blobmap, targetwcs, frozen_galaxies, ps, plots) = X
    mod = np.zeros(tim.getModelShape(), np.float32)
    blobmod = np.zeros(tim.getModelShape(), np.float32)
    assert(len(srcs) == len(srcblobs))
    ### modelMasks during fitblobs()....?
    try:
        Yo,Xo,Yi,Xi,_ = resample_with_wcs(tim.subwcs, targetwcs)
    except OverlapError:
        return None,None
    timblobmap = np.empty(mod.shape, blobmap.dtype)
    timblobmap[:,:] = -1
    timblobmap[Yo,Xo] = blobmap[Yi,Xi]
    del Yo,Xo,Yi,Xi

    srcs_blobs = list(zip(srcs, srcblobs))

    fro_rd = set()
    if frozen_galaxies is not None:
        from tractor.patch import ModelMask
        timblobs = set(timblobmap.ravel())
        timblobs.discard(-1)
        h,w = tim.shape
        mm = ModelMask(0, 0, w, h)
        for fro,bb in frozen_galaxies.items():
            # Does this source (which touches blobs bb) touch any blobs in this tim?
            touchedblobs = timblobs.intersection(bb)
            if len(touchedblobs) == 0:
                continue
            patch = fro.getModelPatch(tim, modelMask=mm)
            if patch is None:
                continue
            patch.addTo(mod)

            assert(patch.shape == mod.shape)
            # np.isin doesn't work with a *set* argument!
            blobmask = np.isin(timblobmap, list(touchedblobs))
            blobmod += patch.patch * blobmask

            if plots:
                import pylab as plt
                plt.clf()
                plt.imshow(blobmask, interpolation='nearest', origin='lower', vmin=0, vmax=1,
                           cmap='gray')
                plt.title('tim %s: frozen-galaxy blobmask' % tim.name)
                ps.savefig()
                plt.clf()
                plt.imshow(patch.patch, interpolation='nearest', origin='lower',
                           cmap='gray')
                plt.title('tim %s: frozen-galaxy patch' % tim.name)
                ps.savefig()

            # Drop this frozen galaxy from the catalog to render, if it is present
            # (ie, if it is in_bounds)
            fro_rd.add((fro.pos.ra, fro.pos.dec))

    NEA = []
    no_nea = [0.,0.,0.]
    pcal = tim.getPhotoCal()
    for src,srcblob in srcs_blobs:
        if src is None:
            NEA.append(no_nea)
            continue
        if (src.pos.ra, src.pos.dec) in fro_rd:
            # Skip frozen galaxy source (here we choose not to compute NEA)
            NEA.append(no_nea)
            continue
        patch = src.getModelPatch(tim)
        if patch is None:
            NEA.append(no_nea)
            continue
        # From patch.addTo() -- find pixel overlap region
        (ih, iw) = mod.shape
        (ph, pw) = patch.shape
        (outx, inx) = get_overlapping_region(
            patch.x0, patch.x0 + pw - 1, 0, iw - 1)
        (outy, iny) = get_overlapping_region(
            patch.y0, patch.y0 + ph - 1, 0, ih - 1)
        if inx == [] or iny == []:
            NEA.append(no_nea)
            continue
        # model image patch
        p = patch.patch[iny, inx]
        # add to model image
        mod[outy, outx] += p
        # mask by blob map
        maskedp = p * (timblobmap[outy,outx] == srcblob)
        # add to blob-masked image
        blobmod[outy, outx] += maskedp
        # per-image NEA computations
        # total flux
        flux = pcal.brightnessToCounts(src.brightness)
        # flux in patch
        pflux = np.sum(p)
        # weighting -- fraction of flux that is in the patch
        fracin = pflux / flux
        # nea
        if pflux == 0: # sum(p**2) can only be zero if all(p==0), and then pflux==0
            nea = 0.
        else:
            nea = pflux**2 / np.sum(p**2)
        mpsq = np.sum(maskedp**2)
        if mpsq == 0 or pflux == 0:
            mnea = 0.
        else:
            mnea = flux**2 / mpsq
        NEA.append([nea, mnea, fracin])

    if hasattr(tim.psf, 'clear_cache'):
        tim.psf.clear_cache()
    return mod, blobmod, NEA

def reprocess_wcat(wcat, brickname,blobmap,maskbits):
	return None,None


def stage_coadds(survey=None, bands=None, version_header=None, targetwcs=None,
                 tims=None, ps=None, brickname=None, ccds=None,
                 custom_brick=False,
                 T=None,
                 refstars=None,
                 blobmap=None,
                 cat=None, pixscale=None, plots=False,
                 coadd_bw=False, brick=None, W=None, H=None, lanczos=True,
                 co_sky=None,
                 saturated_pix=None,
                 refmap=None,
                 frozen_galaxies=None,
                 bailout_mask=None,
                 mp=None,
                 record_event=None,
                 **kwargs):
    '''
    After the `stage_fitblobs` fitting stage, we have all the source
    model fits, and we can create coadds of the images, model, and
    residuals.  We also perform aperture photometry in this stage.
    '''
    
    from functools import reduce
    from legacypipe.survey import apertures_arcsec
    from legacypipe.bits import IN_BLOB
    record_event and record_event('stage_coadds: starting')
    _add_stage_version(version_header, 'COAD', 'coadds')
    tlast = Time()
    
    #obiwan
    # Coadd of simulated galaxies
    if hasattr(tims[0], 'sims_image'):
        sims_mods = [tim.sims_image for tim in tims]
        T_sims_coadds = make_coadds(tims, bands, targetwcs, mods=sims_mods,
                lanczos=lanczos, mp=mp,callback=write_coadd_images, callback_args=(survey, brickname, version_header, tims,
                    targetwcs,co_sky))
        sims_coadd = T_sims_coadds.comods
        del T_sims_coadds
        for band in bands:
            sim_coadd_fn= survey.find_file('model',brick=brickname, band=band,
                     output=True)
            os.rename(sim_coadd_fn,sim_coadd_fn.replace('-model-','-sims-'))

    # Write per-brick CCDs table
    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='ccdinfo',
                            comment='NOAO data product type'))
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=primhdr)

    if plots and False:
        import pylab as plt
        from astrometry.util.plotutils import dimshow
        cat_init = [src for it,src in zip(T.iterative, cat) if not(it)]
        cat_iter = [src for it,src in zip(T.iterative, cat) if it]
        print(len(cat_init), 'initial sources and', len(cat_iter), 'iterative')
        mods_init = mp.map(_get_mod, [(tim, cat_init) for tim in tims])
        mods_iter = mp.map(_get_mod, [(tim, cat_iter) for tim in tims])
        coimgs_init,_ = quick_coadds(tims, bands, targetwcs, images=mods_init)
        coimgs_iter,_ = quick_coadds(tims, bands, targetwcs, images=mods_iter)
        coimgs,_ = quick_coadds(tims, bands, targetwcs)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('First-round data')
        ps.savefig()
        plt.clf()
        dimshow(get_rgb(coimgs_init, bands))
        plt.title('First-round model fits')
        ps.savefig()
        plt.clf()
        dimshow(get_rgb([img-mod for img,mod in zip(coimgs,coimgs_init)], bands))
        plt.title('First-round residuals')
        ps.savefig()
        plt.clf()
        dimshow(get_rgb(coimgs_iter, bands))
        plt.title('Iterative model fits')
        ps.savefig()
        plt.clf()
        dimshow(get_rgb([mod+mod2 for mod,mod2 in zip(coimgs_init, coimgs_iter)], bands))
        plt.title('Initial + Iterative model fits')
        ps.savefig()
        plt.clf()
        dimshow(get_rgb([img-mod-mod2 for img,mod,mod2 in zip(coimgs,coimgs_init,coimgs_iter)], bands))
        plt.title('Iterative model residuals')
        ps.savefig()

    # Render model images...
    record_event and record_event('stage_coadds: model images')

    # Re-add the blob that this galaxy is actually inside
    # (that blob got dropped way earlier, before fitblobs)
    if frozen_galaxies is not None:
        for src,bb in frozen_galaxies.items():
            _,xx,yy = targetwcs.radec2pixelxy(src.pos.ra, src.pos.dec)
            xx = int(xx-1)
            yy = int(yy-1)
            bh,bw = blobmap.shape
            if xx >= 0 and xx < bw and yy >= 0 and yy < bh:
                # in bounds!
                debug('Frozen galaxy', src, 'lands in blob', blobmap[yy,xx])
                if blobmap[yy,xx] != -1:
                    bb.append(blobmap[yy,xx])

    Ireg = np.flatnonzero(T.regular)
    Nreg = len(Ireg)
    bothmods = mp.map(_get_both_mods, [(tim, [cat[i] for i in Ireg], T.blob[Ireg], blobmap,
                                        targetwcs, frozen_galaxies, ps, plots)
                                       for tim in tims])
    mods     = [r[0] for r in bothmods]
    blobmods = [r[1] for r in bothmods]
    NEA      = [r[2] for r in bothmods]
    NEA = np.array(NEA)
    # NEA shape (tims, srcs, 3:[nea, blobnea, weight])
    neas        = NEA[:,:,0]
    blobneas    = NEA[:,:,1]
    nea_wts     = NEA[:,:,2]
    del bothmods, NEA
    tnow = Time()
    debug('Model images:', tnow-tlast)
    tlast = tnow

    # source pixel positions to probe depth maps, etc
    ixy = (np.clip(T.ibx, 0, W-1).astype(int), np.clip(T.iby, 0, H-1).astype(int))
    # convert apertures to pixels
    apertures = apertures_arcsec / pixscale
    # Aperture photometry locations
    apxy = np.vstack((T.bx, T.by)).T

    record_event and record_event('stage_coadds: coadds')

    C = make_coadds(tims, bands, targetwcs, mods=mods, blobmods=blobmods,
                    xy=ixy,
                    ngood=True, detmaps=True, psfsize=True, allmasks=True,
                    lanczos=lanczos,
                    apertures=apertures, apxy=apxy,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, tims,
                                   targetwcs, co_sky),
                    plots=plots, ps=ps, mp=mp)
    record_event and record_event('stage_coadds: extras')

    # Coadds of galaxy sims only, image only
    if hasattr(tims[0], 'sims_image'):
        sims_mods = [tim.sims_image for tim in tims]
        T_sims_coadds = make_coadds(tims, bands, targetwcs, mods=sims_mods,
                                    lanczos=lanczos, mp=mp)
        sims_coadd = T_sims_coadds.comods
        del T_sims_coadds
        image_only_mods= [tim.data-tim.sims_image for tim in tims]
        make_coadds(tims, bands, targetwcs, mods=image_only_mods,
                    lanczos=lanczos, mp=mp)
    ###

    # Save per-source measurements of the maps produced during coadding
    cols = ['nobs', 'anymask', 'allmask', 'psfsize', 'psfdepth', 'galdepth',
            'mjd_min', 'mjd_max']
    # store galaxy sim bounding box in Tractor cat
    if 'sims_xy' in C.T.get_columns():
        cols.append('sims_xy')
    for c in cols:
        T.set(c, C.T.get(c))

    # average NEA stats per band -- after psfsize,psfdepth computed.
    # first init all bands expected by format_catalog
    for band in survey.allbands:
        T.set('nea_%s' % band, np.zeros(len(T), np.float32))
        T.set('blob_nea_%s' % band, np.zeros(len(T), np.float32))
    for iband,band in enumerate(bands):
        num  = np.zeros(Nreg, np.float32)
        den  = np.zeros(Nreg, np.float32)
        bnum = np.zeros(Nreg, np.float32)
        for tim,nea,bnea,nea_wt in zip(
                tims, neas, blobneas, nea_wts):
            if not tim.band == band:
                continue
            iv = 1./(tim.sig1**2)
            I, = np.nonzero(nea)
            wt = nea_wt[I]
            num[I] += iv * wt * 1./(nea[I] * tim.imobj.pixscale**2)
            den[I] += iv * wt
            I, = np.nonzero(bnea)
            bnum[I] += iv * 1./bnea[I]
        # bden is the coadded per-pixel inverse variance derived from psfdepth and psfsize
        # this ends up in arcsec units, not pixels
        bden = T.psfdepth[Ireg,iband] * (4 * np.pi * (T.psfsize[Ireg,iband]/2.3548)**2)
        # numerator and denominator are for the inverse-NEA!
        with np.errstate(divide='ignore', invalid='ignore'):
            nea  = den  / num
            bnea = bden / bnum
        nea [np.logical_not(np.isfinite(nea ))] = 0.
        bnea[np.logical_not(np.isfinite(bnea))] = 0.
        # Set vals in T
        T.get('nea_%s' % band)[Ireg] = nea
        T.get('blob_nea_%s' % band)[Ireg] = bnea

    # Grab aperture fluxes
    assert(C.AP is not None)
    # How many apertures?
    A = len(apertures_arcsec)
    for src,dst in [('apflux_img_%s',       'apflux'),
                    ('apflux_img_ivar_%s',  'apflux_ivar'),
                    ('apflux_masked_%s',    'apflux_masked'),
                    ('apflux_resid_%s',     'apflux_resid'),
                    ('apflux_blobresid_%s', 'apflux_blobresid'),]:
        X = np.zeros((len(T), len(bands), A), np.float32)
        for iband,band in enumerate(bands):
            X[:,iband,:] = C.AP.get(src % band)
        T.set(dst, X)

    # Compute depth histogram
    D = _depth_histogram(brick, targetwcs, bands, C.psfdetivs, C.galdetivs)
    with survey.write_output('depth-table', brick=brickname) as out:
        D.writeto(None, fits_object=out.fits)
    del D

    # Create JPEG coadds
    coadd_list= [('image', C.coimgs, {}),
                 ('model', C.comods, {}),
                 ('blobmodel', C.coblobmods, {}),
                 ('resid', C.coresids, dict(resids=True))]
    if hasattr(tims[0], 'sims_image'):
        coadd_list.append(('simscoadd', sims_coadd, {}))

    for name,ims,rgbkw in coadd_list:
        rgb = get_rgb(ims, bands, **rgbkw)
        kwa = {}
        if coadd_bw and len(bands) == 1:
            rgb = rgb.sum(axis=2)
            kwa = dict(cmap='gray')
        with survey.write_output(name + '-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
            info('Wrote', out.fn)
        del rgb

    # Construct the maskbits map
    maskbits = np.zeros((H,W), np.int16)
    # !PRIMARY
    if not custom_brick:
        U = find_unique_pixels(targetwcs, W, H, None,
                               brick.ra1, brick.ra2, brick.dec1, brick.dec2)
        maskbits |= MASKBITS['NPRIMARY'] * np.logical_not(U).astype(np.int16)
        del U

    # BRIGHT
    if refmap is not None:
        maskbits |= MASKBITS['BRIGHT']  * ((refmap & IN_BLOB['BRIGHT'] ) > 0)
        maskbits |= MASKBITS['MEDIUM']  * ((refmap & IN_BLOB['MEDIUM'] ) > 0)
        maskbits |= MASKBITS['GALAXY']  * ((refmap & IN_BLOB['GALAXY'] ) > 0)
        maskbits |= MASKBITS['CLUSTER'] * ((refmap & IN_BLOB['CLUSTER']) > 0)
        del refmap

    # SATUR
    if saturated_pix is not None:
        for b, sat in zip(bands, saturated_pix):
            maskbits |= (MASKBITS['SATUR_' + b.upper()] * sat).astype(np.int16)

    # ALLMASK_{g,r,z}
    for b,allmask in zip(bands, C.allmasks):
        maskbits |= (MASKBITS['ALLMASK_' + b.upper()] * (allmask > 0))

    # BAILOUT_MASK
    if bailout_mask is not None:
        maskbits |= MASKBITS['BAILOUT'] * bailout_mask.astype(bool)

    # Add the maskbits header cards to version_header
    mbits = [
        ('NPRIMARY',  'NPRIM', 'not primary brick area'),
        ('BRIGHT',    'BRIGH', 'bright star nearby'),
        ('SATUR_G',   'SAT_G', 'g band saturated'),
        ('SATUR_R',   'SAT_R', 'r band saturated'),
        ('SATUR_Z',   'SAT_Z', 'z band saturated'),
        ('ALLMASK_G', 'ALL_G', 'any ALLMASK_G bit set'),
        ('ALLMASK_R', 'ALL_R', 'any ALLMASK_R bit set'),
        ('ALLMASK_Z', 'ALL_Z', 'any ALLMASK_Z bit set'),
        ('WISEM1',    'WISE1', 'WISE W1 (all masks)'),
        ('WISEM2',    'WISE2', 'WISE W2 (all masks)'),
        ('BAILOUT',   'BAIL',  'Bailed out processing'),
        ('MEDIUM',    'MED',   'medium-bright star'),
        ('GALAXY',    'GAL',   'SGA large galaxy'),
        ('CLUSTER',   'CLUST', 'Globular cluster')]
    version_header.add_record(dict(name='COMMENT', value='maskbits bits:'))
    _add_bit_description(version_header, MASKBITS, mbits,
                         'MB_%s', 'MBIT_%i', 'maskbits')

    # Add the fitbits header cards to version_header
    fbits = [
        ('FORCED_POINTSOURCE',  'FPSF',  'forced to be PSF'),
        ('FIT_BACKGROUND',      'FITBG', 'background levels fit'),
        ('HIT_RADIUS_LIMIT',    'RLIM',  'hit radius limit during fit'),
        ('HIT_SERSIC_LIMIT',    'SLIM',  'hit Sersic index limit during fit'),
        ('FROZEN',              'FROZE', 'parameters were not fit'),
        ('BRIGHT',              'BRITE', 'bright star'),
        ('MEDIUM',              'MED',   'medium-bright star'),
        ('GAIA',                'GAIA',  'Gaia source'),
        ('TYCHO2',              'TYCHO', 'Tycho-2 star'),
        ('LARGEGALAXY',         'LGAL',  'SGA large galaxy'),
        ('WALKER',              'WALK',  'fitting moved pos > 1 arcsec'),
        ('RUNNER',              'RUN',   'fitting moved pos > 2.5 arcsec'),
        ('GAIA_POINTSOURCE',    'GPSF',  'Gaia source treated as point source'),
        ('ITERATIVE',           'ITER',  'source detected during iterative detection'),
        ]
    version_header.add_record(dict(name='COMMENT', value='fitbits bits:'))
    _add_bit_description(version_header, FITBITS, fbits,
                         'FB_%s', 'FBIT_%i', 'fitbits')

    if plots:
        import pylab as plt
        from astrometry.util.plotutils import dimshow
        plt.clf()
        ra  = np.array([src.getPosition().ra  for src in cat])
        dec = np.array([src.getPosition().dec for src in cat])
        x0,y0 = T.bx0, T.by0
        ok,x1,y1 = targetwcs.radec2pixelxy(ra, dec)
        x1 -= 1.
        y1 -= 1.
        dimshow(get_rgb(C.coimgs, bands))
        ax = plt.axis()
        for xx0,yy0,xx1,yy1 in zip(x0,y0,x1,y1):
            plt.plot([xx0,xx1], [yy0,yy1], 'r-')
        plt.plot(x1, y1, 'r.')
        plt.axis(ax)
        plt.title('Original to final source positions')
        ps.savefig()

        plt.clf()
        dimshow(get_rgb(C.coimgs, bands))
        ax = plt.axis()
        ps.savefig()

        for src,x,y,rr,dd in zip(cat, x1, y1, ra, dec):
            from tractor import PointSource
            from tractor.galaxy import DevGalaxy, ExpGalaxy
            from tractor.sersic import SersicGalaxy
            ee = []
            ec = []
            cc = None
            green = (0.2,1,0.2)
            if isinstance(src, PointSource):
                plt.plot(x, y, 'o', mfc=green, mec='k', alpha=0.6)
            elif isinstance(src, ExpGalaxy):
                ee = [src.shape]
                cc = '0.8'
                ec = [cc]
            elif isinstance(src, DevGalaxy):
                ee = [src.shape]
                cc = green
                ec = [cc]
            elif isinstance(src, SersicGalaxy):
                ee = [src.shape]
                cc = 'm'
                ec = [cc]
            else:
                print('Unknown type:', src)
                continue

            for e,c in zip(ee, ec):
                G = e.getRaDecBasis()
                angle = np.linspace(0, 2.*np.pi, 60)
                xy = np.vstack((np.append([0,0,1], np.sin(angle)),
                                np.append([0,1,0], np.cos(angle)))).T
                rd = np.dot(G, xy.T).T
                r = rr + rd[:,0] * np.cos(np.deg2rad(dd))
                d = dd + rd[:,1]
                ok,xx,yy = targetwcs.radec2pixelxy(r, d)
                xx -= 1.
                yy -= 1.
                x1,x2,x3 = xx[:3]
                y1,y2,y3 = yy[:3]
                plt.plot([x3, x1, x2], [y3, y1, y2], '-', color=c)
                plt.plot(x1, y1, '.', color=cc, ms=3, alpha=0.6)
                xx = xx[3:]
                yy = yy[3:]
                plt.plot(xx, yy, '-', color=c)
        plt.axis(ax)
        ps.savefig()

    tnow = Time()
    debug('Aperture photometry wrap-up:', tnow-tlast)

    return dict(T=T, apertures_pix=apertures,
                apertures_arcsec=apertures_arcsec,
                maskbits=maskbits,
                version_header=version_header)

def _add_bit_description(header, BITS, bits, bnpat, bitpat, bitmapname):
    for key,short,comm in bits:
        header.add_record(
            dict(name=bnpat % short, value=BITS[key],
                 comment='%s: %s' % (bitmapname, comm)))
    revmap = dict([(bit,name) for name,bit in BITS.items()])
    nicemap = dict([(k,c) for k,short,c in bits])
    for bit in range(16):
        bitval = 1<<bit
        if not bitval in revmap:
            continue
        name = revmap[bitval]
        nice = nicemap.get(name, '')
        header.add_record(
            dict(name=bitpat % bit, value=name,
                 comment='%s bit %i (0x%x): %s' % (bitmapname, bit, bitval, nice)))

def get_fiber_fluxes(cat, T, targetwcs, H, W, pixscale, bands,
                     fibersize=1.5, seeing=1., year=2020.0,
                     plots=False, ps=None):
    from tractor import GaussianMixturePSF
    from legacypipe.survey import LegacySurveyWcs
    import astropy.time
    from tractor.tractortime import TAITime
    from tractor.image import Image
    from tractor.basics import LinearPhotoCal
    import photutils

    # Create a fake tim for each band to construct the models in 1" seeing
    # For Gaia stars, we need to give a time for evaluating the models.
    mjd_tai = astropy.time.Time(year, format='jyear').tai.mjd
    tai = TAITime(None, mjd=mjd_tai)
    # 1" FWHM -> pixels FWHM -> pixels sigma -> pixels variance
    v = ((seeing / pixscale) / 2.35)**2
    data = np.zeros((H,W), np.float32)
    inverr = np.ones((H,W), np.float32)
    psf = GaussianMixturePSF(1., 0., 0., v, v, 0.)
    wcs = LegacySurveyWcs(targetwcs, tai)
    faketim = Image(data=data, inverr=inverr, psf=psf,
                    wcs=wcs, photocal=LinearPhotoCal(1., bands[0]))

    # A model image (containing all sources) for each band
    modimgs = [np.zeros((H,W), np.float32) for b in bands]
    # A blank image that we'll use for rendering the flux from a single model
    onemod = data

    # Results go here!
    fiberflux    = np.zeros((len(cat),len(bands)), np.float32)
    fibertotflux = np.zeros((len(cat),len(bands)), np.float32)

    # Fiber diameter in arcsec -> radius in pix
    fiberrad = (fibersize / pixscale) / 2.

    # For each source, compute and measure its model, and accumulate
    for isrc,(src,sx,sy) in enumerate(zip(cat, T.bx, T.by)):
        if src is None:
            continue
        # This works even if bands[0] has zero flux (or no overlapping
        # images)
        ums = src.getUnitFluxModelPatches(faketim)
        assert(len(ums) == 1)
        patch = ums[0]
        if patch is None:
            continue
        br = src.getBrightness()
        for iband,(modimg,band) in enumerate(zip(modimgs,bands)):
            flux = br.getFlux(band)
            flux_iv = T.flux_ivar[isrc, iband]
            if flux <= 0 or flux_iv <= 0:
                continue
            # Accumulate into image containing all models
            patch.addTo(modimg, scale=flux)
            # Add to blank image & photometer
            patch.addTo(onemod, scale=flux)
            aper = photutils.CircularAperture((sx, sy), fiberrad)
            p = photutils.aperture_photometry(onemod, aper)
            f = p.field('aperture_sum')[0]
            if not np.isfinite(f):
                # If the source is off the brick (eg, ref sources), can be NaN
                continue
            fiberflux[isrc,iband] = f
            # Blank out the image again
            x0,x1,y0,y1 = patch.getExtent()
            onemod[y0:y1, x0:x1] = 0.

    # Now photometer the accumulated images
    # Aperture photometry locations
    apxy = np.vstack((T.bx, T.by)).T
    aper = photutils.CircularAperture(apxy, fiberrad)
    for iband,modimg in enumerate(modimgs):
        p = photutils.aperture_photometry(modimg, aper)
        f = p.field('aperture_sum')
        # If the source is off the brick (eg, ref sources), can be NaN
        I = np.isfinite(f)
        if len(I):
            fibertotflux[I, iband] = f[I]

    if plots:
        import pylab as plt
        for modimg,band in zip(modimgs, bands):
            plt.clf()
            plt.imshow(modimg, interpolation='nearest', origin='lower',
                       vmin=0, vmax=0.1, cmap='gray')
            plt.title('Fiberflux model for band %s' % band)
            ps.savefig()

        for iband,band in enumerate(bands):
            plt.clf()
            flux = [src.getBrightness().getFlux(band) for src in cat]
            plt.plot(flux, fiberflux[:,iband], 'b.', label='FiberFlux')
            plt.plot(flux, fibertotflux[:,iband], 'gx', label='FiberTotFlux')
            plt.plot(flux, T.apflux[:,iband, 1], 'r+', label='Apflux(1.5)')
            plt.legend()
            plt.xlabel('Catalog total flux')
            plt.ylabel('Aperture flux')
            plt.title('Fiberflux: %s band' % band)
            plt.xscale('symlog')
            plt.yscale('symlog')
            ps.savefig()

    return fiberflux, fibertotflux

def _depth_histogram(brick, targetwcs, bands, detivs, galdetivs):
    # Compute the brick's unique pixels.
    U = None
    if hasattr(brick, 'ra1'):
        debug('Computing unique brick pixels...')
        H,W = targetwcs.shape
        U = find_unique_pixels(targetwcs, W, H, None,
                               brick.ra1, brick.ra2, brick.dec1, brick.dec2)
        U = np.flatnonzero(U)
        debug(len(U), 'of', W*H, 'pixels are unique to this brick')

    # depth histogram bins
    depthbins = np.arange(20, 25.001, 0.1)
    depthbins[0] = 0.
    depthbins[-1] = 100.
    D = fits_table()
    D.depthlo = depthbins[:-1].astype(np.float32)
    D.depthhi = depthbins[1: ].astype(np.float32)

    for band,detiv,galdetiv in zip(bands,detivs,galdetivs):
        for det,name in [(detiv, 'ptsrc'), (galdetiv, 'gal')]:
            # compute stats for 5-sigma detection
            with np.errstate(divide='ignore'):
                depth = 5. / np.sqrt(det)
            # that's flux in nanomaggies -- convert to mag
            depth = -2.5 * (np.log10(depth) - 9)
            # no coverage -> very bright detection limit
            depth[np.logical_not(np.isfinite(depth))] = 0.
            if U is not None:
                depth = depth.flat[U]
            if len(depth):
                debug(band, name, 'band depth map: percentiles',
                      np.percentile(depth, np.arange(0,101, 10)))
            # histogram
            D.set('counts_%s_%s' % (name, band),
                  np.histogram(depth, bins=depthbins)[0].astype(np.int32))
    return D

def stage_wise_forced(
    survey=None,
    cat=None,
    T=None,
    targetwcs=None,
    targetrd=None,
    W=None, H=None,
    pixscale=None,
    brickname=None,
    unwise_dir=None,
    unwise_tr_dir=None,
    unwise_modelsky_dir=None,
    brick=None,
    wise_ceres=True,
    unwise_coadds=True,
    version_header=None,
    maskbits=None,
    mp=None,
    record_event=None,
    ps=None,
    plots=False,
    **kwargs):
    '''
    After the model fits are finished, we can perform forced
    photometry of the unWISE coadds.
    '''
    
    from legacypipe.unwise import unwise_phot, collapse_unwise_bitmask, unwise_tiles_touching_wcs
    from legacypipe.survey import wise_apertures_arcsec
    from tractor import NanoMaggies
    record_event and record_event('stage_wise_forced: starting')
    _add_stage_version(version_header, 'WISE', 'wise_forced')

    if not plots:
        ps = None

    tiles = unwise_tiles_touching_wcs(targetwcs)
    info('Cut to', len(tiles), 'unWISE tiles')

    # the way the roiradec box is used, the min/max order doesn't matter
    roiradec = [targetrd[0,0], targetrd[2,0], targetrd[0,1], targetrd[2,1]]

    # Sources to photometer
    do_phot = T.regular.copy()

    # Drop sources within the CLUSTER mask from forced photometry.
    Icluster = None
    if maskbits is not None:
        incluster = (maskbits & MASKBITS['CLUSTER'] > 0)
        if np.any(incluster):
            print('Checking for sources inside CLUSTER mask')
            ra  = np.array([src.getPosition().ra  for src in cat])
            dec = np.array([src.getPosition().dec for src in cat])
            ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)
            xx = np.round(xx - 1).astype(int)
            yy = np.round(yy - 1).astype(int)
            I = np.flatnonzero(ok * (xx >= 0)*(xx < W) * (yy >= 0)*(yy < H))
            if len(I):
                Icluster = I[incluster[yy[I], xx[I]]]
                print('Found', len(Icluster), 'of', len(cat), 'sources inside CLUSTER mask')
                do_phot[Icluster] = False
    Nskipped = len(T) - np.sum(do_phot)

    wcat = []
    for i in np.flatnonzero(do_phot):
        src = cat[i]
        src = src.copy()
        src.setBrightness(NanoMaggies(w=1.))
        wcat.append(src)
    #obiwanTODO
    wcat,Nmorewcat = reprocess_wcat(wcat, brickname,kwargs['blobmap'],maskbits)

    #wcat[0].getPosition().dec wcat[0].getPosition().ra 
    # use Aaron's WISE pixelized PSF model (unwise_psf repository)?
    wpixpsf = True

    # Create list of groups-of-tiles to photometer
    args = []
    # Skip if $UNWISE_COADDS_DIR or --unwise-dir not set.
    if unwise_dir is not None:
        wtiles = tiles.copy()
        wtiles.unwise_dir = np.array([unwise_dir]*len(tiles))
        for band in [1,2,3,4]:
            get_masks = targetwcs if (band == 1) else None
            args.append((wcat, wtiles, band, roiradec, wise_ceres, wpixpsf,
                         unwise_coadds, get_masks, ps, True,
                         unwise_modelsky_dir))

    # Add time-resolved WISE coadds
    # Skip if $UNWISE_COADDS_TIMERESOLVED_DIR or --unwise-tr-dir not set.
    eargs = []
    if unwise_tr_dir is not None:
        tdir = unwise_tr_dir
        TR = fits_table(os.path.join(tdir, 'time_resolved_atlas.fits'))
        debug('Read', len(TR), 'time-resolved WISE coadd tiles')
        TR.cut(np.array([t in tiles.coadd_id for t in TR.coadd_id]))
        debug('Cut to', len(TR), 'time-resolved vs', len(tiles), 'full-depth')
        assert(len(TR) == len(tiles))
        # Ugly -- we need to look up the "{ra,dec}[12]" fields from the non-TR
        # table to support unique areas of tiles.
        imap = dict((c,i) for i,c in enumerate(tiles.coadd_id))
        I = np.array([imap[c] for c in TR.coadd_id])
        for c in ['ra1','ra2','dec1','dec2', 'crpix_w1', 'crpix_w2']:
            TR.set(c, tiles.get(c)[I])
        # How big do we need to make the WISE time-resolved arrays?
        debug('TR epoch_bitmask:', TR.epoch_bitmask)
        # axis= arg to np.count_nonzero is new in numpy 1.12
        Nepochs = max(np.atleast_1d([np.count_nonzero(e)
                                     for e in TR.epoch_bitmask]))
        _,ne = TR.epoch_bitmask.shape
        info('Max number of time-resolved unWISE epochs for these tiles:', Nepochs)
        debug('epoch bitmask length:', ne)
        
        # Add time-resolved coadds
        for band in [1,2]:
            # W1 is bit 0 (value 0x1), W2 is bit 1 (value 0x2)
            bitmask = (1 << (band-1))
            # The epoch_bitmask entries are not *necessarily*
            # contiguous, and not necessarily aligned for the set of
            # overlapping tiles.  We will align the non-zero epochs of
            # the tiles.  (eg, brick 2437p425 vs coadds 2426p424 &
            # 2447p424 in NEO-2).

            # find the non-zero epochs for each overlapping tile
            epochs = np.empty((len(TR), Nepochs), int)
            epochs[:,:] = -1
            for i in range(len(TR)):
                ei = np.flatnonzero(TR.epoch_bitmask[i,:] & bitmask)
                epochs[i,:len(ei)] = ei

            for ie in range(Nepochs):
                # Which tiles have images for this epoch?
                I = np.flatnonzero(epochs[:,ie] >= 0)
                if len(I) == 0:
                    continue
                debug('Epoch index %i: %i tiles:' % (ie, len(I)), TR.coadd_id[I],
                      'epoch numbers', epochs[I,ie])
                eptiles = TR[I]
                eptiles.unwise_dir = np.array([os.path.join(tdir, 'e%03i'%ep)
                                              for ep in epochs[I,ie]])
                eargs.append((ie,(wcat, eptiles, band, roiradec,
                                  wise_ceres, wpixpsf, False, None, ps, False, unwise_modelsky_dir)))
    
    # Run the forced photometry!
    record_event and record_event('stage_wise_forced: photometry')
    phots = mp.map(unwise_phot, args + [a for ie,a in eargs])
    record_event and record_event('stage_wise_forced: results')
    #obiwan
    
    #phots = phots[:-Nmorewcat]    
    # Unpack results...
    WISE = None
    wise_mask_maps = None
    if len(phots):
        # The "phot" results for the full-depth coadds are one table per
        # band.  Merge all those columns.
        wise_models = []
        sim_wise_models = []
        for i,p in enumerate(phots[:len(args)]):
            if p is None:
                (wcat,tiles,band) = args[i][:3]
                print('"None" result from WISE forced phot:', tiles, band)
                continue
            if unwise_coadds:
                wise_models.extend(p.models)
            #obiwan
            sim_unwise_coadds = True
            if sim_unwise_coadds:
                sim_wise_models.extend(p.sim_models)
            if p.maskmap is not None:
                wise_mask_maps = p.maskmap
            if WISE is None:
                if Nmorewcat>0:
                    WISE = p.phot[:-Nmorewcat]
                else:
                    WISE = p.phot
            else:
                # remove duplicates
                p.phot.delete_column('wise_coadd_id')
                # (with move_crpix -- Aaron's update astrometry -- the
                # pixel positions can be *slightly* different per
                # band.  Ignoring that here.)
                p.phot.delete_column('wise_x')
                p.phot.delete_column('wise_y')
                if Nmorewcat>0:
                    WISE.add_columns_from(p.phot[:-Nmorewcat])
                else:
                    WISE.add_columns_from(p.phot)

        if wise_mask_maps is not None:
            wise_mask_maps = [
                collapse_unwise_bitmask(wise_mask_maps, 1),
                collapse_unwise_bitmask(wise_mask_maps, 2)]

        if Nskipped > 0:
            assert(len(WISE) == len(wcat[:-Nmorewcat]))
            WISE = _fill_skipped_values(WISE, Nskipped, do_phot)
            assert(len(WISE) == len(cat))
            assert(len(WISE) == len(T))

        if unwise_coadds:
            from legacypipe.coadds import UnwiseCoadd
            # Create the WCS into which we'll resample the tiles.
            # Same center as "targetwcs" but bigger pixel scale.
            wpixscale = 2.75
            rc,dc = targetwcs.radec_center()
            ww = int(W * pixscale / wpixscale)
            hh = int(H * pixscale / wpixscale)
            wcoadds = UnwiseCoadd(rc, dc, ww, hh, wpixscale)
            wcoadds.add(wise_models, unique=True)
            apphot = wcoadds.finish(survey, brickname, version_header,
                                    apradec=(T.ra,T.dec),
                                    apertures=wise_apertures_arcsec/wpixscale)
            #obiwan
            if sim_unwise_coadds:
                from legacypipe.coadds import SimUnwiseCoadd
                simwcoadds = SimUnwiseCoadd(rc, dc, ww, hh, wpixscale)
                simwcoadds.add(sim_wise_models, unique=True)
                apphot = simwcoadds.finish(survey, brickname, version_header,
                                    apradec=(T.ra,T.dec),
                                    apertures=wise_apertures_arcsec/wpixscale)
            api,apd,apr = apphot
            for iband,band in enumerate([1,2,3,4]):
                WISE.set('apflux_w%i' % band, api[iband])
                WISE.set('apflux_resid_w%i' % band, apr[iband])
                d = apd[iband]
                iv = np.zeros_like(d)
                iv[d != 0.] = 1./(d[d != 0]**2)
                WISE.set('apflux_ivar_w%i' % band, iv)
                print('Setting WISE apphot')

        # Look up mask values for sources
        WISE.wise_mask = np.zeros((len(cat), 2), np.uint8)
        
        WISE.wise_mask[T.in_bounds,0] = wise_mask_maps[0][T.iby[T.in_bounds], T.ibx[T.in_bounds]]
        WISE.wise_mask[T.in_bounds,1] = wise_mask_maps[1][T.iby[T.in_bounds], T.ibx[T.in_bounds]]


    # Unpack time-resolved results...
    WISE_T = None
    if len(phots) > len(args):
        WISE_T = True
    if WISE_T is not None:
        WISE_T = fits_table()
        phots = phots[len(args):]
        for (ie,_),r in zip(eargs, phots):
            debug('Epoch', ie, 'photometry:')
            if r is None:
                debug('Failed.')
                continue
            assert(ie < Nepochs)
            phot = r.phot[:-Nmorewcat]
            phot.delete_column('wise_coadd_id')
            phot.delete_column('wise_x')
            phot.delete_column('wise_y')
            for c in phot.columns():
                if not c in WISE_T.columns():
                    x = phot.get(c)
                    WISE_T.set(c, np.zeros((len(x), Nepochs), x.dtype))
                X = WISE_T.get(c)
                X[:,ie] = phot.get(c)
        if Nskipped > 0:
            assert(len(wcat[:-Nmorewcat]) == len(WISE_T))
            WISE_T = _fill_skipped_values(WISE_T, Nskipped, do_phot)
            assert(len(WISE_T) == len(cat))
            assert(len(WISE_T) == len(T))

    debug('Returning: WISE', WISE)
    debug('Returning: WISE_T', WISE_T)

    return dict(WISE=WISE, WISE_T=WISE_T, wise_mask_maps=wise_mask_maps,
                version_header=version_header,
                wise_apertures_arcsec=wise_apertures_arcsec)


def _fill_skipped_values(WISE, Nskipped, do_phot):
    # Fill in blank values for skipped (Icluster) sources
    # Append empty rows to the WISE results for !do_phot sources.
    Wempty = fits_table()
    Wempty.nil = np.zeros(Nskipped, bool)
    WISE = merge_tables([WISE, Wempty], columns='fillzero')
    WISE.delete_column('nil')
    # Reorder to match "cat" order.
    I = np.empty(len(WISE), int)
    I[:] = -1
    Ido, = np.nonzero(do_phot)
    I[Ido] = np.arange(len(Ido))
    Idont, = np.nonzero(np.logical_not(do_phot))
    I[Idont] = np.arange(len(Idont)) + len(Ido)
    assert(np.all(I > -1))
    WISE.cut(I)
    return WISE

def stage_writecat(
    survey=None,
    version_header=None,
    release=None,
    T=None,
    WISE=None,
    WISE_T=None,
    maskbits=None,
    wise_mask_maps=None,
    apertures_arcsec=None,
    wise_apertures_arcsec=None,
    GALEX=None,
    galex_apertures_arcsec=None,
    cat=None, pixscale=None, targetwcs=None,
    W=None,H=None,
    bands=None, ps=None,
    plots=False,
    brickname=None,
    brickid=None,
    brick=None,
    invvars=None,
    gaia_stars=True,
    co_sky=None,
    record_event=None,
    **kwargs):

    '''
    Final stage in the pipeline: format results for the output
    catalog.
    '''
    from legacypipe.catalog import prepare_fits_catalog
    from legacypipe.utils import copy_header_with_wcs

    record_event and record_event('stage_writecat: starting')
    _add_stage_version(version_header, 'WCAT', 'writecat')

    assert(maskbits is not None)

    if wise_mask_maps is not None:
        # Add the WISE masks in!
        maskbits |= MASKBITS['WISEM1'] * (wise_mask_maps[0] != 0)
        maskbits |= MASKBITS['WISEM2'] * (wise_mask_maps[1] != 0)

    version_header.add_record(dict(name='COMMENT', value='wisemask bits:'))
    wbits = [
        (0, 'BRIGHT',  'BRIGH', 'Bright star core/wings'),
        (1, 'SPIKE',   'SPIKE', 'PSF-based diffraction spike'),
        (2, 'GHOST',   'GHOST', 'Optical ghost'),
        (3, 'LATENT',  'LATNT', 'First latent'),
        (4, 'LATENT2', 'LATN2', 'Second latent image'),
        (5, 'HALO',    'HALO',  'AllWISE-like circular halo'),
        (6, 'SATUR',   'SATUR', 'Bright star saturation'),
        (7, 'SPIKE2',  'SPIK2', 'Geometric diffraction spike')]
    for bit,name,short,comm in wbits:
        version_header.add_record(dict(
            name='WB_%s' % short, value=1<<bit,
            comment='WISE mask bit %i: %s, %s' % (bit, name, comm)))
    for bit,name,_,comm in wbits:
        version_header.add_record(dict(
            name='WBIT_%i' % bit, value=name, comment='WISE: %s' % comm))

    # Record the meaning of ALLMASK/ANYMASK bits
    version_header.add_record(dict(name='COMMENT', value='allmask/anymask bits:'))
    bits = list(DQ_BITS.values())
    bits.sort()
    bitmap = dict((v,k) for k,v in DQ_BITS.items())
    for i in range(16):
        bit = 1<<i
        if not bit in bitmap:
            continue
        version_header.add_record(
            dict(name='AM_%s' % bitmap[bit].upper()[:5], value=bit,
                 comment='ALLMASK/ANYMASK bit 2**%i' % i))
    for i in range(16):
        bit = 1<<i
        if not bit in bitmap:
            continue
        version_header.add_record(
            dict(name='ABIT_%i' % i, value=bitmap[bit],
                 comment='ALLMASK/ANYMASK bit 2**%i=%i meaning' % (i, bit)))

    # create maskbits header
    hdr = copy_header_with_wcs(version_header, targetwcs)
    hdr.add_record(dict(name='IMTYPE', value='maskbits',
                        comment='LegacySurveys image type'))
    with survey.write_output('maskbits', brick=brickname, shape=maskbits.shape) as out:
        out.fits.write(maskbits, header=hdr, extname='MASKBITS')
        if wise_mask_maps is not None:
            out.fits.write(wise_mask_maps[0], extname='WISEM1')
            out.fits.write(wise_mask_maps[1], extname='WISEM2')
        del wise_mask_maps

    T_orig = T.copy()

    T = prepare_fits_catalog(cat, invvars, T, bands, force_keep=T.force_keep_source)
    # Override type for DUP objects
    T.type[T.dup] = 'DUP'

    # The "ra_ivar" values coming out of the tractor fits do *not*
    # have a cos(Dec) term -- ie, they give the inverse-variance on
    # the numerical value of RA -- so we want to make the ra_sigma
    # values smaller by multiplying by cos(Dec); so invvars are /=
    # cosdec^2
    T.ra_ivar /= np.cos(np.deg2rad(T.dec))**2

    # Compute fiber fluxes
    T.fiberflux, T.fibertotflux = get_fiber_fluxes(
        cat, T, targetwcs, H, W, pixscale, bands, plots=plots, ps=ps)

    # For reference *stars* only, plug in the reference-catalog inverse-variances.
    if 'ref_cat' in T.get_columns() and 'ra_ivar' in T_orig.get_columns():
        I = np.isin(T.ref_cat, ['G2', 'T2'])
        if len(I):
            T.ra_ivar [I] = T_orig.ra_ivar [I]
            T.dec_ivar[I] = T_orig.dec_ivar[I]

    # In oneblob.py we have a step where we zero out the fluxes for sources
    # with tiny "fracin" values.  Repeat that here, but zero out more stuff...
    for iband,band in enumerate(bands):
        # we could do this on the 2d arrays...
        I = np.flatnonzero(T.fracin[:,iband] < 1e-3)
        debug('Zeroing out', len(I), 'objs in', band, 'band with small fracin.')
        if len(I):
            # zero out:
            T.flux[I,iband] = 0.
            T.flux_ivar[I,iband] = 0.
            # zero out fracin itself??

    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='catalog',
                            comment='NOAO data product type'))

    if co_sky is not None:
        for band in bands:
            if band in co_sky:
                primhdr.add_record(dict(name='COSKY_%s' % band.upper(),
                                        value=co_sky[band],
                                        comment='Sky level estimated (+subtracted) from coadd'))

    for i,ap in enumerate(apertures_arcsec):
        primhdr.add_record(dict(name='APRAD%i' % i, value=ap,
                                comment='(optical) Aperture radius, in arcsec'))
    if wise_apertures_arcsec is not None:
        for i,ap in enumerate(wise_apertures_arcsec):
            primhdr.add_record(dict(name='WAPRAD%i' % i, value=ap,
                                    comment='(unWISE) Aperture radius, in arcsec'))
    if galex_apertures_arcsec is not None:
        for i,ap in enumerate(galex_apertures_arcsec):
            primhdr.add_record(dict(name='GAPRAD%i' % i, value=ap,
                                    comment='GALEX aperture radius, in arcsec'))

    if WISE is not None:
        # Convert WISE fluxes from Vega to AB.
        # http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2ab
        vega_to_ab = dict(w1=2.699,
                          w2=3.339,
                          w3=5.174,
                          w4=6.620)

        for band in [1,2,3,4]:
            primhdr.add_record(dict(
                name='WISEAB%i' % band, value=vega_to_ab['w%i' % band],
                comment='WISE Vega to AB conv for band %i' % band))

        # Copy columns:
        for c in ['wise_coadd_id', 'wise_x', 'wise_y', 'wise_mask']:
            T.set(c, WISE.get(c))

        for band in [1,2,3,4]:
            # Apply the Vega-to-AB shift *while* copying columns from
            # WISE to T.
            dm = vega_to_ab['w%i' % band]
            fluxfactor = 10.** (dm / -2.5)
            # fluxes
            c = t = 'flux_w%i' % band
            T.set(t, WISE.get(c) * fluxfactor)
            if WISE_T is not None and band <= 2:
                t = 'lc_flux_w%i' % band
                T.set(t, WISE_T.get(c) * fluxfactor)
            # ivars
            c = t = 'flux_ivar_w%i' % band
            T.set(t, WISE.get(c) / fluxfactor**2)
            if WISE_T is not None and band <= 2:
                t = 'lc_flux_ivar_w%i' % band
                T.set(t, WISE_T.get(c) / fluxfactor**2)
            # This is in 1/nanomaggies**2 units also
            c = t = 'psfdepth_w%i' % band
            T.set(t, WISE.get(c) / fluxfactor**2)

            if 'apflux_w%i'%band in WISE.get_columns():
                t = c = 'apflux_w%i' % band
                T.set(t, WISE.get(c) * fluxfactor)
                t = c = 'apflux_resid_w%i' % band
                T.set(t, WISE.get(c) * fluxfactor)
                t = c = 'apflux_ivar_w%i' % band
                T.set(t, WISE.get(c) / fluxfactor**2)

        # Rename more columns
        for cin,cout in [('nobs_w%i',        'nobs_w%i'    ),
                         ('profracflux_w%i', 'fracflux_w%i'),
                         ('prochi2_w%i',     'rchisq_w%i'  )]:
            for band in [1,2,3,4]:
                T.set(cout % band, WISE.get(cin % band))

        if WISE_T is not None:
            for cin,cout in [('nobs_w%i',        'lc_nobs_w%i'),
                             ('profracflux_w%i', 'lc_fracflux_w%i'),
                             ('prochi2_w%i',     'lc_rchisq_w%i'),
                             ('mjd_w%i',         'lc_mjd_w%i'),]:
                for band in [1,2]:
                    T.set(cout % band, WISE_T.get(cin % band))
        # Done with these now!
        WISE_T = None
        WISE = None

    if GALEX is not None:
        for c in ['flux_nuv', 'flux_ivar_nuv', 'flux_fuv', 'flux_ivar_fuv',
                  'apflux_nuv', 'apflux_resid_nuv', 'apflux_ivar_nuv',
                  'apflux_fuv', 'apflux_resid_fuv', 'apflux_ivar_fuv', ]:
            T.set(c, GALEX.get(c))
        GALEX = None

    T.brick_primary = ((T.ra  >= brick.ra1 ) * (T.ra  < brick.ra2) *
                        (T.dec >= brick.dec1) * (T.dec < brick.dec2))
    H,W = maskbits.shape
    T.maskbits = maskbits[np.clip(T.iby, 0, H-1).astype(int),
                          np.clip(T.ibx, 0, W-1).astype(int)]
    del maskbits

    # Set Sersic indices for all galaxy types.
    # sigh, bytes vs strings.  In py3, T.type (dtype '|S3') are bytes.
    T.sersic[np.array([t in ['DEV',b'DEV'] for t in T.type])] = 4.0
    T.sersic[np.array([t in ['EXP',b'EXP'] for t in T.type])] = 1.0
    T.sersic[np.array([t in ['REX',b'REX'] for t in T.type])] = 1.0

    T.fitbits = np.zeros(len(T), np.int16)
    T.fitbits[T.forced_pointsource] |= FITBITS['FORCED_POINTSOURCE']
    T.fitbits[T.fit_background]     |= FITBITS['FIT_BACKGROUND']
    T.fitbits[T.hit_r_limit]        |= FITBITS['HIT_RADIUS_LIMIT']
    T.fitbits[T.hit_ser_limit]      |= FITBITS['HIT_SERSIC_LIMIT']
    # WALKER/RUNNER
    moved = np.hypot(T.bx - T.bx0, T.by - T.by0)
    # radii in pixels:
    walk_radius = 1.  / pixscale
    run_radius  = 2.5 / pixscale
    T.fitbits[moved > walk_radius] |= FITBITS['WALKER']
    T.fitbits[moved > run_radius ] |= FITBITS['RUNNER']
    # do we have Gaia?
    if 'pointsource' in T.get_columns():
        T.fitbits[T.pointsource]       |= FITBITS['GAIA_POINTSOURCE']
    T.fitbits[T.iterative]         |= FITBITS['ITERATIVE']

    for col,bit in [('freezeparams',  'FROZEN'),
                    ('isbright',      'BRIGHT'),
                    ('ismedium',      'MEDIUM'),
                    ('isgaia',        'GAIA'),
                    ('istycho',       'TYCHO2'),
                    ('islargegalaxy', 'LARGEGALAXY')]:
        if not col in T.get_columns():
            continue
        T.fitbits[T.get(col)] |= FITBITS[bit]

    with survey.write_output('tractor-intermediate', brick=brickname) as out:
        T[np.argsort(T.objid)].writeto(None, fits_object=out.fits, primheader=primhdr)

    # After writing tractor-i file, drop (reference) sources outside the brick.
    T.cut(T.in_bounds)

    # The "format_catalog" code expects all lower-case column names...
    for c in T.columns():
        if c != c.lower():
            T.rename(c, c.lower())
    from legacypipe.format_catalog import format_catalog
    with survey.write_output('tractor', brick=brickname) as out:
        format_catalog(T[np.argsort(T.objid)], None, primhdr, bands,
                       survey.allbands, None, release,
                       write_kwargs=dict(fits_object=out.fits),
                       N_wise_epochs=15, motions=gaia_stars, gaia_tagalong=True)

    # write fits file with galaxy-sim stuff (xy bounds of each sim)
    if 'sims_xy' in T.get_columns():
        sims_data = fits_table()
        sims_data.sims_xy = T.sims_xy
        with survey.write_output('galaxy-sims', brick=brickname) as out:
            sims_data.writeto(None, fits_object=out.fits)

    # produce per-brick checksum file.
    with survey.write_output('checksums', brick=brickname, hashsum=False) as out:
        f = open(out.fn, 'w')
        # Write our pre-computed hashcodes.
        for fn,hashsum in survey.output_file_hashes.items():
            f.write('%s *%s\n' % (hashsum, fn))
        f.close()

    record_event and record_event('stage_writecat: done')
    return dict(T=T, version_header=version_header)

def stage_checksum(
        survey=None,
        brickname=None,
        **kwargs):
    '''
    For debugging / special-case processing, write out the current checksums file.
    '''
    # produce per-brick checksum file.
    with survey.write_output('checksums', brick=brickname, hashsum=False) as out:
        f = open(out.fn, 'w')
        # Write our pre-computed hashcodes.
        for fn,hashsum in survey.output_file_hashes.items():
            f.write('%s *%s\n' % (hashsum, fn))
        f.close()

def run_brick(brick, survey, radec=None, pixscale=0.262,
              width=3600, height=3600,
              survey_blob_mask=None,
              release=None,
              zoom=None,
              bands=None,
              allbands='grz',
              nblobs=None, blob=None, blobxy=None, blobradec=None, blobid=None,
              max_blobsize=None,
              nsigma=6,
              saddle_fraction=0.1,
              saddle_min=2.,
              subsky_radii=None,
              reoptimize=False,
              iterative=False,
              wise=True,
              outliers=True,
              cache_outliers=False,
              lanczos=True,
              early_coadds=False,
              blob_image=False,
              do_calibs=True,
              old_calibs_ok=False,
              write_metrics=True,
              gaussPsf=False,
              pixPsf=False,
              hybridPsf=False,
              normalizePsf=False,
              apodize=False,
              splinesky=True,
              subsky=True,
              ubercal_sky=False,
              constant_invvar=False,
              tycho_stars=True,
              gaia_stars=True,
              large_galaxies=True,
              large_galaxies_force_pointsource=True,
              fitoncoadds_reweight_ivar=True,
              less_masking=False,
              fit_on_coadds=True,
              min_mjd=None, max_mjd=None,
              unwise_coadds=True,
              bail_out=False,
              ceres=True,
              wise_ceres=True,
              unwise_dir=None,
              unwise_tr_dir=None,
              unwise_modelsky_dir=None,
              galex=False,
              galex_dir=None,
              threads=None,
              plots=False, plots2=False, coadd_bw=False,
              plot_base=None, plot_number=0,
              command_line=None,
              read_parallel=True,
              record_event=None,
    # These are for the 'stages' infrastructure
              pickle_pat='pickles/runbrick-%(brick)s-%%(stage)s.pickle',
              stages=None,
              force=None, forceall=False, write_pickles=True,
              checkpoint_filename=None,
              checkpoint_period=None,
              prereqs_update=None,
              stagefunc = None,
              ):
    '''Run the full Legacy Survey data reduction pipeline.

    The pipeline is built out of "stages" that run in sequence.  By
    default, this function will cache the result of each stage in a
    (large) pickle file.  If you re-run, it will read from the
    prerequisite pickle file rather than re-running the prerequisite
    stage.  This can yield faster debugging times, but you almost
    certainly want to turn it off (with `writePickles=False,
    forceall=True`) in production.

    Parameters
    ----------
    brick : string
        Brick name such as '2090m065'.  Can be None if *radec* is given.
    survey : a "LegacySurveyData" object (see common.LegacySurveyData), which is in
        charge of the list of bricks and CCDs to be handled, and where output files
        should be written.
    radec : tuple of floats (ra,dec)
        RA,Dec center of the custom region to run.
    pixscale : float
        Brick pixel scale, in arcsec/pixel.  Default = 0.262
    width, height : integers
        Brick size in pixels.  Default of 3600 pixels (with the default pixel
        scale of 0.262) leads to a slight overlap between bricks.
    zoom : list of four integers
        Pixel coordinates [xlo,xhi, ylo,yhi] of the brick subimage to run.
    bands : string
        Filter (band) names to include; default is "grz".

    Notes
    -----
    You must specify the region of sky to work on, via one of:

    - *brick*: string, brick name such as '2090m065'
    - *radec*: tuple of floats; RA,Dec center of the custom region to run

    If *radec* is given, *brick* should be *None*.  If *brick* is given,
    that brick`s RA,Dec center will be looked up in the
    survey-bricks.fits file.

    You can also change the size of the region to reduce:

    - *pixscale*: float, brick pixel scale, in arcsec/pixel.
    - *width* and *height*: integers; brick size in pixels.  3600 pixels
      (with the default pixel scale of 0.262) leads to a slight overlap
      between bricks.
    - *zoom*: list of four integers, [xlo,xhi, ylo,yhi] of the brick
      subimage to run.

    If you want to measure only a subset of the astronomical objects,
    you can use:

    - *nblobs*: None or int; for debugging purposes, only fit the
       first N blobs.
    - *blob*: int; for debugging purposes, start with this blob index.
    - *blobxy*: list of (x,y) integer tuples; only run the blobs
      containing these pixels.
    - *blobradec*: list of (RA,Dec) tuples; only run the blobs
      containing these coordinates.

    Other options:

    - *max_blobsize*: int; ignore blobs with more than this many pixels

    - *nsigma*: float; detection threshold in sigmas.

    - *wise*: boolean; run WISE forced photometry?

    - *early_coadds*: boolean; generate the early coadds?

    - *do_calibs*: boolean; run the calibration preprocessing steps?

    - *old_calibs_ok*: boolean; allow/use old calibration frames?

    - *write_metrics*: boolean; write out a variety of useful metrics

    - *gaussPsf*: boolean; use a simpler single-component Gaussian PSF model?

    - *pixPsf*: boolean; use the pixelized PsfEx PSF model and FFT convolution?

    - *hybridPsf*: boolean; use combo pixelized PsfEx + Gaussian approx model

    - *normalizePsf*: boolean; make PsfEx model have unit flux

    - *splinesky*: boolean; use the splined sky model (default is constant)?

    - *subsky*: boolean; subtract the sky model when reading in tims (tractor images)?

    - *ceres*: boolean; use Ceres Solver when possible?

    - *wise_ceres*: boolean; use Ceres Solver for unWISE forced photometry?

    - *unwise_dir*: string; where to look for unWISE coadd files.
      This may be a colon-separated list of directories to search in
      order.

    - *unwise_tr_dir*: string; where to look for time-resolved
      unWISE coadd files.  This may be a colon-separated list of
      directories to search in order.

    - *unwise_modelsky_dir*: string; where to look for the unWISE sky background
      maps.  The default is to look in the "wise/modelsky" subdirectory of the
      calibration directory.

    - *threads*: integer; how many CPU cores to use

    Plotting options:

    - *coadd_bw*: boolean: if only one band is available, make B&W coadds?
    - *plots*: boolean; make a bunch of plots?
    - *plots2*: boolean; make a bunch more plots?
    - *plot_base*: string, default brick-BRICK, the plot filename prefix.
    - *plot_number*: integer, default 0, starting number for plot filenames.

    Options regarding the "stages":

    - *pickle_pat*: string; filename for 'pickle' files
    - *stages*: list of strings; stages (functions stage_*) to run.

    - *force*: list of strings; prerequisite stages that will be run
      even if pickle files exist.
    - *forceall*: boolean; run all stages, ignoring all pickle files.
    - *write_pickles*: boolean; write pickle files after each stage?

    Raises
    ------
    RunbrickError
        If an invalid brick name is given.
    NothingToDoError
        If no CCDs, or no photometric CCDs, overlap the given brick or region.

    '''
    from astrometry.util.stages import CallGlobalTime, runstage
    from astrometry.util.multiproc import multiproc
    from astrometry.util.plotutils import PlotSequence

    # *initargs* are passed to the first stage (stage_tims)
    # so should be quantities that shouldn't get updated from their pickled
    # values.
    fit_on_coadds=True
    initargs = {}
    # *kwargs* update the pickled values from previous stages
    kwargs = {}

    if force is None:
        force = []
    if stages is None:
        stages=['writecat']
    forceStages = [s for s in stages]
    forceStages.extend(force)
    if forceall:
        kwargs.update(forceall=True)

    if allbands is not None:
        survey.allbands = allbands

    if radec is not None:
        assert(len(radec) == 2)
        ra,dec = radec
        try:
            ra = float(ra)
        except:
            from astrometry.util.starutil_numpy import hmsstring2ra
            ra = hmsstring2ra(ra)
        try:
            dec = float(dec)
        except:
            from astrometry.util.starutil_numpy import dmsstring2dec
            dec = dmsstring2dec(dec)
        info('Parsed RA,Dec', ra,dec)
        initargs.update(ra=ra, dec=dec)
        if brick is None:
            brick = ('custom-%06i%s%05i' %
                         (int(1000*ra), 'm' if dec < 0 else 'p',
                          int(1000*np.abs(dec))))
    initargs.update(brickname=brick, survey=survey)

    if stagefunc is None:
        stagefunc = CallGlobalTime('stage_%s', globals())

    plot_base_default = 'brick-%(brick)s'
    if plot_base is None:
        plot_base = plot_base_default
    ps = PlotSequence(plot_base % dict(brick=brick))
    initargs.update(ps=ps)
    if plot_number:
        ps.skipto(plot_number)

    if release is None:
        release = survey.get_default_release()
        if release is None:
            release = 9999

    if fit_on_coadds:
        # Implied options!
        #subsky = False
        large_galaxies = True
        fitoncoadds_reweight_ivar = True
        large_galaxies_force_pointsource = False

    kwargs.update(ps=ps, nsigma=nsigma, saddle_fraction=saddle_fraction,
                  saddle_min=saddle_min,
                  subsky_radii=subsky_radii,
                  survey_blob_mask=survey_blob_mask,
                  gaussPsf=gaussPsf, pixPsf=pixPsf, hybridPsf=hybridPsf,
                  release=release,
                  normalizePsf=normalizePsf,
                  apodize=apodize,
                  constant_invvar=constant_invvar,
                  splinesky=splinesky,
                  subsky=subsky,
                  ubercal_sky=ubercal_sky,
                  tycho_stars=tycho_stars,
                  gaia_stars=gaia_stars,
                  large_galaxies=large_galaxies,
                  large_galaxies_force_pointsource=large_galaxies_force_pointsource,
                  fitoncoadds_reweight_ivar=fitoncoadds_reweight_ivar,
                  less_masking=less_masking,
                  min_mjd=min_mjd, max_mjd=max_mjd,
                  reoptimize=reoptimize,
                  iterative=iterative,
                  outliers=outliers,
                  cache_outliers=cache_outliers,
                  use_ceres=ceres,
                  wise_ceres=wise_ceres,
                  unwise_coadds=unwise_coadds,
                  bailout=bail_out,
                  do_calibs=do_calibs,
                  old_calibs_ok=old_calibs_ok,
                  write_metrics=write_metrics,
                  lanczos=lanczos,
                  unwise_dir=unwise_dir,
                  unwise_tr_dir=unwise_tr_dir,
                  unwise_modelsky_dir=unwise_modelsky_dir,
                  galex=galex,
                  galex_dir=galex_dir,
                  command_line=command_line,
                  read_parallel=read_parallel,
                  plots=plots, plots2=plots2, coadd_bw=coadd_bw,
                  force=forceStages, write=write_pickles,
                  record_event=record_event)

    if checkpoint_filename is not None:
        kwargs.update(checkpoint_filename=checkpoint_filename)
        if checkpoint_period is not None:
            kwargs.update(checkpoint_period=checkpoint_period)

    if threads and threads > 1:
        from astrometry.util.timingpool import TimingPool, TimingPoolMeas
        pool = TimingPool(threads, initializer=runbrick_global_init,
                          initargs=[])
        poolmeas = TimingPoolMeas(pool, pickleTraffic=False)
        StageTime.add_measurement(poolmeas)
        mp = multiproc(None, pool=pool)
    else:
        from astrometry.util.ttime import CpuMeas
        mp = multiproc(init=runbrick_global_init, initargs=[])
        StageTime.add_measurement(CpuMeas)
        pool = None
    kwargs.update(mp=mp)

    if nblobs is not None:
        kwargs.update(nblobs=nblobs)
    if blob is not None:
        kwargs.update(blob0=blob)
    if blobxy is not None:
        kwargs.update(blobxy=blobxy)
    if blobradec is not None:
        kwargs.update(blobradec=blobradec)
    if blobid is not None:
        kwargs.update(blobid=blobid)
    if max_blobsize is not None:
        kwargs.update(max_blobsize=max_blobsize)

    pickle_pat = pickle_pat % dict(brick=brick)

    prereqs = {
        'tims':None,
        'refs': 'tims',
        'outliers': 'refs',
        'halos': 'outliers',
        'srcs': 'halos',

        # fitblobs: see below

        'coadds': 'fitblobs',

        # wise_forced: see below

        'fitplots': 'fitblobs',
        'psfplots': 'tims',
        'initplots': 'srcs',

        }

    if 'image_coadds' in stages:
        early_coadds = True
    
    #obiwan
    #early_coadds = True

    if early_coadds:
        if blob_image:
            prereqs.update({
                'image_coadds':'srcs',
                'fitblobs':'image_coadds',
                })
        else:
            prereqs.update({
                'image_coadds':'halos',
                'srcs':'image_coadds',
                'fitblobs':'srcs',
                })
    else:
        prereqs.update({
            'fitblobs':'srcs',
            })

    # not sure how to set up the prereqs here. --galex could always require --wise?
    if wise:
        if galex:
            prereqs.update({
                'wise_forced': 'coadds',
                'galex_forced': 'wise_forced',
                'writecat': 'galex_forced',
                })
        else:
            prereqs.update({
                'wise_forced': 'coadds',
                'writecat': 'wise_forced',
                })
    else:
        if galex:
            prereqs.update({
                'galex_forced': 'coadds',
                'writecat': 'galex_forced',
                })
        else:
            prereqs.update({
                'writecat': 'coadds',
                })

    if fit_on_coadds:
        prereqs.update({
            'fit_on_coadds': 'halos',
            'srcs': 'fit_on_coadds',
        })

    # HACK -- set the prereq to the stage after which you'd like to write out checksums.
    prereqs.update({'checksum': 'outliers'})

    if prereqs_update is not None:
        prereqs.update(prereqs_update)

    initargs.update(W=width, H=height, pixscale=pixscale,
                    target_extent=zoom)
    if bands is not None:
        initargs.update(bands=bands)

    def mystagefunc(stage, mp=None, **kwargs):
        # Update the (pickled) survey output directory, so that running
        # with an updated --output-dir overrides the pickle file.
        picsurvey = kwargs.get('survey',None)
        if picsurvey is not None:
            picsurvey.output_dir = survey.output_dir

        flush()
        if mp is not None and threads is not None and threads > 1:
            # flush all workers too
            mp.map(flush, [[]] * threads)
        staget0 = StageTime()
        R = stagefunc(stage, mp=mp, **kwargs)
        flush()
        if mp is not None and threads is not None and threads > 1:
            mp.map(flush, [[]] * threads)
        info('Resources for stage', stage, ':', StageTime()-staget0)
        return R

    t0 = StageTime()
    R = None
    for stage in stages:
        R = runstage(stage, pickle_pat, mystagefunc, prereqs=prereqs,
                     initial_args=initargs, **kwargs)

    info('All done:', StageTime()-t0)

    if pool is not None:
        pool.close()
        pool.join()
    return R

def flush(x=None):
    sys.stdout.flush()
    sys.stderr.flush()

class StageTime(Time):
    '''
    A Time subclass that reports overall CPU use, assuming multiprocessing.
    '''
    measurements = []
    @classmethod
    def add_measurement(cls, m):
        cls.measurements.append(m)
    def __init__(self):
        self.meas = [m() for m in self.measurements]

def get_parser():
    import argparse
    de = ('Main "pipeline" script for the Legacy Survey ' +
          '(DECaLS, MzLS, Bok) data reductions.')

    ep = '''
e.g., to run a small field containing a cluster:

python -u legacypipe/runbrick.py --plots --brick 2440p070 --zoom 1900 2400 450 950 -P pickles/runbrick-cluster-%%s.pickle

'''
    parser = argparse.ArgumentParser(description=de,epilog=ep)

    parser.add_argument('-r', '--run', default=None,
                        help='Set the run type to execute')

    parser.add_argument(
        '-f', '--force-stage', dest='force', action='append', default=[],
        help="Force re-running the given stage(s) -- don't read from pickle.")
    parser.add_argument('-F', '--force-all', dest='forceall',
                        action='store_true', help='Force all stages to run')
    parser.add_argument('-s', '--stage', dest='stage', default=[],
                        action='append', help="Run up to the given stage(s)")
    parser.add_argument('-n', '--no-write', dest='write', default=True,
                        action='store_false')
    parser.add_argument('-w', '--write-stage', action='append', default=None,
                        help='Write a pickle for a given stage: eg "tims", "image_coadds", "srcs"')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')

    parser.add_argument(
        '--checkpoint', dest='checkpoint_filename', default=None,
        help='Write to checkpoint file?')
    parser.add_argument(
        '--checkpoint-period', type=int, default=None,
        help='Period for writing checkpoint files, in seconds; default 600')

    parser.add_argument('-b', '--brick',
        help='Brick name to run; required unless --radec is given')

    parser.add_argument('--radec', nargs=2,
        help='RA,Dec center for a custom location (not a brick)')
    parser.add_argument('--pixscale', type=float, default=0.262,
                        help='Pixel scale of the output coadds (arcsec/pixel)')
    parser.add_argument('-W', '--width', type=int, default=3600,
                        help='Target image width, default %(default)i')
    parser.add_argument('-H', '--height', type=int, default=3600,
                        help='Target image height, default %(default)i')
    parser.add_argument('--zoom', type=int, nargs=4,
                        help='Set target image extent (default "0 3600 0 3600")')

    parser.add_argument('-d', '--outdir', dest='output_dir',
                        help='Set output base directory, default "."')

    parser.add_argument('--release', default=None, type=int,
                        help='Release code for output catalogs (default determined by --run)')

    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Override the $LEGACY_SURVEY_DIR environment variable')

    parser.add_argument('--blob-mask-dir', type=str, default=None,
                        help='The base directory to search for blob masks during sky model construction')

    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to search for cached files')

    parser.add_argument('--threads', type=int, help='Run multi-threaded')
    parser.add_argument('-p', '--plots', dest='plots', action='store_true',
                        help='Per-blob plots?')
    parser.add_argument('--plots2', action='store_true',
                        help='More plots?')

    parser.add_argument(
        '-P', '--pickle', dest='pickle_pat',
        help='Pickle filename pattern, default %(default)s',
        default='pickles/runbrick-%(brick)s-%%(stage)s.pickle')

    parser.add_argument('--plot-base',
                        help='Base filename for plots, default brick-BRICK')
    parser.add_argument('--plot-number', type=int, default=0,
                        help='Set PlotSequence starting number')

    parser.add_argument('--ceres', default=False, action='store_true',
                        help='Use Ceres Solver for all optimization?')

    parser.add_argument('--no-wise-ceres', dest='wise_ceres', default=True,
                        action='store_false',
                        help='Do not use Ceres Solver for unWISE forced phot')

    parser.add_argument('--nblobs', type=int,help='Debugging: only fit N blobs')
    parser.add_argument('--blob', type=int, help='Debugging: start with blob #')
    parser.add_argument('--blobid', help='Debugging: process this list of (comma-separated) blob ids.')
    parser.add_argument(
        '--blobxy', type=int, nargs=2, default=None, action='append',
        help=('Debugging: run the single blob containing pixel <bx> <by>; '+
              'this option can be repeated to run multiple blobs.'))
    parser.add_argument(
        '--blobradec', type=float, nargs=2, default=None, action='append',
        help=('Debugging: run the single blob containing RA,Dec <ra> <dec>; '+
              'this option can be repeated to run multiple blobs.'))

    parser.add_argument('--max-blobsize', type=int,
                        help='Skip blobs containing more than the given number of pixels.')

    parser.add_argument(
        '--check-done', default=False, action='store_true',
        help='Just check for existence of output files for this brick?')
    parser.add_argument('--skip', default=False, action='store_true',
                        help='Quit if the output catalog already exists.')
    parser.add_argument('--skip-coadd', default=False, action='store_true',
                        help='Quit if the output coadd jpeg already exists.')

    parser.add_argument(
        '--skip-calibs', dest='do_calibs', default=True, action='store_false',
        help='Do not run the calibration steps')

    parser.add_argument(
        '--old-calibs-ok', dest='old_calibs_ok', default=False, action='store_true',
        help='Allow old calibration files (where the data validation does not necessarily pass).')

    parser.add_argument('--skip-metrics', dest='write_metrics', default=True,
                        action='store_false',
                        help='Do not generate the metrics directory and files')

    parser.add_argument('--nsigma', type=float, default=6.0,
                        help='Set N sigma source detection thresh')

    parser.add_argument('--saddle-fraction', type=float, default=0.1,
                        help='Fraction of the peak height for selecting new sources.')

    parser.add_argument('--saddle-min', type=float, default=2.0,
                        help='Saddle-point depth from existing sources down to new sources (sigma).')

    parser.add_argument(
        '--reoptimize', action='store_true', default=False,
        help='Do a second round of model fitting after all model selections')

    parser.add_argument(
        '--no-iterative', dest='iterative', action='store_false', default=True,
        help='Turn off iterative source detection?')

    parser.add_argument('--no-wise', dest='wise', default=True,
                        action='store_false',
                        help='Skip unWISE forced photometry')

    parser.add_argument(
        '--unwise-dir', default=None,
        help='Base directory for unWISE coadds; may be a colon-separated list')
    parser.add_argument(
        '--unwise-tr-dir', default=None,
        help='Base directory for unWISE time-resolved coadds; may be a colon-separated list')

    parser.add_argument('--galex', dest='galex', default=False,
                        action='store_true',
                        help='Perform GALEX forced photometry')
    parser.add_argument(
        '--galex-dir', default=None,
        help='Base directory for GALEX coadds')

    parser.add_argument('--early-coadds', action='store_true', default=False,
                        help='Make early coadds?')
    parser.add_argument('--blob-image', action='store_true', default=False,
                        help='Create "imageblob" image?')

    parser.add_argument(
        '--no-lanczos', dest='lanczos', action='store_false', default=True,
        help='Do nearest-neighbour rather than Lanczos-3 coadds')

    parser.add_argument('--gpsf', action='store_true', default=False,
                        help='Use a fixed single-Gaussian PSF')

    parser.add_argument('--no-hybrid-psf', dest='hybridPsf', default=True,
                        action='store_false',
                        help="Don't use a hybrid pixelized/Gaussian PSF model")

    parser.add_argument('--no-normalize-psf', dest='normalizePsf', default=True,
                        action='store_false',
                        help='Do not normalize the PSF model to unix flux')

    parser.add_argument('--apodize', default=False, action='store_true',
                        help='Apodize image edges for prettier pictures?')

    parser.add_argument(
        '--coadd-bw', action='store_true', default=False,
        help='Create grayscale coadds if only one band is available?')

    parser.add_argument('--bands', default=None,
                        help='Set the list of bands (filters) that are included in processing: comma-separated list, default "g,r,z"')

    parser.add_argument('--no-tycho', dest='tycho_stars', default=True,
                        action='store_false',
                        help="Don't use Tycho-2 sources as fixed stars")

    parser.add_argument('--no-gaia', dest='gaia_stars', default=True,
                        action='store_false',
                        help="Don't use Gaia sources as fixed stars")

    parser.add_argument('--no-large-galaxies', dest='large_galaxies', default=True,
                        action='store_false', help="Don't seed (or mask in and around) large galaxies.")
    parser.add_argument('--min-mjd', type=float,
                        help='Only keep images taken after the given MJD')
    parser.add_argument('--max-mjd', type=float,
                        help='Only keep images taken before the given MJD')

    parser.add_argument('--no-splinesky', dest='splinesky', default=True,
                        action='store_false', help='Use constant sky rather than spline.')
    parser.add_argument('--no-subsky', dest='subsky', default=True,
                        action='store_false', help='Do not subtract the sky background.')
    parser.add_argument('--no-unwise-coadds', dest='unwise_coadds', default=True,
                        action='store_false', help='Turn off writing FITS and JPEG unWISE coadds?')
    parser.add_argument('--no-outliers', dest='outliers', default=True,
                        action='store_false', help='Do not compute or apply outlier masks')
    parser.add_argument('--cache-outliers', default=False,
                        action='store_true', help='Use outlier-mask file if it exists?')

    parser.add_argument('--bail-out', default=False, action='store_true',
                        help='Bail out of "fitblobs" processing, writing all blobs from the checkpoint and skipping any remaining ones.')

    parser.add_argument('--fit-on-coadds', default=False, action='store_true',
                        help='Fit to coadds rather than individual CCDs (e.g., large galaxies).')
    parser.add_argument('--no-ivar-reweighting', dest='fitoncoadds_reweight_ivar',
                        default=True, action='store_false',
                        help='Reweight the inverse variance when fitting on coadds.')
    parser.add_argument('--no-galaxy-forcepsf', dest='large_galaxies_force_pointsource',
                        default=True, action='store_false',
                        help='Do not force PSFs within galaxy mask.')
    parser.add_argument('--less-masking', default=False, action='store_true',
                        help='Turn off background fitting within MEDIUM mask.')

    parser.add_argument('--ubercal-sky', dest='ubercal_sky', default=False,
                        action='store_true', help='Use the ubercal sky-subtraction (only used with --fit-on-coadds and --no-subsky).')
    parser.add_argument('--subsky-radii', type=float, nargs=3, default=None,
                        help="""Sky-subtraction radii: rmask, rin, rout [arcsec] (only used with --fit-on-coadds and --no-subsky).
                        Image pixels r<rmask are fully masked and the pedestal sky background is estimated from an annulus
                        rin<r<rout on each CCD centered on the targetwcs.crval coordinates.""")
    parser.add_argument('--read-serial', dest='read_parallel', default=True,
                        action='store_false', help='Read images in series, not in parallel?')
    return parser

def get_runbrick_kwargs(survey=None,
                        brick=None,
                        radec=None,
                        run=None,
                        survey_dir=None,
                        output_dir=None,
                        cache_dir=None,
                        check_done=False,
                        skip=False,
                        skip_coadd=False,
                        stage=None,
                        unwise_dir=None,
                        unwise_tr_dir=None,
                        unwise_modelsky_dir=None,
                        galex_dir=None,
                        write_stage=None,
                        write=True,
                        gpsf=False,
                        bands=None,
                        **opt):
    if stage is None:
        stage = []
    if brick is not None and radec is not None:
        print('Only ONE of --brick and --radec may be specified.')
        return None, -1
    opt.update(radec=radec)

    if survey is None:
        from legacypipe.runs import get_survey
        survey = get_survey(run,
                            survey_dir=survey_dir,
                            output_dir=output_dir,
                            cache_dir=cache_dir)
        info(survey)

    blobdir = opt.pop('blob_mask_dir', None)
    if blobdir is not None:
        from legacypipe.survey import LegacySurveyData
        opt.update(survey_blob_mask=LegacySurveyData(blobdir))

    if check_done or skip or skip_coadd:
        if skip_coadd:
            fn = survey.find_file('image-jpeg', output=True, brick=brick)
        else:
            fn = survey.find_file('tractor', output=True, brick=brick)
        info('Checking for', fn)
        exists = os.path.exists(fn)
        if skip_coadd and exists:
            return survey,0
        if exists:
            try:
                T = fits_table(fn)
                info('Read', len(T), 'sources from', fn)
            except:
                print('Failed to read file', fn)
                import traceback
                traceback.print_exc()
                exists = False

        if skip:
            if exists:
                return survey,0
        elif check_done:
            if not exists:
                print('Does not exist:', fn)
                return survey,-1
            info('Found:', fn)
            return survey,0

    if len(stage) == 0:
        stage.append('writecat')

    opt.update(stages=stage)

    # Remove opt values that are None.
    toremove = [k for k,v in opt.items() if v is None]
    for k in toremove:
        del opt[k]

    if unwise_dir is None:
        unwise_dir = os.environ.get('UNWISE_COADDS_DIR', None)
    if unwise_tr_dir is None:
        unwise_tr_dir = os.environ.get('UNWISE_COADDS_TIMERESOLVED_DIR', None)
    if unwise_modelsky_dir is None:
        unwise_modelsky_dir = os.environ.get('UNWISE_MODEL_SKY_DIR', None)
        if unwise_modelsky_dir is not None and not os.path.exists(unwise_modelsky_dir):
            raise RuntimeError('The directory specified in $UNWISE_MODEL_SKY_DIR does not exist!')
    if galex_dir is None:
        galex_dir = os.environ.get('GALEX_DIR', None)
    opt.update(unwise_dir=unwise_dir, unwise_tr_dir=unwise_tr_dir,
               unwise_modelsky_dir=unwise_modelsky_dir, galex_dir=galex_dir)

    # list of strings if -w / --write-stage is given; False if
    # --no-write given; True by default.
    if write_stage is not None:
        write_pickles = write_stage
    else:
        write_pickles = write
    opt.update(write_pickles=write_pickles)

    opt.update(gaussPsf=gpsf,
               pixPsf=not gpsf)

    if bands is not None:
        bands = bands.split(',')
    opt.update(bands=bands)
    return survey, opt

def main(args=None):
    import datetime
    from legacypipe.survey import get_git_version

    print()
    print('runbrick.py starting at', datetime.datetime.now().isoformat())
    print('legacypipe git version:', get_git_version())
    if args is None:
        print('Command-line args:', sys.argv)
        cmd = 'python'
        for vv in sys.argv:
            cmd += ' {}'.format(vv)
        print(cmd)
    else:
        print('Args:', args)
    print()

    parser = get_parser()
    parser.add_argument(
        '--ps', help='Run "ps" and write results to given filename?')
    parser.add_argument(
        '--ps-t0', type=int, default=0, help='Unix-time start for "--ps"')

    opt = parser.parse_args(args=args)

    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    optdict = vars(opt)
    ps_file = optdict.pop('ps', None)
    ps_t0   = optdict.pop('ps_t0', 0)
    verbose = optdict.pop('verbose')

    survey, kwargs = get_runbrick_kwargs(**optdict)
    if kwargs in [-1, 0]:
        return kwargs
    kwargs.update(command_line=' '.join(sys.argv))

    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    if opt.plots:
        import matplotlib
        matplotlib.use('Agg')
        import pylab as plt
        plt.figure(figsize=(12,9))
        plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.93,
                            hspace=0.2, wspace=0.05)

    if ps_file is not None:
        import threading
        from collections import deque
        from legacypipe.utils import run_ps_thread
        ps_shutdown = threading.Event()
        ps_queue = deque()
        def record_event(msg):
            from time import time
            ps_queue.append((time(), msg))
        kwargs.update(record_event=record_event)
        if ps_t0 > 0:
            record_event('start')

        ps_thread = threading.Thread(
            target=run_ps_thread,
            args=(os.getpid(), os.getppid(), ps_file, ps_shutdown, ps_queue),
            name='run_ps')
        ps_thread.daemon = True
        print('Starting thread to run "ps"')
        ps_thread.start()

    debug('kwargs:', kwargs)

    rtn = -1
    try:
        run_brick(opt.brick, survey, **kwargs)
        rtn = 0
    except NothingToDoError as e:
        print()
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        print()
        rtn = 0
    except RunbrickError as e:
        print()
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        print()
        rtn = -1

    if ps_file is not None:
        # Try to shut down ps thread gracefully
        ps_shutdown.set()
        print('Attempting to join the ps thread...')
        ps_thread.join(1.0)
        if ps_thread.isAlive():
            print('ps thread is still alive.')

    return rtn

if __name__ == '__main__':
    from astrometry.util.ttime import MemMeas
    Time.add_measurement(MemMeas)
    sys.exit(main())

# Test bricks & areas

# A single, fairly bright star
# python -u legacypipe/runbrick.py -b 1498p017 -P 'pickles/runbrick-z-%(brick)s-%%(stage)s.pickle' --zoom 1900 2000 2700 2800
# python -u legacypipe/runbrick.py -b 0001p000 -P 'pickles/runbrick-z-%(brick)s-%%(stage)s.pickle' --zoom 80 380 2970 3270
