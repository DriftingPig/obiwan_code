import numpy as np
from legacypipe.survey import *
from astropy.table import vstack,Table
import astropy.io.fits as fits
from astrometry.util.fits import fits_table
import glob
import os
#shifter --module=mpich-cle6 --image=legacysurvey/legacypipe:DR9.7.1 /bin/bash  
topdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/deep1/divided_randoms/"
region = 'south'
fns = glob.glob(topdir+'*')
for fn in fns:
    dat = fits_table(fn)
    brickname = os.path.basename(fn).replace("brick_","").replace(".fits","")
    survey = LegacySurveyData(survey_dir=None)
    brickinfo= survey.get_brick_by_name(brickname)
    brickwcs = wcs_for_brick(brickinfo)
    W, H, pixscale = brickwcs.get_width(), brickwcs.get_height(), brickwcs.pixel_scale()
    targetwcs = wcs_for_brick(brickinfo, W=W, H=H, pixscale=pixscale)
    flag, target_x, target_y = targetwcs.radec2pixelxy(dat.ra, dat.dec)
    dat.set('sim_x', target_x)
    dat.set('sim_y', target_y)
    maskbits_dr9 = fits.getdata("/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/%s/coadd/%s/%s/legacysurvey-%s-maskbits.fits.fz"%(region, brickname[:3],brickname,brickname))
    nexp_g = fits.getdata("/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/%s/coadd/%s/%s/legacysurvey-%s-nexp-g.fits.fz"%(region, brickname[:3],brickname,brickname))
    nexp_r = fits.getdata("/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/%s/coadd/%s/%s/legacysurvey-%s-nexp-r.fits.fz"%(region, brickname[:3],brickname,brickname))
    nexp_z = fits.getdata("/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/%s/coadd/%s/%s/legacysurvey-%s-nexp-z.fits.fz"%(region, brickname[:3],brickname,brickname))
    bx = (target_x+0.5).astype(int)
    by = (target_y+0.5).astype(int)  
    mask_flag = maskbits_dr9[(by),(bx)]
    mask_flag_g = nexp_g[(by),(bx)]
    mask_flag_r = nexp_r[(by),(bx)]
    mask_flag_z = nexp_z[(by),(bx)]
    dat.set('nobs_g', mask_flag_g)
    dat.set('nobs_r', mask_flag_r) 
    dat.set('nobs_z', mask_flag_z)
    dat.set('maskbits', mask_flag)
    dat.writeto(fn,overwrite=True)
    print(brickname)



