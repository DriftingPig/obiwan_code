"""
This is a script for making per brick randoms 
It needs to be in a legacysurvey docker enviroment to use the wcs function for maskbits columns
to use this, change the outdir, bricklist path in make_randoms function
#TODO:
more user friendly interface
"""
import numpy as np
import astropy.io.fits as fits
from astropy.table import vstack,Table
import os
#make a set of randoms within the bricks we are interested in, and masked out unwanted region

import numpy as np
from astrometry.util.fits import fits_table
#making randoms containing ra dec x y maskbits matched_cosmos
def _make_randoms(X):
    (brickname,outdir) = X
    margin = 0.
    print(brickname)
    fn = outdir+'random-%s.fits'%(brickname)
    import os
    if os.path.isfile(fn):
        return None
    from legacypipe.survey import wcs_for_brick,LegacySurveyData
    
    from tractor.sfd import SFDMap
    os.environ['LEGACY_SURVEY_DIR']='/global/cfs/cdirs/cosmo/work/legacysurvey/dr9'
    os.environ['DUST_DIR']='/global/cfs/projectdirs/cosmo/data/dust/v0_1'#
    survey = LegacySurveyData(survey_dir=None)
    
    surveybricks = fits.getdata('/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/survey-bricks.fits.gz')
    surveybrick_i = surveybricks[surveybricks['BRICKNAME']==brickname]
    ra1 = surveybrick_i['RA1'][0]-margin
    ra2 = surveybrick_i['RA2'][0]+margin
    dec1 = surveybrick_i['DEC1'][0]-margin
    dec2 = surveybrick_i['DEC2'][0]+margin
    
    random_state=np.random.RandomState()
    u1,u2= random_state.uniform(size=(2, 5000) )
    cmin = np.sin(dec1*np.pi/180)
    cmax = np.sin(dec2*np.pi/180)
    RA   = ra1 + u1*(ra2-ra1)
    DEC  = 90-np.arccos(cmin+u2*(cmax-cmin))*180./np.pi
    
    brick = survey.get_brick_by_name(brickname)
    brickwcs = wcs_for_brick(brick)
    #import pdb;pdb.set_trace()
    W, H, pixscale = brickwcs.get_width(), brickwcs.get_height(), brickwcs.pixel_scale()
    print(W,H,pixscale)
    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    flag, target_x, target_y = targetwcs.radec2pixelxy(RA, DEC)
    
    #sel = (target_x>=0)&(target_x<3599)&(target_y>=0)&(target_y<3599)
    sel=np.ones_like(RA,dtype=np.bool)
    
    T = fits_table()
    T.set('id',np.arange(sel.sum()))
    T.set('ra',RA[sel])
    T.set('dec',DEC[sel])
    T.set('bx',target_x[sel])
    T.set('by',target_y[sel])
    T.set('brickname',np.array([brickname]*sel.sum(),dtype=np.str))
    ebv = SFDMap().ebv(RA[sel],DEC[sel])
    T.set('ebv',ebv)
    maskbits_dr9 = fits.getdata("/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/south/coadd/%s/%s/legacysurvey-%s-maskbits.fits.fz"%(brickname[:3],brickname,brickname))
    nexp_g = fits.getdata("/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/south/coadd/%s/%s/legacysurvey-%s-nexp-g.fits.fz"%(brickname[:3],brickname,brickname))
    nexp_r = fits.getdata("/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/south/coadd/%s/%s/legacysurvey-%s-nexp-r.fits.fz"%(brickname[:3],brickname,brickname))
    nexp_z = fits.getdata("/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/south/coadd/%s/%s/legacysurvey-%s-nexp-z.fits.fz"%(brickname[:3],brickname,brickname))
    bx = (T.bx+0.5).astype(int)
    by = (T.by+0.5).astype(int)
    mask_flag = maskbits_dr9[(by),(bx)]
    mask_flag_g = nexp_g[(by),(bx)]
    mask_flag_r = nexp_r[(by),(bx)]
    mask_flag_z = nexp_z[(by),(bx)]
    T.set('maskbits',mask_flag)
    T.set('nobs_g',mask_flag_g)
    T.set('nobs_r',mask_flag_r)
    T.set('nobs_z',mask_flag_z)
    fn = outdir+'random-%s.fits'%(brickname)
    print(fn)
    T.writeto(fn,overwrite=True)
    print("written %s, length=%d"%(fn,len(target_x[sel])))
    
def make_randoms():    
    #cosmos 
    outdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_new_seed/output/rs0_cosmos80/'
    bricklist = np.loadtxt('/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_new_seed/bricklist.txt',dtype=np.str)
    #new run
    outdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_resampled_seed/randoms/"
    bricklist = np.loadtxt('/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_new_seed/bricklist.txt',dtype=np.str)
    #deep1
    outdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/deep1/randoms/"
    bricklist = np.loadtxt('/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/brickstat/deep1/FinishedBricks.txt',dtype=np.str)
    import subprocess
    subprocess.call(["mkdir",outdir])
    import multiprocessing as mp
    p = mp.Pool(30)
    lists = []
    for brickname in bricklist:
        lists.append((brickname, outdir))
    p.map(_make_randoms, lists)
    
    
#shifter --module=mpich-cle6 --image=legacysurvey/legacypipe:DR9.7.1 /bin/bash        
make_randoms()