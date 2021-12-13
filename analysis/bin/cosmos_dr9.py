#collect cosmos repeats data, add more cards to it
import numpy as np
import astropy.io.fits as fits
import os
import sys
sys.path.append('/global/cscratch1/sd/huikong/Obiwan/desihub/desitarget/py/')
sys.path.append('/global/cscratch1/sd/huikong/Obiwan/desihub/desiutil/py/')
sys.path.append('/global/cscratch1/sd/huikong/Obiwan/desihub/desimodel/py')
from desitarget.sv3 import sv3_cuts
from astropy.table import Table, vstack
bricklist = ["1498p022","1498p025","1498p017","1498p020","1501p017","1501p020","1501p022","1501p025","1503p017","1503p020","1503p022","1503p025","1506p017","1506p020","1506p022","1506p025"]
assert(len(np.unique(bricklist))==16)
topdir_dr9 = "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/"
savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_dr9/"
def get_dr9_tractor():
    final_tab = None
    for brickname in bricklist:
        print(brickname)
        fn = os.path.join(topdir_dr9, 'tractor',brickname[:3],'tractor-%s.fits'%brickname)
        tab = Table.read(fn)
        if final_tab is None:
            final_tab = tab
        else:
            final_tab = vstack((final_tab,tab))
    final_tab.write(savedir+'/cosmos_dr9.fits')
def add_lrg_card(data):
    gflux = data['flux_g']/data['mw_transmission_g']
    rflux = data['flux_r']/data['mw_transmission_r']
    zflux = data['flux_z']/data['mw_transmission_z']
    w1flux = data['flux_w1']/data['mw_transmission_w1']
    zfiberflux = data['fiberflux_z']/data['mw_transmission_z']
    nobs_g = data['nobs_g']
    nobs_r = data['nobs_r']
    nobs_z = data['nobs_z']
    flux_ivar_r = data['flux_ivar_r']
    flux_ivar_z = data['flux_ivar_z']
    flux_ivar_w1 = data['flux_ivar_w1']
    gaia_phot_g_mean_mag = data['gaia_phot_g_mean_mag']
    maskbits = data['maskbits']
    zfibertotflux = data['fibertotflux_z']/data['mw_transmission_z']
    lrg, lrg_lowdens = sv3_cuts.isLRG(gflux = gflux,rflux=rflux,zflux=zflux,w1flux=w1flux,zfiberflux=zfiberflux,\
                         gnobs=nobs_g,rnobs=nobs_r,znobs=nobs_z,\
                        rfluxivar=flux_ivar_r,zfluxivar=flux_ivar_z,w1fluxivar=flux_ivar_w1,\
                        gaiagmag=gaia_phot_g_mean_mag,maskbits=maskbits,zfibertotflux=zfibertotflux,primary=None, south=True)
    new_data = Table(data)
    new_data['lrg_sv3'] = lrg
    new_data.write(savedir+'/cosmos_dr9.fits',overwrite=True)

def random_maker(idx=0,region = 'south'):#idx=0-19
    randoms_fn = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/%s/randoms/randoms-%s-1-%d.fits'%(region,region,idx)
    random = fits.getdata(randoms_fn)
    brickmask = np.zeros(len(random),dtype=np.bool)
    for brickname in bricklist:
        print(brickname)
        sel = (random['BRICKNAME']==brickname)
        print(sel.sum())
        brickmask += sel
    print(brickmask.sum())
    data = Table(random[brickmask])
    print(len(data))
    data.write(savedir+'/randoms_cosmos_%d.fits'%idx,overwrite=True)

def stack_randoms():
    tab = None
    for i in range(20):
        print(i)
        tab_i = Table.read(savedir+'/randoms_cosmos_%d.fits'%i)
        if tab is None:
            tab = tab_i
        else:
            tab = vstack((tab,tab_i))
    tab.write(savedir+'/randoms_all.fits')
get_dr9_tractor()
for i in range(0,20):
    random_maker(idx=i)
data = fits.getdata(savedir+'/cosmos_dr9.fits')
add_lrg_card(data)
stack_randoms()
