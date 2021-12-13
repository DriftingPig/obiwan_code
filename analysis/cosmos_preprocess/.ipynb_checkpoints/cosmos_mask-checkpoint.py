import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import glob
import os
savedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/'
import subprocess
subprocess.call(["mkdir",savedir+'maskbits'])
#mask here is defined as regions with g/r/z depth>0 across all bands

def main():
    topdir = '/global/cscratch1/sd/dstn/dr9-cosmos-subs/'
    sets = ['80', '81', '82', '83', '84', '85', '86', '87', '88', '89']
    baseline = os.path.join(topdir,sets[0])
    baseline_fns = glob.glob(os.path.join(baseline,'tractor','*','tractor-*'))
    baseline_fns.sort()

    for baseline_fn in baseline_fns:
        maskbit_img = np.zeros((3600,3600))
        maskbit_fn = []
        brickname = os.path.basename(baseline_fn).replace('tractor-','').replace('.fits','')
        print(brickname)
        maskbit_fn.append(os.path.join(topdir,sets[0],'coadd',brickname[:3],brickname,'legacysurvey-'+brickname+'-depth-g.fits.fz'))
        maskbit_fn.append(os.path.join(topdir,sets[0],'coadd',brickname[:3],brickname,'legacysurvey-'+brickname+'-depth-r.fits.fz'))
        maskbit_fn.append(os.path.join(topdir,sets[0],'coadd',brickname[:3],brickname,'legacysurvey-'+brickname+'-depth-z.fits.fz'))
        for one_set in sets[1:]:
            ccd_alt_fn_g = maskbit_fn[0].replace('/80/','/'+one_set+'/')
            ccd_alt_fn_r = maskbit_fn[1].replace('/80/','/'+one_set+'/')
            ccd_alt_fn_z = maskbit_fn[2].replace('/80/','/'+one_set+'/')
            maskbit_fn.append(ccd_alt_fn_g)
            maskbit_fn.append(ccd_alt_fn_r)
            maskbit_fn.append(ccd_alt_fn_z)
        for one_maskbit_fn in maskbit_fn:
            img = fits.getdata(one_maskbit_fn)
            maskbit_img+=(img==0)
        bool_maskbit_img = (maskbit_img==0)
        hdu = fits.ImageHDU(data=bool_maskbit_img.astype(int))
        hdu.writeto(savedir+'/maskbits/%s.fits'%brickname,overwrite=True)
        
main()
        
    
