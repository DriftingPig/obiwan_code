import astropy.io.fits as fits
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy as hp

topdir_map = "/global/cfs/cdirs/desi/users/huikong/maps/"
map1024 = fits.getdata(topdir_map+"maps1024.fits")

topdir_subset = "/global/cfs/cdirs/desi/users/huikong/decals_ngc/subset/"
lrg = fits.getdata(topdir_subset + "subset_dr9_lrg_sv3.fits")
random = fits.getdata(topdir_subset + "subset_random.fits")
obiwan = fits.getdata(topdir_subset + "subset_rs0.fits")

obiwan_psf_g  = np.zeros((3,8))
obiwan_psf_r  = np.zeros((3,8))
obiwan_psf_z  = np.zeros((3,8))
obiwan_psf_w1 = np.zeros((3,8))

#selection for "randoms"
sel0 = (obiwan['psfdepth_z']>0)&(obiwan['psfdepth_r']>0)&(obiwan['psfdepth_g']>0)&(obiwan['psfdepth_w1']>0)&(obiwan['matched'])&(obiwan['maskbits']==0)
#selection for "data"
sel1 = (obiwan['lrg_sv3'][sel0])&(obiwan['matched'][sel0])

lim_low = {'g':24.2,'r':23.75,'z':22.7,'w1':21.25}
lim_high = {'g':25.00,'r':24.50,'z':23.7,'w1':21.55}

coeffs = {'g':3.214, 'r':2.165, 'z':1.211, 'w1':0.184}

pix_i = hp.pixelfunc.ang2pix(256, obiwan[sel0]['ra'], obiwan[sel0]['dec'], lonlat=True)
#pix_g = hp.pixelfunc.ang2pix(256, obiwan[sel0]['ra'][sel1], obiwan[sel0]['dec'][sel1], lonlat=True)
pixs = np.unique(pix_i)
new_pixs = np.sort(pixs)
new_pix_array = np.array_split(new_pixs,10)
patches = []
#patches_g = []
for i in range(10):
    patches.append(np.zeros_like(pix_i, dtype = np.bool))
    for j in new_pix_array[i]:
        idx = np.where(pix_i == j)[0]
        patches[i][idx] = True 
        #idx = np.where(pix_g == j)[0]
        #patches_g[i][idx] = True

#idxes = np.array_split(arr,10)
#psfdepth, inf for optical 
for band in ['g','r','z']:
    from astropy.table import Table, vstack
    var = -2.5*(np.log10(5/np.sqrt(obiwan['psfdepth_%s'%band][sel0]))-9)-coeffs[band]*obiwan['ebv'][sel0]
    ratios2 = []
    bins = np.linspace(lim_low[band],lim_high[band],9)
    for i in range(0,8):
        ssel = (var>bins[i])&(var<bins[i+1])
        sel_lrg = ssel&sel1

        ratio = sel_lrg.sum()/ssel.sum()
        ratios2.append(ratio)
    x = (bins[1:]+bins[:-1])/2.
    y = ratios2/np.array(ratios2).mean()
    

    y_jk = []
    for i in range(10):
        ratios2 = []
        idx = np.zeros_like(pix_i,dtype=np.bool)
        #idx_g = np.zeros_like(pix_g)
        for j in range(10):
            if j != i:
                    idx += patches[j]
        #print(len(idx),len(data))
        var = -2.5*(np.log10(5/np.sqrt(obiwan['psfdepth_%s'%band][sel0][idx]))-9)-coeffs[band]*obiwan['ebv'][sel0][idx]
        ratios2 = []
        for k in range(0,8):
            ssel = (var>bins[k])&(var<bins[k+1])
            sel_lrg = ssel&sel1[idx]
            ratio = sel_lrg.sum()/ssel.sum()
            ratios2.append(ratio)
        y_i = ratios2/np.array(ratios2).mean()
        y_jk.append(y_i)

    y_error = None
    for l in range(10):
        if y_error is not None:
            y_error += (y_jk[l]-y)**2
        else:
            y_error = (y_jk[l]-y)**2
    y_error = np.sqrt(y_error*10/9)
    exec('obiwan_psf_%s[0] = x'%band)
    exec('obiwan_psf_%s[1] = y'%band)
    exec('obiwan_psf_%s[2] = y_error'%band)

savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
np.savetxt(savedir+'obiwan_g.txt', obiwan_psf_g)
np.savetxt(savedir+'obiwan_r.txt', obiwan_psf_r)
np.savetxt(savedir+'obiwan_z.txt', obiwan_psf_z)

