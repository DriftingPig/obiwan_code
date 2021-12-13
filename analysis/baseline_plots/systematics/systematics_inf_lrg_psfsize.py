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

lrg_psf_g  = np.zeros((3,8))
lrg_psf_r  = np.zeros((3,8))
lrg_psf_z  = np.zeros((3,8))
lrg_psf_w1 = np.zeros((3,8)) 

sel0 = (random['psfdepth_z']>0)&(random['psfdepth_r']>0)&(random['psfdepth_g']>0)&(random['psfdepth_w1']>0)&(random['maskbits']==0)
sel1 = (lrg['psfdepth_z']>0)&(lrg['psfdepth_r']>0)&(lrg['psfdepth_g']>0)&(lrg['psfdepth_w1']>0)&(lrg['maskbits']==0)

lim_low = {'g':24.2,'r':23.75,'z':22.7,'w1':21.25}
lim_high = {'g':25.00,'r':24.50,'z':23.7,'w1':21.55}


arr = np.arange(len(random))
np.random.shuffle(arr)
idxes = np.array_split(arr,10)

coeffs = {'g':3.214, 'r':2.165, 'z':1.211, 'w1':0.184}

pix_i = hp.pixelfunc.ang2pix(256, random[sel0]['ra'], random[sel0]['dec'], lonlat=True)
pix_g = hp.pixelfunc.ang2pix(256, lrg[sel1]['ra'], lrg[sel1]['dec'], lonlat=True)

pixs = np.unique(pix_i)
new_pixs = np.sort(pixs)
new_pix_array = np.array_split(new_pixs,10)
patches = []
patches_g = []
for i in range(10):
    patches.append(np.zeros_like(pix_i, dtype = np.bool))
    patches_g.append(np.zeros_like(pix_g, dtype = np.bool))
    for j in new_pix_array[i]:
        idx = np.where(pix_i == j)[0]
        patches[i][idx] = True 
        idx = np.where(pix_g == j)[0]
        patches_g[i][idx] = True

for band in ['g','r','z']:
    from astropy.table import Table, vstack
    #var = -2.5*(np.log10(5/np.sqrt(random['psfdepth_%s'%band][sel0]))-9)-coeffs[band]*random['ebv'][sel0]
    #var2 = -2.5*(np.log10(5/np.sqrt(lrg['psfdepth_%s'%band][sel1]))-9)-coeffs[band]*lrg['ebv'][sel1]
    var = random['psfsize_%s'%band][sel0]
    var2 = lrg['psfsize_%s'%band][sel1]
    ratios2 = []
    #bins = np.linspace(lim_low[band],lim_high[band],9)
    low = np.percentile(var,3)
    high = np.percentile(var,97)
    bins = np.linspace(low,high,9)

    for i in range(0,8):
        ssel = (var>bins[i])&(var<bins[i+1])
        sel_lrg = (var2>bins[i])&(var2<bins[i+1])
        ratio = sel_lrg.sum()/ssel.sum()
        ratios2.append(ratio)
    x = (bins[1:]+bins[:-1])/2.
    y = ratios2/np.array(ratios2).mean()
    

    y_jk = []
    for i in range(10):
        ratios2 = []
        idx = np.zeros_like(pix_i,dtype=np.bool)
        idx_g = np.zeros_like(pix_g,dtype=np.bool)
        for j in range(10):
            if j != i:
                    idx += patches[j]
                    idx_g += patches_g[j]
        #print(len(idx),len(data))
        #var = -2.5*(np.log10(5/np.sqrt(random['psfdepth_%s'%band][sel0][idx]))-9)-coeffs[band]*random['ebv'][sel0][idx]
        #var2 = -2.5*(np.log10(5/np.sqrt(lrg['psfdepth_%s'%band][sel1][idx_g]))-9)-coeffs[band]*lrg['ebv'][sel1][idx_g]
        var = random['psfsize_%s'%band][sel0][idx]
        var2 = lrg['psfsize_%s'%band][sel1][idx_g]
        ratios2 = []
        for k in range(0,8):
            ssel = (var>bins[k])&(var<bins[k+1])
            sel_lrg = (var2>bins[k])&(var2<bins[k+1])
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

    exec('lrg_psf_%s[0] = x'%band)
    exec('lrg_psf_%s[1] = y'%band)
    exec('lrg_psf_%s[2] = y_error'%band)



savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
np.savetxt(savedir+'lrg_g_psfsize.txt', lrg_psf_g)
np.savetxt(savedir+'lrg_r_psfsize.txt', lrg_psf_r)
np.savetxt(savedir+'lrg_z_psfsize.txt', lrg_psf_z)

