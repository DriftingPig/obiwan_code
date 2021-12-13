import astropy.io.fits as fits
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy as hp

topdir_map = "/global/cfs/cdirs/desi/users/huikong/maps/"
map1024 = fits.getdata(topdir_map+"maps1024.fits")

sel = (map1024['psfdepth_rmag_ebv']>0)&(map1024['psfdepth_gmag_ebv']>0)&(map1024['psfdepth_zmag_ebv']>0)&(map1024['psfdepth_w1mag_ebv']>0)&(map1024['rs0_num_full']>0)
obiwan_psf_g_1024  = np.zeros((3,8))
obiwan_psf_r_1024  = np.zeros((3,8))
obiwan_psf_z_1024  = np.zeros((3,8))
obiwan_psf_w1_1024 = np.zeros((3,8))

lrg_psf_g_1024  = np.zeros((3,8))
lrg_psf_r_1024  = np.zeros((3,8))
lrg_psf_z_1024  = np.zeros((3,8))
lrg_psf_w1_1024 = np.zeros((3,8))

lim_low = {'g':24.2,'r':23.75,'z':22.7,'w1':21.25}
lim_high = {'g':25.00,'r':24.50,'z':23.7,'w1':21.55}

for band in ['g','r','z','w1']:
    from astropy.table import Table, vstack
    var = map1024['psfdepth_%smag_ebv'%band][sel]
    ratios2 = []
    ratios_err2 = []
    ratios3 = []
    ratios4 = []
    low = lim_low[band]
    high = lim_high[band]
    #low = np.percentile(var,3)
    #high = np.percentile(var,97)
    
    bins = np.linspace(low,high,9)
    for i in range(0,8):
    
        sel_lrg = (var>bins[i])&(var<bins[i+1])
        n_lrg = map1024[sel][sel_lrg]['lrg_num_full'].sum()
        n_random = map1024[sel][sel_lrg]['ran_num_full'].sum()
        n_obiwan = map1024[sel][sel_lrg]['obiwan_num_full'].sum()
        n_obiwanr = map1024[sel][sel_lrg]['rs0_num_full'].sum()
        
        ratios_err2.append(1./np.sqrt(n_lrg))
        ratios2.append(n_lrg/n_random)
        ratios3.append(n_obiwan/n_obiwanr)
        
    x = (bins[1:]+bins[:-1])/2.
    y = ratios2/np.array(ratios2).mean()
    y2 = ratios3/np.array(ratios3).mean()
    
    new_pix_array = np.array_split(np.arange(sel.sum()),10)

    patches = []
    for i in range(10):
       patches.append(np.zeros(sel.sum(), dtype = np.bool))
       for j in range(10):
           if i!=j:
              patches[i][new_pix_array[j]] = True 

    y_jk = []
    y_jk2 = []
    for i in range(10):

        idx = patches[i]
        ratios2 = []
        var = map1024['psfdepth_%smag_ebv'%band][sel][idx]
        ratios2 = []
        ratios_err2 = []
        ratios3 = []
        ratios4 = []
        low = lim_low[band]
        high = lim_high[band]
        #low = np.percentile(var,3)
        #high = np.percentile(var,97)
        bins = np.linspace(low,high,9)
        for i in range(0,8):

            sel_lrg = (var>bins[i])&(var<bins[i+1])
            n_lrg = map1024[sel][idx][sel_lrg]['lrg_num_full'].sum()
            n_random = map1024[sel][idx][sel_lrg]['ran_num_full'].sum()
            n_obiwan = map1024[sel][idx][sel_lrg]['obiwan_num_full'].sum()
            n_obiwanr = map1024[sel][idx][sel_lrg]['rs0_num_full'].sum()

            ratios2.append(n_lrg/n_random)
            ratios3.append(n_obiwan/n_obiwanr)

        y0 = ratios2/np.array(ratios2).mean()
        y20 = ratios3/np.array(ratios3).mean()
        y_jk.append(y0)
        y_jk2.append(y20)

    y_error = None
    y_error2 = None
    for l in range(10):
        if y_error is not None:
            y_error += (y_jk[l]-y)**2
            y_error2 += (y_jk2[l]-y2)**2
        else:
            y_error = (y_jk[l]-y)**2
            y_error2 = (y_jk2[l]-y2)**2
    y_error = np.sqrt(y_error*10/9)
    y_error2 = np.sqrt(y_error2*10/9)

    exec('obiwan_psf_%s_1024[0] = x'%band)
    exec('obiwan_psf_%s_1024[1] = y2'%band)
    exec('obiwan_psf_%s_1024[2] = y_error2'%band)
    exec('lrg_psf_%s_1024[0] = x'%band)
    exec('lrg_psf_%s_1024[1] = y'%band)
    exec('lrg_psf_%s_1024[2] = y_error'%band)

savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
np.savetxt(savedir+'obiwan_g_1024.txt', obiwan_psf_g_1024)
np.savetxt(savedir+'obiwan_r_1024.txt', obiwan_psf_r_1024)
np.savetxt(savedir+'obiwan_z_1024.txt', obiwan_psf_z_1024)
np.savetxt(savedir+'obiwan_w1_1024.txt', obiwan_psf_w1_1024)

np.savetxt(savedir+'lrg_g_1024.txt', lrg_psf_g_1024)
np.savetxt(savedir+'lrg_r_1024.txt', lrg_psf_r_1024)
np.savetxt(savedir+'lrg_z_1024.txt', lrg_psf_z_1024)
np.savetxt(savedir+'lrg_w1_1024.txt', lrg_psf_w1_1024)


