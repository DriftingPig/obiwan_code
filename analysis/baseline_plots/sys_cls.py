import sys
sys.path.append("/global/homes/h/huikong/obiwan_analysis/py")
import astropy.io.fits as fits
from astropy.table import Table
import numpy as np
import healpy as hp
import subprocess
import glob
from utils import multiproc
import fitsio
from astropy.table import Table, vstack
import random as rdm

class systematics(object):
    lim_low = {'g':24.3,'r':23.75,'z':22.7,'w1':21.25}
    lim_high = {'g':25.00,'r':24.50,'z':23.7,'w1':21.55}
    
    ebv_low = 0.011
    ebv_high = 0.124
    star_low = 286
    star_high = 2440
    
    def __init__(self, feature = None, sample_g = None, sample_r = None, res = None, nest = None, mode = None, savedir = None, savefn = None, bands = ['g','r', 'z']):
        assert(mode in ['hpix', 'inf'])
        self.feature = feature
        self.sample_g = sample_g
        self.sample_r = sample_r
        self.res = res
        self.nest = nest
        self.mode = mode
        self.bands = bands
        self.savedir = savedir
        self.savefn = savefn
        
    def run_inf(self, jk_list = None):
        for band in self.bands:
            bins = np.linspace(self.lim_low[band],self.lim_high[band],9)
            ave = len(self.sample_g)/len(self.sample_r)
            ratios2 = []
            for i in range(0,8):
                if jk_list is None:
                    var = -2.5*(np.log10(5/np.sqrt(self.sample_g['psfdepth_%s'%band]))-9)
                    var_random = -2.5*(np.log10(5/np.sqrt(self.sample_r['psfdepth_%s'%band]))-9)
                else:
                    sel_lrg = self.sample_r['lrg_sv3'][jk_list]
                    var        = -2.5*(np.log10(5/np.sqrt(self.sample_r['psfdepth_%s'%band]))-9)[jk_list][sel_lrg]
                    var_random = -2.5*(np.log10(5/np.sqrt(self.sample_r['psfdepth_%s'%band]))-9)[jk_list]
                sel_lrg = (var>bins[i])&(var<bins[i+1])
                sel_random = (var_random>bins[i])&(var_random<bins[i+1])
                ratio = sel_lrg.sum()/sel_random.sum()/ave
                ratios2.append(ratio)
            x = (bins[1:]+bins[:-1])/2.
            y = np.array(ratios2)
            if jk_list is None:
                np.savetxt(self.savedir+self.savefn%band, np.array([x,y]))
                print("written "+self.savedir+self.savefn%band)
            else:
                return y
    def run_inf_jk(self, tot_jk = 10):
        for band in ['g','r','z']:
            self.bands = [band]
            N = len(self.sample_r)
            ids = np.arange(N)
            rdm.shuffle(ids)
            ids_split = np.array_split(ids,tot_jk)
            y_list = []
            for i in range(tot_jk):
                ids_jk = []
                for j in range(tot_jk):
                    if i!=j:
                        ids_jk.append(ids_split[j])
                ids_jk_final = np.concatenate(ids_jk)
                y = self.run_inf(jk_list = ids_jk_final)
                y_list.append(y)
            y_mean = np.zeros(len(y_list[0]))
            y_std = np.zeros(len(y_list[0]))
            for i in range(tot_jk):
                y_mean += y_list[i]
            y_mean /= tot_jk
            for i in range(tot_jk):
                y_std += (y_list[i]-y_mean)**2
            y_std = np.sqrt(y_std/tot_jk*(tot_jk-1))
            np.savetxt(self.savedir+self.savefn%band, np.array(y_std))
            
            
    def run_res(self,psfsize = False, ebv = False, star = False):
        if psfsize:
            self.psfsize_low = {"g":1.18,"r":1.08,"z":1.02}
            self.psfsize_high = {"g":1.9,"r":1.77,"z":1.62}
            self.lim_low = {'g':self.psfsize_low["g"], 'r':self.psfsize_low['r'],'z':self.psfsize_low['z']}
            self.lim_high = {'g':self.psfsize_high["g"],'r':self.psfsize_high["r"],'z':self.psfsize_high["z"]}
        if ebv:
            self.lim_low = {'x':self.ebv_low}
            self.lim_high = {'x':self.ebv_high}
        if star:
            self.lim_low = {'x':self.star_low}
            self.lim_high = {'x':self.star_high}
            
        for band in self.bands:
            bins = np.linspace(self.lim_low[band],self.lim_high[band],9)
            ave = self.feature[self.sample_g].sum()/self.feature[self.sample_r].sum()
            ratios = []
            for i in range(0,8):
                if ebv is False and star is False and psfsize is False:
                    var = self.feature['psfdepth_%smag'%band]
                if psfsize:
                    var = self.feature['psfsize_%s'%band]
                if ebv:
                    var = self.feature["ebv"]
                if star:
                    var = self.feature["stardens"]
                sel = (var>bins[i])&(var<bins[i+1])
                n_lrg = (self.feature[self.sample_g][sel]).sum()
                n_random = (self.feature[self.sample_r][sel]).sum()
                ratios.append(n_lrg/n_random/ave)
            x = (bins[1:]+bins[:-1])/2.
            y = np.array(ratios)
            np.savetxt(self.savedir+self.savefn%band, np.array([x,y]))
            print("written "+self.savedir+self.savefn%band)
            
if __name__ == "__main__":
    
    topdir_subset = "/global/cfs/cdirs/desi/users/huikong/decals_ngc/subset/"
    lrg = fitsio.read(topdir_subset + "subset_dr9_lrg_sv3.fits", columns = ['ra','dec','maskbits','psfdepth_g','psfdepth_r','psfdepth_z','psfdepth_w1','nobs_g','nobs_r','nobs_z'])
    random = fitsio.read(topdir_subset + "subset_random.fits", columns = ['ra','dec','maskbits', 'psfdepth_g', 'psfdepth_r', 'psfdepth_z', 'psfdepth_w1','nobs_g','nobs_r','nobs_z'])
    obiwan_rs0 = fitsio.read(topdir_subset + "subset_rs0.fits", columns = ['lrg_sv3','ra','dec','maskbits', 'psfdepth_g', 'psfdepth_r', 'psfdepth_z', 'psfdepth_w1','psfdepth_w1','nobs_g','nobs_r','nobs_z'])
    obiwan_more_rs0 = fitsio.read(topdir_subset + "subset_more_rs0.fits", columns = ['lrg_sv3','ra','dec','maskbits', 'psfdepth_g', 'psfdepth_r', 'psfdepth_z', 'psfdepth_w1','nobs_g','nobs_r','nobs_z']) 
    
    sel_rs0_lrg = (obiwan_rs0['lrg_sv3'])&(obiwan_rs0['maskbits']==0)&(obiwan_rs0['psfdepth_r']>0)&(obiwan_rs0['psfdepth_g']>0)&(obiwan_rs0['psfdepth_z']>0)&(obiwan_rs0['psfdepth_w1']>0)
    sel_mrs0_lrg = (obiwan_more_rs0['lrg_sv3'])&(obiwan_more_rs0['maskbits']==0)&(obiwan_more_rs0['psfdepth_r']>0)&(obiwan_more_rs0['psfdepth_g']>0)&(obiwan_more_rs0['psfdepth_z']>0)&(obiwan_more_rs0['psfdepth_w1']>0)
    
    sample_g = vstack((Table(obiwan_rs0[sel_rs0_lrg]), Table(obiwan_more_rs0[sel_mrs0_lrg])))
    
    sel_rs0 = (obiwan_rs0['maskbits']==0)&(obiwan_rs0['nobs_g']>0)&(obiwan_rs0['nobs_r']>0)&(obiwan_rs0['nobs_z']>0)&(obiwan_rs0['maskbits']==0)&(obiwan_rs0['psfdepth_r']>0)&(obiwan_rs0['psfdepth_g']>0)&(obiwan_rs0['psfdepth_z']>0)&(obiwan_rs0['psfdepth_w1']>0)
    
    sel_more_rs0 = (obiwan_more_rs0['maskbits']==0)&(obiwan_more_rs0['nobs_g']>0)&(obiwan_more_rs0['nobs_r']>0)&(obiwan_more_rs0['nobs_z']>0)&(obiwan_more_rs0['psfdepth_r']>0)&(obiwan_more_rs0['psfdepth_g']>0)&(obiwan_more_rs0['psfdepth_z']>0)&(obiwan_more_rs0['psfdepth_w1']>0)
    """
    #obiwan sys inf
    sample_r = vstack((Table(obiwan_rs0[sel_rs0]), Table(obiwan_more_rs0[sel_more_rs0])))
    savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
    obiwan_sys = systematics(feature = None, sample_g = sample_g, sample_r = sample_r, res = None, nest = None, mode = 'inf', savedir = savedir, savefn = 'obiwan_%s_psfdepth_inf.txt', bands = ['g','r', 'z'])
    obiwan_sys.run_inf()
    obiwan_sys.savefn = 'obiwan_%s_psfdepth_inf_std.txt'
    obiwan_sys.run_inf_jk()
    """
    
    """
    #lrg sys inf
    savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
    sel_lrg = (lrg['psfdepth_z']>0)&(lrg['psfdepth_r']>0)&(lrg['psfdepth_g']>0)&(lrg['psfdepth_w1']>0)&(lrg['maskbits']==0)
    sel_random = (random['psfdepth_z']>0)&(random['psfdepth_r']>0)&(random['psfdepth_g']>0)&(random['psfdepth_w1']>0)&(random['maskbits']==0)
    lrg_sys = systematics(feature = None, sample_g = lrg[sel_lrg], sample_r = random[sel_random], res = None, nest = None, mode = 'inf', savedir = savedir, savefn = 'lrg_%s_psfdepth_inf.txt', bands = ['g','r', 'z'])
    lrg_sys.run_inf()
    """

    """
    
    #obiwan sys 1024
    savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
    map1024 = fits.getdata("/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits")
    lrg_sys = systematics(feature = map1024, sample_g = 'lrg_num', sample_r = 'all_num', res = 1024, nest = False, mode = 'hpix', savedir = savedir, savefn = 'obiwan_%s_psfdepth_hpix1024.txt', bands = ['g','r', 'z','w1'])
    lrg_sys.run_res()
    """
    
    """
    #lrg sys 1024
    savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
    map1024 = fits.getdata("/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits")
    lrg_sys = systematics(feature = map1024, sample_g = 'lrgsv3_num', sample_r = 'randoms_masked', res = 1024, nest = False, mode = 'hpix', savedir = savedir, savefn = 'lrgsv3_%s_psfdepth_hpix1024.txt', bands = ['g','r', 'z','w1'])
    lrg_sys.run_res()
    """
    
    """
    #psfsize 1024
    savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
    map1024 = fits.getdata("/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits")
    lrg_sys = systematics(feature = map1024, sample_g = 'lrg_num', sample_r = 'all_num', res = 1024, nest = False, mode = 'hpix', savedir = savedir, savefn = 'obiwan_%s_psfsize_hpix1024.txt', bands = ['g','r', 'z'])
    lrg_sys.run_res(psfsize = True)
    """
    
    """
    #lrg psfsize 1024
    savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
    map1024 = fits.getdata("/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits")
    lrg_sys = systematics(feature = map1024, sample_g = 'lrgsv3_num', sample_r = 'randoms_masked', res = 1024, nest = False, mode = 'hpix', savedir = savedir, savefn = 'lrg_%s_psfsize_hpix1024.txt', bands = ['g','r', 'z'])
    lrg_sys.run_res(psfsize = True)
    """
    
    #lrg ebv 1024
    savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
    map1024 = fits.getdata("/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits")
    lrg_sys = systematics(feature = map1024, sample_g = 'lrgsv3_num', sample_r = 'randoms_masked', res = 1024, nest = False, mode = 'hpix', savedir = savedir, savefn = 'lrg_%s_ebv_hpix1024.txt', bands = ['x'])
    lrg_sys.run_res(ebv = True)
    
    #obiwan ebv 1024
    savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
    map1024 = fits.getdata("/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits")
    lrg_sys = systematics(feature = map1024, sample_g = 'lrg_num', sample_r = 'all_num', res = 1024, nest = False, mode = 'hpix', savedir = savedir, savefn = 'obiwan_%s_ebv_hpix1024.txt', bands = ['x'])
    lrg_sys.run_res(ebv = True)
    
    #lrg star 1024
    savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
    map1024 = fits.getdata("/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits")
    lrg_sys = systematics(feature = map1024, sample_g = 'lrgsv3_num', sample_r = 'randoms_masked', res = 1024, nest = False, mode = 'hpix', savedir = savedir, savefn = 'lrg_%s_star_hpix1024.txt', bands = ['x'])
    lrg_sys.run_res(star = True)
    
    #obiwan star 1024
    savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/savedir/"
    map1024 = fits.getdata("/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits")
    lrg_sys = systematics(feature = map1024, sample_g = 'lrg_num', sample_r = 'all_num', res = 1024, nest = False, mode = 'hpix', savedir = savedir, savefn = 'obiwan_%s_star_hpix1024.txt', bands = ['x'])
    lrg_sys.run_res(star = True)
    
    