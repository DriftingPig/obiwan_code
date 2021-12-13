#this code runs to check flux bias & std against various systematic maps
#I get interesting trends from it
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os
from astropy.table import Table
def sys_from_map(topdir, fn_randoms, fn_data, band, sysname, percentile=0.1):
    #upperdir = os.path.dirname(os.path.abspath(catalog.outdir))
    fn_maps = "/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/pixweight/main/resolve/dark/pixweight-1-dark.fits"
    #fn_randoms = "%s/subset/subset_random.fits"%upperdir
    #fn_data = "%s/subset/subset_rs%d.fits"%(upperdir,startid)
    #fn_data_dr9 = "%s/subset/subset_dr9_LRG_sv3.fits"%upperdir
    
    maps = fits.getdata(fn_maps)[sysname]
    randoms = fits.getdata(fn_randoms)
    data = fits.getdata(fn_data)
    data = data[(data['maskbits']==0)&data['matched']]
    
    data = Table(data)
    data['sim_w1flux'] = data['mw_transmission_w1']*10**(-(data['sim_w1'] - 22.5)/2.5)

    bits = [1, 5, 6, 7, 8, 11, 12, 13]
    for bit in bits:
        mb = (randoms['maskbits']==0)
    sel = (randoms['nobs_g'] > 0) & (randoms['nobs_r'] > 0) & (randoms['nobs_z'] > 0)&mb
    randoms = randoms[sel]
    
    res = 256
    pixs = hp.ang2pix(res,data['ra'],data['dec'],nest = True, lonlat = True)
    pixsr = hp.ang2pix(res,randoms['ra'],randoms['dec'],nest = True, lonlat = True)
    
    N = 12*res**2
    hist_n = np.zeros(N)
    lists_n = [[] for i in range(N)]
    val = (data['flux_%s'%band] - data['sim_%sflux'%band])*np.sqrt(data['flux_ivar_%s'%band])
    for i in range(len(pixs)):
        if np.abs(val[i])<6:
            n = pixs[i]
            hist_n[n]+=1
            lists_n[n].append(data['flux_%s'%band][i] - data['sim_%sflux'%band][i])
            
    sel = (hist_n>0)
    
    idx = np.arange(N)
    idx = idx[sel]
    hist_n = hist_n[sel]

    ra,dec = hp.pix2ang(res,np.arange(N), nest = True,lonlat = True)
    ra = ra[sel]
    dec = dec[sel]
    
    srcs = hist_n
 
    
    maps = maps[sel]
    sysmin = np.percentile(maps,percentile)
    sysmax = np.percentile(maps,100-percentile)
    nsysbin = 10
    
    bins_boundary = np.linspace(sysmin, sysmax, nsysbin+1)
    bins_center = (bins_boundary[1:]+bins_boundary[:-1])/2.
    binwidth = (sysmax-sysmin)/nsysbin
    
    bins_data = np.zeros(nsysbin)
    bins_bias = [[] for i in range(10)]
    #bins_std = [[] for i in range(10)]
    bins_data_tot = 0.
    
    for i in range(sel.sum()):
        sys = maps[i]
        n_i = srcs[i]
        bin_num = int((sys-sysmin)/binwidth)
        bins_data_tot += n_i
        if bin_num>=0 and bin_num<nsysbin:
            bins_data[bin_num]+=n_i
            bins_bias[bin_num].append(np.array(lists_n[idx[i]]).ravel())
    #import pdb;pdb.set_trace()
    bins_bias = [np.concatenate(bins_bias[i]) for i in range(10)]
    bins_bias_final = [bins_bias[i].mean() for i in range(10)]
    bins_std_final = [bins_bias[i].std() for i in range(10)]
    
    y_error_data_bias = [bins_bias[i].std()/np.sqrt(len(bins_bias[i])) for i in range(10)]
    y_error_data_std = [bins_bias[i].std()/len(bins_bias[i]) for i in range(10)]
    
    
    np.savetxt(topdir+'/sys_scatter.txt', np.array([bins_center,bins_bias_final,y_error_data_bias, bins_std_final,y_error_data_std]))
    return np.array([bins_center,bins_bias_final,y_error_data_bias, bins_std_final,y_error_data_std])
        
        
def make_sys_plot(Type,topdir,fn_randoms, fn_data):
    sys_list = ["STARDENS", "EBV", "PSFDEPTH_G","PSFDEPTH_R","PSFDEPTH_Z","PSFDEPTH_W1","PSFSIZE_G","PSFSIZE_R","PSFSIZE_Z"]
    i = 0
    nsysbin=10
    plt.title(Type)
    for band in ['g','r','z','w1']:
        plt.figure(figsize = (9,9))
        i = 0
        for sys in sys_list:
            i+=1
            sys_from_map(topdir, fn_randoms, fn_data, band,sys)
            fn1 = topdir+'/sys_scatter.txt'
            dat = np.loadtxt(fn1)#transpose()?

            plt.subplot(3,3,i)
            x_value = dat[0]
            y_value_bias = dat[1]
            y_value_bias_err = dat[2]
            y_value_std = dat[3]
            y_value_std_err = dat[4]
 
            if Type == 'bias':       
                plt.errorbar(x_value,y_value_bias,y_value_bias_err)
            else:
                plt.errorbar(x_value,y_value_std,y_value_std_err)
            plt.xlabel(sys)  
               
        plt.tight_layout()    
        plt.savefig(topdir+'/fig_systematics_%s_%s.png'%(Type,band))
        plt.clf()
               

topdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/deep1/output/rs9999_plots/"
fn_data = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/deep1/subset/subset_rs9999.fits"
fn_randoms = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/deep1/subset/subset_intput_random.fits"
make_sys_plot(Type = 'bias', topdir = topdir,fn_randoms = fn_randoms, fn_data = fn_data)
make_sys_plot(Type = 'std', topdir = topdir,fn_randoms = fn_randoms, fn_data = fn_data)               
