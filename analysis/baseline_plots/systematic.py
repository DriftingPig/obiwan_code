#systematics plots directly from runs
#generate systematics maps on a per brick level and add them up
#this propably only useful for running a small region, thus not quite useful
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os
from astropy.table import Table
from mpi4py import MPI

def _text_loc(idx, y_value,x_value):
    
    y_cen = (y_value.max()-y_value.min())*(6-idx)/8+y_value.min()
    x_cen = (x_value.max()-x_value.min())*5/8+x_value.min()
    return y_cen, x_cen

def random_mask(randoms):
    bits = [1, 5, 6, 7, 8, 11, 12, 13]
    maskbits = np.ones_like(randoms['maskbits'], dtype='?')
    for bit in bits:
        maskbits &= ((randoms['maskbits']& 2**bit)==0)
    
    sel_obs = (randoms['nobs_g']>0)&(randoms['nobs_r']>0)&(randoms['nobs_z']>0)
    return randoms[maskbits&sel_obs]

def source_mask(source,i=0):
    #sel = source['matched_cosmos']&(source['maskbits']==0)&source['set_%d_lrg_sv3'%i]#&source['matched']&source['set_80_lrg_sv3']&source['set_%d_lrg_sv3'%i]&
    #1, 8, 9, 12, 13
    bits = [1, 5, 6, 7, 8, 11, 12, 13]
    maskbits = np.ones_like(source['maskbits'], dtype='?')
    for bit in bits:
        maskbits &= ((source['maskbits']& 2**bit)==0)
    sel_obs = (source['nobs_g']>0)&(source['nobs_r']>0)&(source['nobs_z']>0)
    sel = sel_obs&maskbits&source['lrg_sv3']
    print(sel.sum())
    return source[sel]

def radec_plot(source1,source2,topdir,sysmax,sysmin):
    print(sysmax,sysmin)
    plt.clf()
    plt.plot(source1['ra'],source1['dec'],'r,')
    sel = (source1['ebv']>sysmin)&(source1['ebv']<sysmax)
    plt.plot(source1['ra'][sel],source1['dec'][sel],'g,')
    plt.plot(source2['ra'],source2['dec'],'b.')
    plt.savefig(topdir+'/fig_radec.png')
    

def sys_maps(catalog, sys = 'ebv', percentile=1,startid=0):
    randoms = fits.getdata(catalog.find_file('random'))
    source = catalog.processed_one
    num = int(catalog.subsection[-2:])-80
    fn = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cosmos_all_stacked/masked_cosmos_set%d.fits'%num
    source = fits.getdata(fn)
    
    randoms = fits.getdata("/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_dr9/randoms_all.fits")
    source = fits.getdata("/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_dr9/cosmos_dr9.fits")
    randoms = random_mask(randoms)
    source = source_mask(source,i=num)
    
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid+catalog.subsection
    
    #resolution
    
    sysmin = np.percentile(source['ebv'],percentile)
    sysmax = np.percentile(source['ebv'],100-percentile)
    nsysbin = 10
    
    ave = len(source)/len(randoms)
    
    sel_source = (source[sys]>=sysmin)&(source[sys]<=sysmax)
    sel_randoms = (randoms[sys]>=sysmin)&(randoms[sys]<=sysmax)
    
    nbins_source,bins,_ = plt.hist(source[sel_source][sys], bins=nsysbin)
    nbins_randoms,bins,_ = plt.hist(randoms[sel_randoms][sys], bins=bins)
    
    import pdb;pdb.set_trace()
    radec_plot(randoms,source,topdir,sysmax=bins[10],sysmin=bins[9])
    
    plt.clf()
    plt.figure(figsize=(6,6))
    
    
    
    x_value = (bins[1:]+bins[:-1])/2.
    width = (bins[:-1]-bins[1:]).mean()
    plt.bar(x_value, nbins_randoms/nbins_randoms.max(),alpha=0.5,width = width*0.99)
    
    y_value = nbins_source/nbins_randoms/ave
    y_error = np.sqrt(nbins_source/(nbins_randoms)**2./(ave)**2.+(nbins_source/ave)**2./(nbins_randoms)**3.)
    
    plt.errorbar(x_value, y_value, y_error)
    plt.title("%s, cutting +/-%d%%"%(sys,percentile))
    plt.plot(x_value,np.ones_like(x_value),'k--')
    #get chi2
    
    chin = np.sum((y_value-1.)**2./y_error**2.)
    z = np.polyfit(x_value,y_value,1,w=1./y_error)
    b=z[1];m=z[0]
    chilin = np.sum((y_value-(m*x_value+b))**2./y_error**2.)
    y_cen,x_cen = _text_loc(0, y_value,x_value)
    plt.text(x_cen,y_cen,"chi2:%.3f,lin:%.3f"%(chin/(nsysbin-1),chilin/(nsysbin-1)))
    
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid+catalog.subsection
    import subprocess
    subprocess.call(["mkdir","-p",topdir])
    plt.savefig(topdir+'/fig_systematics_%s.png'%sys)
    

def sys_from_map(catalog, startid, sysname,percentile):
    upperdir = os.path.dirname(os.path.abspath(catalog.outdir))
    fn_maps = "/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/pixweight/main/resolve/dark/pixweight-1-dark.fits"
    fn_randoms = "%s/subset/subset_random.fits"%upperdir
    fn_data = "%s/subset/subset_rs%d.fits"%(upperdir,startid)
    fn_data_dr9 = "%s/subset/subset_dr9_LRG_sv3.fits"%upperdir
    outdir = catalog.outdir
    get_sys(fn_randoms,fn_data,fn_data_dr9, sysname,percentile,outdir)

def sys_real_lrgs(slice_idx = None, slice_tot = None):
    #topdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/LRG_SV3/sweep/"
    #fn_data_dr9 = topdir+'LRG_SV3_south_desi.fits'
    #topdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/LRG_SV3/truth/"
    #fn_randoms = topdir+'randoms_south_desi.fits'
    fn_rs0 = "/global/cfs/cdirs/desi/users/huikong/decals_ngc/subset/subset_rs0.fits"
    fn_data_dr9 = "/global/cfs/cdirs/desi/users/huikong/decals_ngc/subset/subset_dr9_lrg_sv3.fits"
    fn_randoms = "/global/cfs/cdirs/desi/users/huikong/decals_ngc/subset/subset_random.fits"
    #fn_maps = "/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/pixweight/main/resolve/dark/pixweight-1-dark.fits"
    fn_maps =  "/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/0.49.0/resolve/combined/pixmap_south_nside_256_minobs_1_maskbits_189111213.fits"
    #fn_maps = "/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/0.49.0/resolve/combined/pixmap_south_nside_1024_minobs_1_maskbits_189111213.fits"
    new_tab = Table.read(fn_maps)
    import fitsio
    data_dr9 = fitsio.read(fn_data_dr9,columns=['ra','dec','maskbits'])
    #data_dr9 = fits.getdata(fn_data_dr9)
    randoms = fitsio.read(fn_randoms,columns=['ra','dec','maskbits','nobs_g','nobs_r','nobs_z'])
    #randoms = fits.getdata(fn_randoms)
    rs0 = fitsio.read(fn_rs0,columns=['ra','dec','maskbits','lrg_sv3','matched'])
    #rs0 = fits.getdata(fn_rs0)
    data_dr9 = data_dr9[(data_dr9['maskbits']==0)]
    sel = (randoms['nobs_g'] > 0) & (randoms['nobs_r'] > 0) & (randoms['nobs_z'] > 0) & (randoms['maskbits']==0)
    randoms = randoms[sel]
    res = 256
    obiwan = rs0[rs0['lrg_sv3']&rs0['matched']&(rs0['maskbits']==0)]
    pixs_obiwan = hp.ang2pix(res,obiwan['ra'],obiwan['dec'],nest = False, lonlat = True)
    rs0 = rs0[(rs0['maskbits']==0)]
    pixsr1 = hp.ang2pix(res,rs0['ra'],rs0['dec'],nest = False, lonlat = True)
    pixsr = hp.ang2pix(res,randoms['ra'],randoms['dec'],nest = False, lonlat = True)
    pixsdr9 = hp.ang2pix(res,data_dr9['ra'],data_dr9['dec'],nest = False, lonlat = True)
    N = len(new_tab)
    hist_n = np.zeros(N)
    hist_n3 = np.zeros(N)
    hist_n4 = np.zeros(N)
    hist_nr = np.zeros(N)
    for i in range(slice_idx, N, slice_tot):
        idx = new_tab['HPXPIXEL'][i]
        if i%1000==0:
            print(i,N)
        u = len(np.where(pixsr==idx)[0])
        hist_nr[i]=u
        u2 = len(np.where(pixsdr9==idx)[0])
        hist_n[i] = u2
        u3 = len(np.where(pixs_obiwan==idx)[0])
        hist_n3[i] = u3
        u4 = len(np.where(pixsr1==idx)[0])
        hist_n4[i] = u4
    del(data_dr9)
    del(randoms)
    new_tab['lrg_num'] = hist_n
    new_tab['ran_num'] = hist_nr
    new_tab['rs0_num'] = hist_n4
    new_tab['obiwan_num'] = hist_n3
    new_tab.write("/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/maps_truth/maps256_part%d_of%d.fits"%(slice_idx, slice_tot), overwrite=True)



def get_sys(fn_randoms,fn_data,fn_data_dr9, sysname,percentile,outdir):
    fn_maps = "/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/pixweight/main/resolve/dark/pixweight-1-dark.fits"
    
    maps = fits.getdata(fn_maps)[sysname]
    randoms = fits.getdata(fn_randoms)
    data = fits.getdata(fn_data)
    data_dr9 = fits.getdata(fn_data_dr9)
    data = data[data['lrg_sv3']&(data['maskbits']==0)&data['matched']]
    data_dr9 = data_dr9[(data_dr9['maskbits']==0)]
    
    
    bits = [1, 5, 6, 7, 8, 11, 12, 13]
    for bit in bits:
        mb = ((randoms['maskbits'] & 2**bit) == 0)
        mb = (randoms['maskbits']==0)
    sel = (randoms['nobs_g'] > 0) & (randoms['nobs_r'] > 0) & (randoms['nobs_z'] > 0)&mb
    randoms = randoms[sel]
    
    res = 256
    pixs = hp.ang2pix(res,data['ra'],data['dec'],nest = True, lonlat = True)
    pixsr = hp.ang2pix(res,randoms['ra'],randoms['dec'],nest = True, lonlat = True)
    pixsdr9 = hp.ang2pix(res,data_dr9['ra'],data_dr9['dec'],nest = True, lonlat = True)
    N = 12*res**2
    hist_n = np.zeros(N)
    for i in range(len(pixs)):
        n = pixs[i]
        hist_n[n]+=1

    hist_nr = np.zeros(N)
    for i in range(len(pixsr)):
        n = pixsr[i]
        hist_nr[n]+=1  
    
    hist_dr9 = np.zeros(N)
    for i in range(len(pixsdr9)):
        n = pixsdr9[i]
        hist_dr9[n]+=1
        
    map_loc = np.arange(N)
    sel = (hist_nr>hist_nr.max()*0.9)        
    ra,dec = hp.pix2ang(res,np.arange(N), nest = True,lonlat = True)
    ra = ra[sel]
    dec = dec[sel]
    n_data = hist_n[sel]
    n_uniform = hist_nr[sel]
    n_dr9 = hist_dr9[sel]
    maps = maps[sel]
    
    sysmin = np.percentile(maps,percentile)
    sysmax = np.percentile(maps,100-percentile)
    nsysbin = 10
    ave = len(data)/len(randoms)
    bins_boundary = np.linspace(sysmin, sysmax, nsysbin+1)
    bins_center = (bins_boundary[1:]+bins_boundary[:-1])/2.
    binwidth = (sysmax-sysmin)/nsysbin
    
    bins_data = np.zeros(nsysbin)
    bins_data_tot = 0.
    for i in range(len(n_data)):
        sys = maps[i]
        n_i = n_data[i]
        bin_num = int((sys-sysmin)/binwidth)
        bins_data_tot += n_i
        if bin_num>=0 and bin_num<nsysbin:
            bins_data[bin_num]+=n_i
            
    bins_uniform = np.zeros(nsysbin)
    bins_uniform_tot = 0.        
    for i in range(len(n_uniform)):
        sys = maps[i]
        n_i = n_uniform[i]
        bin_num = int((sys-sysmin)/binwidth)
        bins_uniform_tot += n_i
        if bin_num>=0 and bin_num<nsysbin:
            bins_uniform[bin_num]+=n_i   
            
    bins_dr9 = np.zeros(nsysbin)
    bins_dr9_tot = 0.        
    for i in range(len(n_dr9)):
        sys = maps[i]
        n_i = n_dr9[i]
        bin_num = int((sys-sysmin)/binwidth)
        bins_dr9_tot += n_i
        if bin_num>=0 and bin_num<nsysbin:
            bins_dr9[bin_num]+=n_i   
    
    ave_data = bins_data_tot/bins_uniform_tot
    y_error_data = np.sqrt(bins_data/(bins_uniform)**2./(ave_data)**2.+(bins_data/ave_data)**2./(bins_uniform)**3.)
    
    ave_dr9 = bins_dr9_tot/bins_uniform_tot
    y_error_dr9 = np.sqrt(bins_dr9/(bins_uniform)**2./(ave_dr9)**2.+(bins_dr9/ave_dr9)**2./(bins_uniform)**3.)
    
    ave_dr9_2 = bins_dr9_tot/bins_data_tot
    y_error_dr9_2 = np.sqrt(bins_dr9/(bins_data)**2./(ave_dr9_2)**2.+(bins_dr9/ave_dr9_2)**2./(bins_data)**3.)
    
    
    topdir = outdir+'/rs%d_plots/'%startid
    import subprocess
    subprocess.call(["mkdir","-p",topdir])
    np.savetxt(topdir+'/data.txt', np.array([bins_data/bins_data_tot,y_error_data]))
    np.savetxt(topdir+'/uniform.txt', bins_uniform/bins_uniform_tot)
    np.savetxt(topdir+'/dr9.txt', np.array([bins_dr9/bins_dr9_tot,y_error_dr9,y_error_dr9]))
    np.savetxt(topdir+'/bins.txt', bins_center)
        
        
def make_sys_plot(catalog,startid):
    sys_list = ["STARDENS", "EBV", "PSFDEPTH_G","PSFDEPTH_R","PSFDEPTH_Z","PSFDEPTH_W1","PSFSIZE_G","PSFSIZE_R","PSFSIZE_Z"]
    plt.figure(figsize = (9,9))
    i = 0
    nsysbin=10
    for sys in sys_list:
        i+=1
        sys_from_map(catalog,startid,sysname = sys,percentile=0.1)
        
        topdir = catalog.outdir+'/rs%d_plots/'%startid
        fn1 = topdir+"bins.txt"
        fn2 = topdir+'data.txt'
        fn3 = topdir+'dr9.txt'
        fn4 = topdir+'uniform.txt'
        dat1 = np.loadtxt(fn1)
        dat2 = np.loadtxt(fn2)
        dat3 = np.loadtxt(fn3)
        dat4 = np.loadtxt(fn4)
        
        plt.subplot(3,3,i)
        x_value = dat1
        y_value1 = dat2[0]/dat4
        y_value2 = dat3[0]/dat4
        chin1 = np.sum((y_value1-1.)**2./dat2[1]**2.)
        chin2 = np.sum((y_value2-1.)**2./dat3[1]**2.)
        plt.errorbar(dat1,y_value1,dat2[1],label='obiwan')
        plt.errorbar(dat1,y_value2,dat3[1],label='dr9')
        plt.legend()
        plt.gca().set_ylim((0.9,1.1))
        plt.xlabel(sys)
        
        y_cen,x_cen = _text_loc(0, np.array([0.9,1.1]),x_value)
        #plt.text(x_cen,y_cen,"obiwan:%.3f"%(chin1/(nsysbin-1)))
        y_cen,x_cen = _text_loc(1, np.array([0.9,1.1]),x_value)
        #plt.text(x_cen,y_cen,"dr9:%.3f"%(chin2/(nsysbin-1)))
        print("%s finished"%sys)
    plt.tight_layout()    
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    import subprocess
    subprocess.call(["mkdir","-p",topdir])
    plt.savefig(topdir+'/fig_systematics.png')
        

def density_plot(catalog,startid):
    plt.clf()
    upperdir = os.path.dirname(os.path.abspath(catalog.outdir))
    fn_randoms = "%s/subset/subset_random.fits"%upperdir
    fn_data = "%s/subset/subset_rs%d.fits"%(upperdir,startid)
    fn_data_dr9 = "%s/subset/subset_dr9_LRG_sv3.fits"%upperdir
    
    randoms = fits.getdata(fn_randoms)
    data = fits.getdata(fn_data)
    data_dr9 = fits.getdata(fn_data_dr9)
    data = data[data['lrg_sv3']]
    
    bits = [1,12,13]
    for bit in bits:
        mb = ((randoms['maskbits'] & 2**bit) == 0)
    sel = (randoms['nobs_g'] > 0) & (randoms['nobs_r'] > 0) & (randoms['nobs_z'] > 0)&mb
    randoms = randoms[sel]
    res = 256
    pixs = hp.ang2pix(res,data['ra'],data['dec'],nest = True, lonlat = True)
    pixsr = hp.ang2pix(res,randoms['ra'],randoms['dec'],nest = True, lonlat = True)
    pixsdr9 = hp.ang2pix(res,data_dr9['ra'],data_dr9['dec'],nest = True, lonlat = True)
    N = 12*res**2
    hist_n = np.zeros(N)
    for i in range(len(pixs)):
        n = pixs[i]
        hist_n[n]+=1

    hist_nr = np.zeros(N)
    for i in range(len(pixsr)):
        n = pixsr[i]
        hist_nr[n]+=1  
    
    hist_dr9 = np.zeros(N)
    for i in range(len(pixsdr9)):
        n = pixsdr9[i]
        hist_dr9[n]+=1
        
    sel = (hist_nr>hist_nr.max()*0.9)        
    ra,dec = hp.pix2ang(res,np.arange(N), nest = True,lonlat = True)
    ra = ra[sel]
    dec = dec[sel]
    n_data = hist_n[sel]
    n_uniform = hist_nr[sel]
    n_dr9 = hist_dr9[sel]

    plt.figure(figsize = (6,4))
    plt.scatter(ra,dec,c=n_data/n_uniform*n_uniform.mean()/n_data.mean(),s=9, cmap = 'seismic',vmax=1.5,vmin=0.5)
    plt.colorbar()
    
    topdir = catalog.outdir+'/rs%d_plots/'%startid
    import subprocess
    
    subprocess.call(["mkdir","-p",topdir])
    plt.savefig(topdir+'/fig_density_obiwan.png')
    plt.clf()
    plt.figure(figsize=(6,4))
    plt.scatter(ra,dec,c=n_dr9/n_uniform*n_uniform.mean()/n_dr9.mean(),s=9,cmap = 'seismic',vmax=1.5,vmin=0.5)
    plt.colorbar()
    plt.savefig(topdir+'/fig_density_dr9.png')
    plt.clf()

#sys_real_lrgs(0,160)
#srun -N 20 -n 80 shifter --module=mpich-cle6 --image=legacysurvey/legacypipe:DR9.7.1 python systematic.py
if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    print(rank)
    sys_real_lrgs(slice_idx = rank, slice_tot = size)
    print("%d/%d finished"%(rank,size))
