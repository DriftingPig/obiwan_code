import astropy.io.fits as fits
import numpy as np
import healpy as hp
from scipy.stats import binom

mehdi_bins = [7.02702134e-03, 2.55666104e-02, 4.41061995e-02, 6.26457886e-02,
       8.11853777e-02, 9.97249668e-02, 1.18264556e-01, 1.36804145e-01,
       1.55343734e-01, 2.28763794e+02, 6.33866348e+02, 1.03896890e+03,
       1.44407146e+03, 1.84917401e+03, 2.25427657e+03, 2.65937912e+03,
       3.06448167e+03, 3.46958423e+03, 2.29994796e+01, 2.30884464e+01,
       2.31774132e+01, 2.32663800e+01, 2.33553468e+01, 2.34443136e+01,
       2.35332804e+01, 2.36222472e+01, 2.37112140e+01, 2.35933077e+01,
       2.36789212e+01, 2.37645346e+01, 2.38501481e+01, 2.39357615e+01,
       2.40213750e+01, 2.41069884e+01, 2.41926018e+01, 2.42782153e+01,
       2.26207504e+01, 2.27106404e+01, 2.28005304e+01, 2.28904204e+01,
       2.29803103e+01, 2.30702003e+01, 2.31600903e+01, 2.32499803e+01,
       2.33398703e+01, 2.31730928e+01, 2.32682611e+01, 2.33634295e+01,
       2.34585978e+01, 2.35537661e+01, 2.36489344e+01, 2.37441028e+01,
       2.38392711e+01, 2.39344394e+01, 2.37500343e+01, 2.38394661e+01,
       2.39288978e+01, 2.40183296e+01, 2.41077613e+01, 2.41971931e+01,
       2.42866248e+01, 2.43760566e+01, 2.44654883e+01, 2.28993138e+01,
       2.30033768e+01, 2.31074398e+01, 2.32115028e+01, 2.33155659e+01,
       2.34196289e+01, 2.35236919e+01, 2.36277549e+01, 2.37318180e+01,
       2.13103427e+01, 2.14002311e+01, 2.14901196e+01, 2.15800080e+01,
       2.16698964e+01, 2.17597849e+01, 2.18496733e+01, 2.19395617e+01,
       2.20294501e+01, 2.06719850e+01, 2.08013537e+01, 2.09307224e+01,
       2.10600911e+01, 2.11894598e+01, 2.13188285e+01, 2.14481972e+01,
       2.15775659e+01, 2.17069345e+01, 1.28054805e+00, 1.41219781e+00,
       1.54384758e+00, 1.67549734e+00, 1.80714710e+00, 1.93879686e+00,
       2.07044662e+00, 2.20209638e+00, 2.33374614e+00, 1.47777746e+00,
       1.60505604e+00, 1.73233462e+00, 1.85961320e+00, 1.98689178e+00,
       2.11417036e+00, 2.24144894e+00, 2.36872751e+00, 2.49600609e+00,
       9.69267151e-01, 1.05680562e+00, 1.14434409e+00, 1.23188256e+00,
       1.31942103e+00, 1.40695950e+00, 1.49449797e+00, 1.58203644e+00,
       1.66957491e+00]


def random_hpix(N_tot=2987042,pix_tot = 98027, seed = 233423):
    #N_lrgsv3 = 5228721, N_mock1 = 2987042, pix_tot_ndecals = 110503, bmzls = 98027
    np.random.seed(seed=seed)
    randoms = binom.rvs(N_tot, 1./pix_tot, size=pix_tot)
    return randoms
    
def run_random_sys():
    fn_maps = "/global/cscratch1/sd/huikong/LSSutils/mock_data/0.57.0/nlrg_features_bmzls_256.fits"
    savedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/mock_var/'

    #features:
    #['EBV', 'STARDENS', 'galdepth_rmag_ebv', 'galdepth_gmag_ebv', 'galdepth_zmag_ebv', 'psfdepth_rmag_ebv', 'psfdepth_gmag_ebv', 'psfdepth_zmag_ebv', 'psfdepth_w1mag_ebv', 'psfdepth_w2mag_ebv', 'PSFSIZE_R', 'PSFSIZE_G', 'PSFSIZE_Z']
    #psfdepth, limits, consistent with the scripts for systematics
    lim_low = {'g':24.3,'r':23.75,'z':22.7,'w1':21.25}
    lim_high = {'g':25.00,'r':24.50,'z':23.7,'w1':21.55}
     
    idx_g = 6
    idx_r = 5
    idx_z = 7
    idx_w1 = 8
     
    footprint = fits.getdata(fn_maps)
    for j in range(1,1000):
        print(j)
        mock_i = random_hpix(seed = 233423+j)
        var_g  = footprint['features'][:,6]
        var_r  = footprint['features'][:,5]
        var_z  = footprint['features'][:,7]
        var_w1 = footprint['features'][:,8]
        var_w2 = footprint['features'][:,9]
        for band in ['g','r','z','w1']:
            nbins = []
            #bins = np.linspace(lim_low[band],lim_high[band],9)
            bins = np.array(mehdi_bins[eval("idx_%s"%band)*9:(eval("idx_%s"%band)+1)*9])
            if j==1:
                print(bins)
            for i in range(8):
                sel_lrg = (eval('var_%s'%band)>bins[i])&(eval('var_%s'%band)<bins[i+1])
                n_mocklrg = mock_i[sel_lrg].sum()
                n_random = sel_lrg.sum()
                ave = mock_i.sum()/len(mock_i)
                nbins.append(np.array(n_mocklrg/n_random/ave))
            bin_center = (bins[1:]+bins[:-1])/2.
            np.savetxt(savedir+'rmock_%d_%s.txt'%(j,band), (bin_center,nbins))    

def run_random_cov():
    savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/mock_var/"
    fn_sys = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/mock_var/rmock_%d_%s.txt"
    for band in ['g','r','z','w1']:
        dat_all = []
        std = np.zeros(8)
        for i in range(1,1000):
            dat = np.loadtxt(fn_sys%(i,band))[1]
            dat_all.append(dat)
        mean = np.zeros(8)
        for dat_all_i in dat_all:
            mean += dat_all_i
   
        mean = mean/len(dat_all)
        for dat_all_i in dat_all:
            std += (dat_all_i - mean)**2
        std = np.sqrt(std/999*(999/998))
        np.savetxt(savedir+'rmock_cov_%s.txt'%band,std)            
            
def run_mock_sys():
    fn_mock = "/global/cscratch1/sd/huikong/LSSutils/mock_data/v1/lrg-zero-%d-f1z1.fits"
    fn_maps = "/global/cscratch1/sd/huikong/LSSutils/mock_data/0.57.0/nlrg_features_bmzls_256.fits"
    savedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/mock_var/'

    #features:
    #['EBV', 'STARDENS', 'galdepth_rmag_ebv', 'galdepth_gmag_ebv', 'galdepth_zmag_ebv', 'psfdepth_rmag_ebv', 'psfdepth_gmag_ebv', 'psfdepth_zmag_ebv', 'psfdepth_w1mag_ebv', 'psfdepth_w2mag_ebv', 'PSFSIZE_R', 'PSFSIZE_G', 'PSFSIZE_Z']
    #psfdepth, limits, consistent with the scripts for systematics
    lim_low = {'g':24.3,'r':23.75,'z':22.7,'w1':21.25}
    lim_high = {'g':25.00,'r':24.50,'z':23.7,'w1':21.55}
     
    idx_g = 6
    idx_r = 5
    idx_z = 7
    idx_w1 = 8
     
    footprint = fits.getdata(fn_maps)
    for j in range(1,1000):
        print(j)
        fn = fn_mock%j
        mock_i = hp.read_map(fn)[footprint['hpix']] #RING, nside=256, cut to ndecals
        var_g  = footprint['features'][:,6]
        var_r  = footprint['features'][:,5]
        var_z  = footprint['features'][:,7]
        var_w1 = footprint['features'][:,8]
        var_w2 = footprint['features'][:,9]
        for band in ['g','r','z','w1']:
            nbins = []
            #bins = np.linspace(lim_low[band],lim_high[band],9)
            bins = np.array(mehdi_bins[eval("idx_%s"%band)*9:(eval("idx_%s"%band)+1)*9])
            if j==1:
                print(bins)
            for i in range(8):
                sel_lrg = (eval('var_%s'%band)>bins[i])&(eval('var_%s'%band)<bins[i+1])
                n_mocklrg = mock_i[sel_lrg].sum()
                n_random = sel_lrg.sum()
                ave = mock_i.sum()/len(mock_i)
                nbins.append(np.array(n_mocklrg/n_random/ave))
            bin_center = (bins[1:]+bins[:-1])/2.
            np.savetxt(savedir+'mock_%d_%s.txt'%(j,band), (bin_center,nbins))

def run_mock_jknife():
    N = 20
    import multiprocessing as mp
    p = mp.Pool(N)
    mocks_split = np.array_split(np.arange(1000)+1,N)
    p.map(_run_mock_jknife,mocks_split)
def _run_mock_jknife(mock_ids):
    fn_mock = "/global/cscratch1/sd/huikong/LSSutils/mock_data/v1/lrg-zero-%d-f1z1.fits"
    fn_maps = "/global/cscratch1/sd/huikong/LSSutils/mock_data/0.57.0/nlrg_features_ndecals_256.fits"
    savedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/mock_var/'

    #features:
    #['EBV', 'STARDENS', 'galdepth_rmag_ebv', 'galdepth_gmag_ebv', 'galdepth_zmag_ebv', 'psfdepth_rmag_ebv', 'psfdepth_gmag_ebv', 'psfdepth_zmag_ebv', 'psfdepth_w1mag_ebv', 'psfdepth_w2mag_ebv', 'PSFSIZE_R', 'PSFSIZE_G', 'PSFSIZE_Z']
    #psfdepth, limits, consistent with the scripts for systematics
    lim_low = {'g':24.3,'r':23.75,'z':22.7,'w1':21.25}
    lim_high = {'g':25.00,'r':24.50,'z':23.7,'w1':21.55}

    
    
    footprint = fits.getdata(fn_maps)
    for band in ['g','r','z','w1']:
        bins = np.linspace(lim_low[band],lim_high[band],9)
        bin_center = (bins[1:]+bins[:-1])/2.
        for j in mock_ids:
            print(j)
            nbins_all = []
            fn = fn_mock%j
            N_samples = 10
            ids = np.arange(len(footprint))
            ids_split = np.array_split(ids,N_samples)
            for k in range(N_samples):
                print("jk:%d"%k)
                jk_id = ids_split[k]
                mock_i = hp.read_map(fn)[footprint['hpix']] #RING, nside=256, cut to ndecals
                var_g  = footprint['features'][:,6].copy()
                var_r  = footprint['features'][:,5].copy()
                var_z  = footprint['features'][:,7].copy()
                var_w1 = footprint['features'][:,8].copy()
                var_w2 = footprint['features'][:,9].copy()
                var_g[jk_id]=0
                var_r[jk_id]=0
                var_z[jk_id]=0
                var_w1[jk_id]=0
                jknife = np.ones(len(footprint),dtype=np.bool)
                jknife[jk_id] = False
                mock_i[jk_id] = 0
                nbins = []
                for i in range(8):
                    sel_lrg = (eval('var_%s'%band)>bins[i])&(eval('var_%s'%band)<bins[i+1])&jknife
                    n_mocklrg = mock_i[sel_lrg].sum()
                    n_random = sel_lrg.sum()
                    ave = mock_i.sum()/jknife.sum()
                    nbins.append(np.array(n_mocklrg/n_random/ave))
                nbins_all.append(np.array(nbins))
            np.savetxt(savedir+'mock_jk_%d_%s.txt'%(j,band), np.array(nbins_all))
            std = np.zeros(8)
            mean = np.zeros(8)
            for nbins_all_i in nbins_all:
                mean += nbins_all_i
            mean = mean/N_samples
            for nbins_all_i in nbins_all:
                std += (nbins_all_i-mean)**2
            std = np.sqrt(std/N_samples*(N_samples-1))
            np.savetxt(savedir+'cov_jk_%d_%s.txt'%(j,band), np.array(std))

def get_mean_cov_jk():
    #mean cov from 1000 mocks
    fn = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/mock_var/cov_jk_%d_%s.txt'
    for band in ['g','r','z','w1']:
        mean = np.zeros(8)
        for i in range(1000):
            dat = np.loadtxt(fn%(i,band))
            mean += dat
        mean = mean/1000
        np.savetxt("/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/mock_var/cov_mock_mean_%s.txt"%band,mean)
            
def run_mock_cov():
    savedir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/mock_var/"
    fn_sys = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/mock_var/mock_%d_%s.txt"
    for band in ['g','r','z','w1']:
        dat_all = []
        std = np.zeros(8)
        for i in range(1,1000):
            dat = np.loadtxt(fn_sys%(i,band))[1]
            dat_all.append(dat)
        mean = np.zeros(8)
        for dat_all_i in dat_all:
            mean += dat_all_i
   
        mean = mean/len(dat_all)
        for dat_all_i in dat_all:
            std += (dat_all_i - mean)**2
        std = np.sqrt(std/999*(999/998))
        np.savetxt(savedir+'cov_%s.txt'%band,std)

            
if __name__ == "__main__":
    #run_mock_jknife()
    #get_mean_cov()
    #run_mock_sys()
    #run_mock_cov()
    run_random_sys()
    run_random_cov()
    