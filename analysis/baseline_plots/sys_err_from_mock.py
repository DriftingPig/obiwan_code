# obtain error bar of systematic weights from mocks
import astropy.io.fits as fits
import numpy as np
import healpy as hp
from scipy.stats import binom
import subprocess
import glob
#1000 mocks in total 
class mock_cov(object):
    psfsize_low = {"g":1.18,"r":1.08,"z":1.02}
    psfsize_high = {"g":1.9,"r":1.77,"z":1.62}
    ebv_low = 0.011
    ebv_high = 0.124
    star_low = 286
    star_high = 2440
    def __init__(self,name,region):
        assert(region in ['ndecals','sdecals','bmzls'])
        self.region = region
        self.database()
        self.fn_maps = self.map
        self.save_fn = name+'_%d_%s.txt'
        self.cov_fn = 'cov_'+name+'_%s.txt'
        self.fn_sys = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/mock_var/"+name+"_%d_%s.txt"
        self.name = name
    def database(self):
        #savedir
        self.savedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/mock_var/'
        #mock name
        self.fn_mock = "/global/cscratch1/sd/huikong/LSSutils/mock_data/v1/lrg-zero-%d-f1z1.fits"
        #number of data from Mehdi's mock
        self.N_mock = 2987042
        #tot number of pixel in ndecals
        self.pix_tot_ndecals = 110503
        #tot number of pixel in bmzls
        self.pix_tot_bmzls = 98027
        #total number of lrgs
        self.N_lrgsv3 = 5228721
        self.obiwan_lrg_ngc = 3165830
        #maps
        self.map = "/global/cscratch1/sd/huikong/LSSutils/mock_data/0.57.0/nlrg_features_%s_256.fits"%self.region
        #features order in the map['features']
        self.features = ['EBV', 'STARDENS', 'galdepth_rmag_ebv', 'galdepth_gmag_ebv', 'galdepth_zmag_ebv', 'psfdepth_rmag_ebv', 'psfdepth_gmag_ebv', 'psfdepth_zmag_ebv', 'psfdepth_w1mag_ebv', 'psfdepth_w2mag_ebv', 'PSFSIZE_R', 'PSFSIZE_G', 'PSFSIZE_Z']
        
    def random_hpix(self):
        # a uniform random catalog based on bimodial distribution
        np.random.seed(seed=self.seed)
        randoms = binom.rvs(self.N_tot, 1./self.pix_tot, size=self.pix_tot)
        return randoms

    def run_sys(self,mode, psfsize=False, ebv = False, star = False):
        #psfdepth, limits, consistent with the scripts for systematics
        lim_low = {'g':24.3,'r':23.75,'z':22.7,'w1':21.25}
        lim_high = {'g':25.00,'r':24.50,'z':23.7,'w1':21.55}

        if psfsize:
            lim_low = {'g':self.psfsize_low["g"], 'r':self.psfsize_low['r'],'z':self.psfsize_low['z']}
            lim_high = {'g':self.psfsize_high["g"],'r':self.psfsize_high["r"],'z':self.psfsize_high["z"]}
        
        if ebv:
            lim_low = {'x':self.ebv_low}
            lim_high = {'x':self.ebv_high}
            idx_x = 0

        if star:
            lim_low = {'x':self.star_low}
            lim_high = {'x':self.star_high}
            idx_x = 1



        #psfdepth
        idx_g = 6
        idx_r = 5
        idx_z = 7
        idx_w1 = 8

        if psfsize:
            idx_g = 10
            idx_r = 11
            idx_z = 12

        footprint = fits.getdata(self.fn_maps)
        for j in range(1,1000):
            print(j)
            self.seed += j
            if mode == 'random':
                mock_i = self.random_hpix()
            elif mode == 'mock':
                fn = self.fn_mock%j
                mock_i = hp.read_map(fn)[footprint['hpix']]
            else:
                raise ValueError()
            var_g  = footprint['features'][:,idx_g]
            var_r  = footprint['features'][:,idx_r]
            var_z  = footprint['features'][:,idx_z]
            var_w1 = footprint['features'][:,8]
            var_w2 = footprint['features'][:,9]
            if ebv or star:
                var_x = footprint['features'][:,idx_x]
            if ebv or star:
                bands = ["x"]
            elif psfsize:
                bands = ["g","r","z"]
            else:
                bands =  ["g","r","z","w1"]
            self.bands = bands
            for band in bands:
                nbins = []
                bins = np.linspace(lim_low[band],lim_high[band],9)
                #bins = np.array(mehdi_bins[eval("idx_%s"%band)*9:(eval("idx_%s"%band)+1)*9])
                if j==1:
                    print(bins)
                for i in range(8):
                    sel_lrg = (eval('var_%s'%band)>bins[i])&(eval('var_%s'%band)<bins[i+1])
                    n_mocklrg = mock_i[sel_lrg].sum()
                    n_random = sel_lrg.sum()
                    ave = mock_i.sum()/len(mock_i)
                    nbins.append(np.array(n_mocklrg/n_random/ave))
                bin_center = (bins[1:]+bins[:-1])/2.
                np.savetxt(self.savedir+self.save_fn%(j,band), (bin_center,nbins))  



    
    def run_cov(self):
        for band in self.bands:
            dat_all = []
            std = np.zeros(8)
            for i in range(1,1000):
                dat = np.loadtxt(self.fn_sys%(i,band))[1]
                dat_all.append(dat)
            mean = np.zeros(8)
            for dat_all_i in dat_all:
                mean += dat_all_i

            mean = mean/len(dat_all)
            for dat_all_i in dat_all:
                std += (dat_all_i - mean)**2
            std = np.sqrt(std/999*(999/998))
            np.savetxt(self.savedir+self.cov_fn%band,std)
            print("written "+self.savedir+self.cov_fn%band)
        self.clean()
        
    def clean(self):
        fns = glob.glob(self.savedir+self.name+'*')
        for fn in fns:
            subprocess.call(["rm",fn])
        
    #total number of sources in the footprint
    @property
    def N_tot(self):
        return self._N_tot
    @N_tot.setter
    def N_tot(self,value):
        self._N_tot = value
     
    #total number of pixels in the same footprint
    @property
    def pix_tot(self):
        return self._pix_tot
    @pix_tot.setter
    def pix_tot(self,value):
        self._pix_tot = value
        
    #seed used in this set
    @property
    def seed(self):
        return self._seed
    @seed.setter
    def seed(self,value):
        self._seed = value
        
    
if __name__ == "__main__":
    """
    mocks = mock_cov(name = 'mocks', region = 'ndecals')
    mocks.run_sys(mode = 'mock')
    mocks.run_cov()
    
    randoms_lrg = mock_cov(name = 'rlrg', region = 'ndecals')
    randoms_lrg.N_tot = randoms_lrg.N_mock
    randoms_lrg.pix_tot = randoms_lrg.pix_tot_ndecals
    randoms_lrg.seed = 4321
    randoms_lrg.run_sys(mode = 'random')
    randoms_lrg.run_cov()
    
    randoms_lrgsv3 = mock_cov(name = 'rlrgsv3', region = 'ndecals')
    randoms_lrgsv3.N_tot = randoms_lrg.N_lrgsv3
    randoms_lrgsv3.pix_tot = randoms_lrg.pix_tot_ndecals
    randoms_lrgsv3.seed = 1234
    randoms_lrgsv3.run_sys(mode = 'random')
    randoms_lrgsv3.run_cov()
    """

    """
    randoms_lrgsv3 = mock_cov(name = 'robiwanlrgsv3', region = 'ndecals')
    randoms_lrgsv3.N_tot = randoms_lrgsv3.obiwan_lrg_ngc
    randoms_lrgsv3.pix_tot = randoms_lrgsv3.pix_tot_ndecals
    randoms_lrgsv3.seed = 1234
    randoms_lrgsv3.run_sys(mode = 'random')
    randoms_lrgsv3.run_cov()
    """

    """
    randoms_lrgsv3 = mock_cov(name = 'rlrgpsfsize', region = 'ndecals')
    randoms_lrgsv3.N_tot = randoms_lrgsv3.obiwan_lrg_ngc
    randoms_lrgsv3.pix_tot = randoms_lrgsv3.pix_tot_ndecals
    randoms_lrgsv3.seed = 5678
    randoms_lrgsv3.run_sys(mode = 'mock',psfsize=True)
    randoms_lrgsv3.run_cov()
    """

    """
    randoms_lrgsv3 = mock_cov(name = 'rpsfsize', region = 'ndecals')
    randoms_lrgsv3.N_tot = randoms_lrgsv3.obiwan_lrg_ngc
    randoms_lrgsv3.pix_tot = randoms_lrgsv3.pix_tot_ndecals
    randoms_lrgsv3.seed = 5678
    randoms_lrgsv3.run_sys(mode = 'random',psfsize=True)
    randoms_lrgsv3.run_cov()
    """
    
    
    randoms_lrgsv3 = mock_cov(name = 'rlrgebv', region = 'ndecals')
    randoms_lrgsv3.N_tot = randoms_lrgsv3.obiwan_lrg_ngc
    randoms_lrgsv3.pix_tot = randoms_lrgsv3.pix_tot_ndecals
    randoms_lrgsv3.seed = 5679
    randoms_lrgsv3.run_sys(mode = 'mock',psfsize=False, ebv = True)
    randoms_lrgsv3.run_cov()

    randoms_lrgsv3 = mock_cov(name = 'rlrgstar', region = 'ndecals')
    randoms_lrgsv3.N_tot = randoms_lrgsv3.obiwan_lrg_ngc
    randoms_lrgsv3.pix_tot = randoms_lrgsv3.pix_tot_ndecals
    randoms_lrgsv3.seed = 5679
    randoms_lrgsv3.run_sys(mode = 'mock',psfsize=False, star = True)
    randoms_lrgsv3.run_cov()
    

    randoms_lrgsv3 = mock_cov(name = 'rebv', region = 'ndecals')
    randoms_lrgsv3.N_tot = randoms_lrgsv3.obiwan_lrg_ngc
    randoms_lrgsv3.pix_tot = randoms_lrgsv3.pix_tot_ndecals
    randoms_lrgsv3.seed = 42
    randoms_lrgsv3.run_sys(mode = 'random',psfsize=False, ebv = True)
    randoms_lrgsv3.run_cov()

    randoms_lrgsv3 = mock_cov(name = 'rstar', region = 'ndecals')
    randoms_lrgsv3.N_tot = randoms_lrgsv3.obiwan_lrg_ngc
    randoms_lrgsv3.pix_tot = randoms_lrgsv3.pix_tot_ndecals
    randoms_lrgsv3.seed = 4242
    randoms_lrgsv3.run_sys(mode = 'random',psfsize=False, star = True)
    randoms_lrgsv3.run_cov()

 
