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
class catalg2hp(object):
    topdir_rongpu = "/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/0.49.0/resolve/combined/"
    rongpu1024    = topdir_rongpu+"pixmap_south_nside_1024_minobs_1_maskbits_189111213.fits"
    rongpu256     = topdir_rongpu+"pixmap_south_nside_256_minobs_1_maskbits_189111213.fits"
    savedir       = "/global/cfs/cdirs/desi/users/huikong/maps/v2/"   
    coeffs        = {'g':3.214, 'r':2.165, 'z':1.211, 'w1':0.184}
    lim_low       = {'g':24.2,'r':23.75,'z':22.7,'w1':21.4}
    lim_high      = {'g':25.00,'r':24.50,'z':23.7,'w1':21.8}
    def __init__(self,catalog_fn = None, card_name = None, map_fn = None, hpix_card = None, res = 256, nest = False):
        self.res = res
        self.nest = nest
        self.catalog_fn = catalog_fn
        self.card_name = card_name
        self.map_fn = map_fn
        self.hpix_card = hpix_card
    def run(self, selection = None, mp = None, threads = None):
        """
        add healpix density of a file to a map with card_name
        """
        input_map = Table.read(self.map_fn)
        if selection is None:
            input_catalog = fitsio.read(self.catalog_fn, columns = ['ra', 'dec'], ext = 1)
        else:
            input_catalog = fits.getdata(self.catalog_fn)[selection]
        self.hpix = hp.pixelfunc.ang2pix(self.res, input_catalog['ra'], input_catalog['dec'], nest=self.nest, lonlat=True)
        self.hpix.sort()
        self.map_hpix = input_map[self.hpix_card]
        ids = np.array_split(np.arange(len(self.hpix)),threads)
        number_list = mp.map(self.number_counts, ids)
        number_list_final = np.zeros(len(input_map))
        for number_list_i in number_list:
            number_list_final += number_list_i
        input_map[self.card_name] = number_list_final
        input_map.write(self.savedir+'map%d.fits'%self.res, overwrite=True)
        print("written %s"%(self.savedir+'map%d.fits'%self.res))
    def number_counts(self,id_list):
        number_counts = np.zeros(len(self.map_hpix))
        idx0 = id_list[0]
        moving_id = np.where(self.map_hpix >= self.hpix[idx0])[0].min()
        lost = 0
        print(moving_id)
  
        for iid in id_list:
            if iid%1000000 == 0:
                print(iid)
            hpix_i = self.hpix[iid]
            while(self.map_hpix[moving_id] < hpix_i):
                moving_id += 1
            if self.map_hpix[moving_id] == hpix_i:
                number_counts[moving_id] += 1
            else:
                lost += 1
        print("lost:%d/%d, %f"%(lost,len(self.hpix), lost/len(self.hpix)))
        return number_counts
        
        
if __name__ == "__main__":
    #notes: next = False for rongpu's map; nest = True for Ashley's map
    #adding obiwan_num, random_num, lrg_num per pixel for relavent healpix pixels
    
    threads = 1
    mp = multiproc(threads)
    """
    #rs0 all
    catalog_fn = "/global/cfs/cdirs/desi/users/huikong/decals_ngc/subset/subset_rs0.fits"
    card_name = "rs0_all_num"
    map_fn = catalg2hp().rongpu1024
    #map_fn = "/global/cfs/cdirs/desi/users/huikong/maps/v2/map256.fits"
    hpix_card = "HPXPIXEL"
    dat = fitsio.read(catalog_fn, columns = ['lrg_sv3', 'matched', 'maskbits','psfdepth_g','psfdepth_r','psfdepth_z','psfdepth_w1'], ext=1)
    selection = (dat['matched'])&(dat['maskbits']==0) & (dat['psfdepth_g']>0)& (dat['psfdepth_r']>0)& (dat['psfdepth_z']>0)& (dat['psfdepth_w1']>0)
    #selection = None
    catalgcls = catalg2hp(catalog_fn = catalog_fn, card_name = card_name, map_fn = map_fn, hpix_card = hpix_card, res=1024)
    catalgcls.run(selection = selection, mp = mp, threads = threads)
    
    #rs0 lrg
    map_fn = "/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits"
    card_name = "rs0_lrg_num"
    selection = (dat['lrg_sv3'])&(dat['matched'])&(dat['maskbits']==0) & (dat['psfdepth_g']>0)& (dat['psfdepth_r']>0)& (dat['psfdepth_z']>0)& (dat['psfdepth_w1']>0)
    catalgcls = catalg2hp(catalog_fn = catalog_fn, card_name = card_name, map_fn = map_fn, hpix_card = hpix_card, res=1024)
    catalgcls.run(selection = selection, mp = mp, threads = threads)
    del(dat)
    """
    #lrg sv3
    catalog_fn = "/global/cfs/cdirs/desi/users/huikong/decals_ngc/subset/subset_dr9_lrg_sv3.fits"
    card_name = "lrgsv3_num"
    map_fn = "/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits"
    hpix_card = "HPXPIXEL"
    dat = fitsio.read(catalog_fn, columns = ['maskbits','psfdepth_g','psfdepth_r','psfdepth_z','psfdepth_w1'], ext=1)
    selection = (dat['maskbits']==0) & (dat['psfdepth_g']>0)& (dat['psfdepth_r']>0)& (dat['psfdepth_z']>0)& (dat['psfdepth_w1']>0)
    catalgcls = catalg2hp(catalog_fn = catalog_fn, card_name = card_name, map_fn = map_fn, hpix_card = hpix_card, res=1024)
    catalgcls.run(selection = selection, mp = mp, threads = threads)
    del(dat)
    """
    #more_rs0 lrg
    catalog_fn = "/global/cfs/cdirs/desi/users/huikong/decals_ngc/subset/subset_more_rs0.fits"
    card_name = "more_rs0_lrg_num"
    map_fn = "/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits"
    hpix_card = "HPXPIXEL"
    dat = fitsio.read(catalog_fn, columns = ['lrg_sv3', 'matched', 'maskbits','psfdepth_g','psfdepth_r','psfdepth_z','psfdepth_w1'], ext=1)
    selection = (dat['lrg_sv3'])&(dat['matched'])&(dat['maskbits']==0) & (dat['psfdepth_g']>0)& (dat['psfdepth_r']>0)& (dat['psfdepth_z']>0)& (dat['psfdepth_w1']>0)
    catalgcls = catalg2hp(catalog_fn = catalog_fn, card_name = card_name, map_fn = map_fn, hpix_card = hpix_card, res=1024)
    catalgcls.run(selection = selection, mp = mp, threads = threads)
    
    #more_rs0 all
    catalog_fn = "/global/cfs/cdirs/desi/users/huikong/decals_ngc/subset/subset_more_rs0.fits"
    card_name = "more_rs0_all_num"
    map_fn = "/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits"
    hpix_card = "HPXPIXEL"
    dat = fitsio.read(catalog_fn, columns = ['matched', 'maskbits','psfdepth_g','psfdepth_r','psfdepth_z','psfdepth_w1'], ext=1)
    selection = (dat['matched'])&(dat['maskbits']==0) & (dat['psfdepth_g']>0)& (dat['psfdepth_r']>0)& (dat['psfdepth_z']>0)& (dat['psfdepth_w1']>0)
    catalgcls = catalg2hp(catalog_fn = catalog_fn, card_name = card_name, map_fn = map_fn, hpix_card = hpix_card, res=1024)
    catalgcls.run(selection = selection, mp = mp, threads = threads)
    
    #add rs0 and more_rs0
    input_map = Table.read(map_fn)
    input_map['lrg_num'] = input_map['more_rs0_lrg_num']+input_map['rs0_lrg_num']
    input_map['all_num'] = input_map['more_rs0_all_num']+input_map['rs0_all_num']
    input_map.write(map_fn, overwrite = True)
    """
    #randoms
    catalog_fn = "/global/cfs/cdirs/desi/users/huikong/decals_ngc/subset/subset_random.fits"
    card_name = "randoms_masked"
    map_fn = "/global/cfs/cdirs/desi/users/huikong/maps/v2/map1024.fits"
    hpix_card = "HPXPIXEL"
    dat = fitsio.read(catalog_fn, columns = ['maskbits','psfdepth_g','psfdepth_r','psfdepth_z','psfdepth_w1'], ext=1)
    selection = (dat['maskbits']==0) & (dat['psfdepth_g']>0)& (dat['psfdepth_r']>0)& (dat['psfdepth_z']>0)& (dat['psfdepth_w1']>0)
    catalgcls = catalg2hp(catalog_fn = catalog_fn, card_name = card_name, map_fn = map_fn, hpix_card = hpix_card, res=1024)
    catalgcls.run(selection = selection, mp = mp, threads = threads)
    
    
    