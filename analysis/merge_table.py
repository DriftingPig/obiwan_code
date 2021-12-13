from astrometry.util.fits import fits_table, merge_tables
import glob
from astropy.table import Table
import numpy as np
#shifter --module=mpich-cle6 --image=legacysurvey/legacypipe:DR9.7.1 /bin/bash
def stack_sweep(chunk=4):
    # currently 2.3G in total
    outdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/LRG_SV3/sweep/"
    fns = glob.glob(outdir+'sweep*')
    fn_stack = np.array_split(fns, chunk)
    for i in range(chunk):
        print("stacking %d"%i)
        TT = []
        ofn = outdir+'chunk%d.fits'%i
        for fn in fn_stack[i]:
            TT.append(fits_table(fn))
        T = merge_tables(TT)
        T.writeto(ofn, overwrite = True)
        print("written %s"%ofn)

def stack_all(chunk4):
    outdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/LRG_SV3/sweep/"
    fns = glob.glob(outdir+'chunk*')
    TT=[]
    ofn = outdir+'LRG_SV3_south.fits'
    for fn in fns:
        TT.append(fits_table(fn))
    T = merge_tables(TT)
    T.writeto(ofn, overwrite = True)
    print("written %s"%ofn)

def get_desi():
    outdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/LRG_SV3/sweep/"
    fn = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/LRG_SV3/sweep/LRG_SV3_south.fits"
    T = fits_table(fn)
    sel = (T.dec>-18)
    T_new = T[sel]
    T_new.writeto(outdir+'LRG_SV3_south_desi.fits', overwrite=True)

def stack_maps():
    import fitsio
    topdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/maps_truth/"
    N = 80
    new_tab = Table.read("/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/0.49.0/resolve/combined/pixmap_south_nside_1024_minobs_1_maskbits_189111213.fits")
    L = len(new_tab)
    lrg_num = np.zeros(L)
    ran_num = np.zeros(L)
    rs0_num = np.zeros(L)
    obiwan_num = np.zeros(L)
    for i in range(N):
        print(i)
        fn = topdir+"maps1024_part%d_of80.fits"%i
        data_dr9 = fitsio.read(fn, columns=['lrg_num','ran_num','rs0_num','obiwan_num'])
        lrg_num += data_dr9['lrg_num']
        ran_num += data_dr9['ran_num']
        rs0_num += data_dr9['rs0_num']
        obiwan_num += data_dr9['obiwan_num']
    new_tab['lrg_num'] = lrg_num
    new_tab['ran_num'] = ran_num
    new_tab['rs0_num'] = rs0_num
    new_tab['obiwan_num'] = obiwan_num
    new_tab.write("/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/maps_truth/maps1024.fits")



#stack_sweep()
#stack_all(4)
#get_desi()
stack_maps()
