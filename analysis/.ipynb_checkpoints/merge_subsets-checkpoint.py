from astrometry.util.fits import fits_table, merge_tables
import glob
from astropy.table import Table
import numpy as np
import fitsio
#shifter --module=mpich-cle6 --image=legacysurvey/legacypipe:DR9.7.1 /bin/bash

tot_sets = 30
subsection = 'more_rs0'
name_for_run = "decals_ngc"
fn = "/global/cfs/cdirs/desi/users/huikong/%s/subset/subset_%s_part%s_of%d.fits"%(name_for_run, subsection, "%d", tot_sets)
fn_pz = fn[:-5]+'-pz.fits'
fn_tpz = fn[:-5]+'-tpz.fits'
#match photo z's
for i in range(tot_sets):
    print(i)
    ori_tab = fits_table(fn%i)
    input_z = fits_table(fn_tpz%i)
    output_z = fits_table(fn_pz%i)
    assert(len(ori_tab)==len(input_z))
    assert(len(ori_tab)==len(output_z))
    
    ori_tab.pz_in_mean = -99
    ori_tab.pz_in_std = -99
    ori_tab.pz_out_mean = -99
    ori_tab.pz_out_std = -99
    #import pdb;pdb.set_trace()
    ori_tab.pz_in_mean = input_z.z_phot_mean
    ori_tab.pz_in_std = input_z.z_phot_std
    ori_tab.pz_in_L68 = input_z.z_phot_l68
    ori_tab.pz_in_U68 = input_z.z_phot_u68
    ori_tab.pz_out_mean = output_z.z_phot_mean
    ori_tab.pz_out_std = output_z.z_phot_std
    ori_tab.pz_out_L68 = output_z.z_phot_l68
    ori_tab.pz_out_U68 = output_z.z_phot_u68
    ori_tab.writeto(fn%i, overwrite = True)
    print("written "+fn%i)


ofn = '/global/cfs/cdirs/desi/users/huikong/%s/subset/subset_%s.fits'%(name_for_run,  subsection)
TT = []
for i in range(tot_sets):
        TT.append(fits_table(fn%i))
T = merge_tables(TT)
T.writeto(ofn, overwrite = True)
print("written %s"%ofn)



