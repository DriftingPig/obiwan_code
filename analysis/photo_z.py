# Run on NERSC
# Compute the photo-z's for a single sweep file
# Simple quality cuts are applied, objects that do not meet the cuts are assigned value -99

from __future__ import division, print_function
import sys, os, warnings, gc, time, glob, argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from astropy.table import Table, hstack
import fitsio
import glob
import joblib
###############
test_run = False
###############

n_process = 30

# field = 'north'
# field = 'south'

n_estimators = 100  # Number of trees in a forest
n_perturb = 20  # Number of perturbed sample

pz_dtype = [('Z_PHOT_MEAN', 'f4'), ('Z_PHOT_MEDIAN', 'f4'), ('Z_PHOT_STD', 'f4'), 
            ('Z_PHOT_L68', 'f4'), ('Z_PHOT_U68', 'f4'), ('Z_PHOT_L95', 'f4'), ('Z_PHOT_U95', 'f4')]

####################################################################################################################


def unpack(cat):
    gmag = 22.5-2.5*np.log10(cat['FLUX_G_EC'])
    rmag = 22.5-2.5*np.log10(cat['FLUX_R_EC'])
    zmag = 22.5-2.5*np.log10(cat['FLUX_Z_EC'])
    w1mag = 22.5-2.5*np.log10(cat['FLUX_W1_EC'])
    w2mag = 22.5-2.5*np.log10(cat['FLUX_W2_EC'])
    radius1 = np.array(cat['SHAPE_R'])
    return gmag, rmag, zmag, w1mag, w2mag, radius1

# sweep_fn = 'sweep-260p015-270p020.fits'
def compute_photoz(field = None, sweep_fn = None, mode = 'output',**kwargs):

    np.random.seed(1456)
    print(mode)  
    #why is this not working?
    #keys = kwargs.keys()
    #for k in keys:
    #    exec(k+"=kwargs[%s%s%s]"%("\"",k,"\""))
    tree_dir = kwargs['tree_dir']
    specz_full_path = kwargs['specz_full_path']
    specz_train_path = kwargs['specz_train_path']
    specz_train = kwargs['specz_train']
    specz_full = kwargs['specz_full']
    regrf_all = kwargs['regrf_all']

    #print()
    print(sweep_fn)

    #output_path = os.path.join(output_dir, sweep_fn[:-5]+'-pz.fits')
    if mode == 'output':
        output_path = sweep_fn[:-5]+'-pz.fits'
    else:
        output_path = sweep_fn[:-5]+'-tpz.fits'
    if os.path.isfile(output_path):
        return None
    Path(output_path).touch(exist_ok=False)
     
    if mode == 'output':
        columns = ['RELEASE', 'BRICKID', 'OBJID', 'TYPE', 'RA', 'DEC', 'DCHISQ', 'EBV',
        'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2',
        'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FLUX_IVAR_W2',
        'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2',
        'NOBS_G', 'NOBS_R', 'NOBS_Z',
        'SHAPE_R', 'SHAPE_R_IVAR', 'SHAPE_E1', 'SHAPE_E2']
        columns = [i.lower() for i in columns]
        cat = Table(fitsio.read(sweep_fn, columns=columns))
    if mode == 'input':
        columns = ['RELEASE', 'BRICKID', 'OBJID', 'sim_sersic_n', 'RA', 'DEC', 'DCHISQ', 'EBV',
        'sim_gflux', 'sim_rflux', 'sim_zflux', 'sim_w1', 'sim_w2',
        'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FLUX_IVAR_W2',
        'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2',
        'NOBS_G', 'NOBS_R', 'NOBS_Z',
        'sim_rhalf', 'SHAPE_R_IVAR', 'sim_e1', 'sim_e2']
        columns = [i.lower() for i in columns]
        cat = Table(fitsio.read(sweep_fn, columns=columns))
        #cat = Table(fitsio.read(os.path.join(sweep_dir, sweep_fn), columns=columns))
        cat['flux_g'] = cat['sim_gflux']
        cat['flux_r'] = cat['sim_rflux']
        cat['flux_z'] = cat['sim_zflux']
        cat['flux_w1'] = cat['mw_transmission_w1']*(10**(0.4*(22.5-cat['sim_w1'])))
        cat['flux_w2'] = cat['mw_transmission_w2']*(10**(0.4*(22.5-cat['sim_w2'])))
        cat['shape_r'] = cat['sim_rhalf']
        cat['shape_e1'] = cat['sim_e1']
        cat['shape_e2'] = cat['sim_e2']
        cat['type'] = b = np.full(len(cat), "       ")
        cat['type'][cat['sim_sersic_n']==0] = 'PSF'
        cat['type'][cat['sim_sersic_n']==4] = 'DEV'
        cat['type'][(cat['sim_sersic_n']==1)&(cat['sim_e1']==0)&(cat['sim_e2']==0)] = 'REX'
        cat['type'][(cat['sim_sersic_n']==1)&((cat['sim_e1']!=0)|(cat['sim_e2']!=0))] = 'EXP'
        cat['type'][(cat['sim_sersic_n']!=0)&(cat['sim_sersic_n']!=1)&(cat['sim_sersic_n']!=4)] = 'SER'
        
                              
    
    #print(len(cat))

    if test_run:
        cat = cat[::1000]
        print(len(cat))

    cat_id_full = cat[['RELEASE'.lower(), 'BRICKID'.lower(), 'OBJID'.lower()]].copy()

    # Sanity check to make sure that there are not extra spaces in type names
    types = ['PSF', 'REX', 'EXP', 'DEV', 'SER', 'DUP']
    for tmp in np.unique(cat['TYPE'.lower()]):
        if tmp not in types:
            raise ValueError('Type {} not recognized'.format(tmp))

    full_size = len(cat)

    id_full = np.array(cat['OBJID'.lower()], dtype='int64') * int(1e10) \
              + np.array(cat['BRICKID'.lower()], dtype='int64') * int(1e4) \
              + np.array(cat['RELEASE'.lower()], dtype='int64')

    mask_bad = np.full(len(cat), False, dtype=bool)
    mask_bad |= ~((cat['NOBS_G'.lower()]>=1) & (cat['NOBS_R'.lower()]>=1) & (cat['NOBS_Z'.lower()]>=1))
    mask_bad |= ~((cat['FLUX_IVAR_G'.lower()]>0) & (cat['FLUX_IVAR_R'.lower()]>0) & (cat['FLUX_IVAR_Z'.lower()]>0))
    mask_bad |= (cat['TYPE'.lower()]=='DUP')

    print('{:} ({:.1f}%) objects removed'.format(np.sum(mask_bad), np.sum(mask_bad)/len(mask_bad)*100))

    if np.sum(~mask_bad)==0:
        cat_pz_full = Table(data=np.full(full_size, -99, dtype=pz_dtype))
        cat_pz_full['Z_SPEC'] = np.ones(len(cat_pz_full), dtype='float32') * (-99.)
        cat_pz_full['SURVEY'] = '          '
        cat_pz_full['TRAINING'] = False
        cat_pz_full = hstack([cat_id_full, cat_pz_full])
        cat_pz_full.write(output_path, overwrite=True)
        return None

    cat = cat[~mask_bad]
    print(len(cat))

    # Assign inf to invalid FLUX_IVAR_W1 and FLUX_IVAR_W2
    # (Not sure if it actually happens for TYPE!='DUP' objects though)
    mask = cat['FLUX_IVAR_W1'.lower()]==0
    cat['FLUX_IVAR_W1'.lower()][mask] = np.inf
    mask = cat['FLUX_IVAR_W2'.lower()]==0
    cat['FLUX_IVAR_W2'.lower()][mask] = np.inf

    # Assign inf to invalid zero SHAPE_R_IVAR
    mask = cat['SHAPE_R_IVAR'.lower()]==0
    cat['SHAPE_R_IVAR'.lower()][mask] = np.inf

    # axis ratio
    e = np.array(np.sqrt(cat['SHAPE_E1'.lower()]**2+cat['SHAPE_E2'.lower()]**2))
    q = (1+e)/(1-e)

    # shape probability (definition of shape probability in Soo et al. 2017)
    p = np.ones(len(cat))*0.5
    # DCHISQ[:, 2] is DCHISQ_EXP; DCHISQ[:, 3] is DCHISQ_DEV
    mask_chisq = (cat['DCHISQ'.lower()][:, 3]>0) & (cat['DCHISQ'.lower()][:, 2]>0)
    p[mask_chisq] = cat['DCHISQ'.lower()][:, 3][mask_chisq]/(cat['DCHISQ'.lower()][:, 3]+cat['DCHISQ'.lower()][:, 2])[mask_chisq]

    ####################################################################################################################

    cat_pz = Table(data=np.full(len(cat), -99, dtype=pz_dtype))

    # print('Computing photo-z\'s and photo-z errors')

    col_list = ['FLUX_G_EC', 'FLUX_R_EC', 'FLUX_Z_EC', 'FLUX_W1_EC', 'FLUX_W2_EC']
    mag_max = 30
    mag_fill = 100

    chunk_size = float(2e4)
    if len(cat)<=chunk_size:
        n_split = 1
    else:
        n_split = int(np.ceil(len(cat)/chunk_size))

    for chunk_index in range(n_split):

        # print('chunk {} / {}'.format(chunk_index+1, n_split))

        if n_split==1:
            idx_chunk = np.arange(len(cat))
        else:
            if chunk_index==0:
                idx_chunk = np.arange(int(chunk_size))
            elif chunk_index==(n_split-1):
                idx_chunk = np.arange(int(chunk_size)*(n_split-1), len(cat))
            else:
                idx_chunk = np.arange(int(chunk_size)*chunk_index, int(chunk_size)*(chunk_index+1))

        cat_chunk, q_chunk, p_chunk = cat[idx_chunk], q[idx_chunk], p[idx_chunk]

        dtype1 = [('FLUX_G_EC', 'f4'), ('FLUX_R_EC', 'f4'), ('FLUX_Z_EC', 'f4'), ('FLUX_W1_EC', 'f4'), ('FLUX_W2_EC', 'f4'), ('SHAPE_R', 'f4')]
        cat1 = Table(data=np.zeros(len(cat_chunk), dtype=dtype1))

        z_phot_array = np.zeros((len(cat1), n_perturb*n_estimators), dtype='float32')

        for tree_index in range(n_estimators):

            # print(tree_index*n_perturb, '/', n_estimators*n_perturb)

            # Predict!
            for perturb_index in range(n_perturb):

                cat1['FLUX_G_EC'] = np.array((cat_chunk['FLUX_G'.lower()]+np.random.randn(len(cat_chunk))/np.sqrt(cat_chunk['FLUX_IVAR_G'.lower()]))/cat_chunk['MW_TRANSMISSION_G'.lower()], dtype='float32')
                cat1['FLUX_R_EC'] = np.array((cat_chunk['FLUX_R'.lower()]+np.random.randn(len(cat_chunk))/np.sqrt(cat_chunk['FLUX_IVAR_R'.lower()]))/cat_chunk['MW_TRANSMISSION_R'.lower()], dtype='float32')
                cat1['FLUX_Z_EC'] = np.array((cat_chunk['FLUX_Z'.lower()]+np.random.randn(len(cat_chunk))/np.sqrt(cat_chunk['FLUX_IVAR_Z'.lower()]))/cat_chunk['MW_TRANSMISSION_Z'.lower()], dtype='float32')
                cat1['FLUX_W1_EC'] = np.array((cat_chunk['FLUX_W1'.lower()]+np.random.randn(len(cat_chunk))/np.sqrt(cat_chunk['FLUX_IVAR_W1'.lower()]))/cat_chunk['MW_TRANSMISSION_W1'.lower()], dtype='float32')
                cat1['FLUX_W2_EC'] = np.array((cat_chunk['FLUX_W2'.lower()]+np.random.randn(len(cat_chunk))/np.sqrt(cat_chunk['FLUX_IVAR_W2'.lower()]))/cat_chunk['MW_TRANSMISSION_W2'.lower()], dtype='float32')
                cat1['SHAPE_R'] = np.array(cat_chunk['SHAPE_R'.lower()]+np.random.randn(len(cat_chunk))/np.sqrt(cat_chunk['SHAPE_R_IVAR'.lower()]), dtype='float32')

                # Fill in negative fluxes
                for index in range(len(col_list)):
                    mask = (cat1[col_list[index]]<10**(0.4*(22.5-mag_max))) | (~np.isfinite(cat1[col_list[index]]))
                    cat1[col_list[index]][mask] = 10**(0.4*(22.5-mag_fill))

                gmag1, rmag1, zmag1, w1mag1, w2mag1, radius1 = unpack(cat1)
                data1 = np.column_stack((gmag1-rmag1, rmag1-zmag1, zmag1-w1mag1, w1mag1-w2mag1, rmag1, radius1, q_chunk, p_chunk))

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    z_phot_array[:, tree_index*n_perturb+perturb_index] = regrf_all[tree_index].predict(data1)

            # clear cache
            gc.collect()

        del cat1

        cat_pz['Z_PHOT_MEAN'][idx_chunk] = np.mean(z_phot_array, axis=1)
        cat_pz['Z_PHOT_STD'][idx_chunk] = np.std(z_phot_array, axis=1)
        # WARNING: z_phot_array becomes undefined after this
        cat_pz['Z_PHOT_L95'][idx_chunk], cat_pz['Z_PHOT_L68'][idx_chunk], cat_pz['Z_PHOT_MEDIAN'][idx_chunk], cat_pz['Z_PHOT_U68'][idx_chunk], cat_pz['Z_PHOT_U95'][idx_chunk] = \
            np.percentile(z_phot_array, [2.5, 16., 50., 84., 97.5], axis=1, overwrite_input=True)

    cat_pz_full = Table(data=np.full(full_size, -99, dtype=pz_dtype))
    cat_pz_full[~mask_bad] = cat_pz

    ######################## Match to spectroscopic truth table #############################

    print('Matching to truth table')

    cat_pz_full['Z_SPEC'] = np.ones(len(cat_pz_full), dtype='float32') * (-99.)
    cat_pz_full['SURVEY'] = '          '
    cat_pz_full['TRAINING'] = False

    _, idx1, idx2 = np.intersect1d(np.array(specz_full['id']), id_full, return_indices=True)
    cat_pz_full['Z_SPEC'][idx2] = specz_full['redshift'][idx1]
    cat_pz_full['SURVEY'][idx2] = specz_full['survey'][idx1]

    _, idx1, idx2 = np.intersect1d(np.array(specz_train['id']), id_full, return_indices=True)
    cat_pz_full['TRAINING'][idx2] = True

    ##########################################################################################

    # Add RELEASE, BRICKID and OBJID
    cat_pz_full = hstack([cat_id_full, cat_pz_full])
    cat_pz_full.write(output_path, overwrite=True)
    print("written %s"%output_path)
    return None

def photoz_main(input_fn, mode, field):
    tree_dir = '/global/cfs/cdirs/desi/users/rongpu/ls_dr9.0_photoz/individual_trees/20210130-'+field
    specz_full_path = '/global/cfs/cdirs/desi/users/rongpu/ls_dr9.0_photoz/truth/truth_combined_dr9.0_20210127_{}.fits'.format(field)
    specz_train_path = '/global/cfs/cdirs/desi/users/rongpu/ls_dr9.0_photoz/truth/truth_combined_dr9.0_20210127_{}_ds.fits'.format(field)
    specz_train = Table(fitsio.read(specz_train_path, columns=['id', 'redshift', 'survey']))
    specz_full = Table(fitsio.read(specz_full_path, columns=['id', 'redshift', 'survey']))
    regrf_all = []
    for tree_index in range(n_estimators):
        regrf_all.append(joblib.load(os.path.join(tree_dir, 'regrf_20210130_{:d}.pkl'.format(tree_index))))
    
    kwargs = {"tree_dir":tree_dir, "specz_full_path":specz_full_path, "specz_train_path":specz_train_path,"specz_train":specz_train,"specz_full":specz_full,"regrf_all":regrf_all}
    compute_photoz(field = field, sweep_fn = input_fn , mode = mode, **kwargs)

