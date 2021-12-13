import sys
fn1 = '/global/cscratch1/sd/huikong/Obiwan/desihub/desitarget/py/'   
fn2 = '/global/cscratch1/sd/huikong/Obiwan/desihub/desiutil/py/'
fn3 = '/global/cscratch1/sd/huikong/Obiwan/desihub/desimodel/py'
fn4 = './cosmos_preprocess'
old_path = sys.path
sys.path = [sys.path[0], fn1, fn2, fn3, fn4]
for path in old_path[1:]:
    sys.path.append(path)

#keys to be collected to generate sweep files
sweep_keys= ['BRICKNAME','RA','DEC','TYPE','OBJID','EBV','FLUX_G','FLUX_R','FLUX_Z','FLUX_W1','FLUX_W2','FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','MW_TRANSMISSION_W1','MW_TRANSMISSION_W2','NOBS_G','NOBS_R','NOBS_Z','NOBS_W1','NOBS_W2','SHAPE_R','SHAPE_E1','SHAPE_E2','FIBERFLUX_G','FIBERFLUX_R','FIBERFLUX_Z','MASKBITS','SERSIC','DCHISQ','PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z','GALDEPTH_G','GALDEPTH_R','GALDEPTH_Z','WISEMASK_W1','WISEMASK_W2','ANYMASK_G','ANYMASK_R','ANYMASK_Z','BX','BY']
