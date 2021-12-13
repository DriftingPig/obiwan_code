import glob
import astropy.io.fits as fits
import numpy as np
import sys
import os
from astropy.table import Table
sys.path.append('/global/cscratch1/sd/huikong/Obiwan/desihub/desitarget/py/')
sys.path.append('/global/cscratch1/sd/huikong/Obiwan/desihub/desiutil/py/')
sys.path.append('/global/cscratch1/sd/huikong/Obiwan/desihub/desimodel/py/')
from desitarget import cuts
from desitarget.sv1 import sv1_cuts
from desitarget.sv3 import sv3_cuts
import subprocess
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table,vstack
keys = ['BRICKNAME','RA','DEC','TYPE','OBJID','EBV','FLUX_G','FLUX_R','FLUX_Z','FLUX_W1','FLUX_W2','FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','MW_TRANSMISSION_W1','MW_TRANSMISSION_W2','NOBS_G','NOBS_R','NOBS_Z','NOBS_W1','NOBS_W2','SHAPE_R','SHAPE_E1','SHAPE_E2','FIBERFLUX_G','FIBERFLUX_R','FIBERFLUX_Z','MASKBITS','SERSIC','DCHISQ','PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z','GALDEPTH_G','GALDEPTH_R','GALDEPTH_Z','WISEMASK_W1','WISEMASK_W2','ANYMASK_G','ANYMASK_R','ANYMASK_Z','BX','BY','GAIA_PHOT_G_MEAN_MAG','FIBERTOTFLUX_Z']

topdir = '/global/cscratch1/sd/dstn/dr9-cosmos-subs/'
savedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/'
sets = glob.glob(topdir+'*')
sets.sort()



def get_snr(dat_i):
    dchisq = np.zeros(len(dat_i))
    sel_psf = (dat_i['TYPE']=='PSF')
    dchisq[sel_psf] = dat_i['DCHISQ'][:,0][sel_psf]
    sel_rex = (dat_i['TYPE']=='REX')
    dchisq[sel_rex] = dat_i['DCHISQ'][:,1][sel_rex]
    sel_dev = (dat_i['TYPE']=='DEV')
    dchisq[sel_dev] = dat_i['DCHISQ'][:,2][sel_dev]
    sel_exp = (dat_i['TYPE']=='EXP')
    dchisq[sel_exp] = dat_i['DCHISQ'][:,3][sel_exp]
    sel_ser = (dat_i['TYPE']=='SER')
    dchisq[sel_ser] = dat_i['DCHISQ'][:,4][sel_ser]
    sel_dup = (dat_i['TYPE']=='DUP') #keep gaia source,set snr to 1 so they don't get rejected
    dchisq[sel_dup] = np.ones_like(dchisq[sel_dup])
    gflux = dat_i['FLUX_G']/dat_i['MW_TRANSMISSION_G']
    rflux = dat_i['FLUX_R']/dat_i['MW_TRANSMISSION_R']
    zflux = dat_i['FLUX_Z']/dat_i['MW_TRANSMISSION_Z']
    #gsnr = np.array((dchisq>0)&(gflux>0),dtype=np.int)
    #rsnr = np.array((dchisq>0)&(rflux>0),dtype=np.int)
    #zsnr = np.array((dchisq>0)&(zflux>0),dtype=np.int)
    gsnr = np.array((gflux>0),dtype=np.int)
    rsnr = np.array((rflux>0),dtype=np.int)
    zsnr = np.array((zflux>0),dtype=np.int)
    return gsnr,rsnr,zsnr

def faint_cut(dat_i):
    gflux = dat_i['FLUX_G']/dat_i['MW_TRANSMISSION_G']
    rflux = dat_i['FLUX_R']/dat_i['MW_TRANSMISSION_R']
    zflux = dat_i['FLUX_Z']/dat_i['MW_TRANSMISSION_Z']
    w1flux = dat_i['FLUX_W1']/dat_i['MW_TRANSMISSION_W1']
    
    gmag= 22.5 - 2.5 * np.log10(gflux)
    rmag= 22.5 - 2.5 * np.log10(rflux)
    zmag = 22.5 - 2.5 * np.log10(zflux)
    w1mag = 22.5 - 2.5 * np.log10(w1flux)
    
    shape_r = dat_i['shape_r']
    
    sel = (gmag>18)&(rmag>18)&(zmag>18)&(w1mag>18)&(shape_r<6)
    
    return sel

def get_LRG_sv3_sel(dat_i,south=True,source_type='LRG'):
    gflux = dat_i['FLUX_G']/dat_i['MW_TRANSMISSION_G']
    rflux = dat_i['FLUX_R']/dat_i['MW_TRANSMISSION_R']
    zflux = dat_i['FLUX_Z']/dat_i['MW_TRANSMISSION_Z']
    w1flux = dat_i['FLUX_W1']/dat_i['MW_TRANSMISSION_W1']
    w2flux = dat_i['FLUX_W2']/dat_i['MW_TRANSMISSION_W2']
    gnobs = dat_i['NOBS_G']
    rnobs = dat_i['NOBS_R']
    znobs = dat_i['NOBS_Z']
    maskbits = dat_i['MASKBITS']
    zfiberflux=dat_i['FIBERFLUX_Z']/dat_i['MW_TRANSMISSION_Z']
    rfluxivar=dat_i['FLUX_IVAR_R']
    zfluxivar=dat_i['FLUX_IVAR_Z']
    w1fluxivar=dat_i['FLUX_IVAR_W1']
    gaiagmag=dat_i['GAIA_PHOT_G_MEAN_MAG']
    zfibertotflux = dat_i['FIBERTOTFLUX_Z']/dat_i['MW_TRANSMISSION_Z']
    lrg, lrg_lowdens = sv3_cuts.isLRG(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
            zfiberflux=zfiberflux, rfluxivar=rfluxivar, zfluxivar=zfluxivar, w1fluxivar=w1fluxivar,
            gaiagmag=gaiagmag, gnobs=gnobs, rnobs=rnobs, znobs=znobs, maskbits=maskbits,
            zfibertotflux=zfibertotflux, primary=None, south=True)
    return lrg, lrg_lowdens
def get_LRG_sv3_like(dat_i,south=True,source_type='LRG'):
    import sys
    sys.path.append('./cosmos_preprocess')
    from dr9_tracer_like import isLRG_like
    cut_method = isLRG_like('LRG_sv3_like')
    lrg = cut_method.isLRGlike_color(dat_i)
    return lrg

def get_ELG_sv3_sel(dat_i,south=True):
    gflux = dat_i['FLUX_G']/dat_i['MW_TRANSMISSION_G']
    rflux = dat_i['FLUX_R']/dat_i['MW_TRANSMISSION_R']
    zflux = dat_i['FLUX_Z']/dat_i['MW_TRANSMISSION_Z']
    w1flux = dat_i['FLUX_W1']/dat_i['MW_TRANSMISSION_W1']
    w2flux = dat_i['FLUX_W2']/dat_i['MW_TRANSMISSION_W2']
    gfiberflux=dat_i['FIBERFLUX_G']/dat_i['MW_TRANSMISSION_G']
    gsnr = dat_i['FLUX_IVAR_G']
    rsnr = dat_i['FLUX_IVAR_R']
    zsnr = dat_i['FLUX_IVAR_Z']
    gnobs = dat_i['NOBS_G']
    rnobs = dat_i['NOBS_R']
    znobs = dat_i['NOBS_Z']
    maskbits = dat_i['MASKBITS']
    L = locals()
    L.pop('dat_i')
    kwargs = dict([(k,L[k]) for k in L.keys()])
    elg, elg_hip = sv3_cuts.isELG(primary = None, **kwargs)
    return elg, elg_hip

def get_LRG_sel(dat_i,south=True,source_type='LRG'):
    assert(source_type in ['LRG','LRG_OPT','LRG_IR','SV_LRG_OPT','SV_LRG_IR','LRG_OPT_EXT','LRG_IR_EXT'])
    gflux = dat_i['FLUX_G']/dat_i['MW_TRANSMISSION_G']
    rflux = dat_i['FLUX_R']/dat_i['MW_TRANSMISSION_R']
    zflux = dat_i['FLUX_Z']/dat_i['MW_TRANSMISSION_Z']
    w1flux = dat_i['FLUX_W1']/dat_i['MW_TRANSMISSION_W1']
    w2flux = dat_i['FLUX_W2']/dat_i['MW_TRANSMISSION_W2']
    gnobs = dat_i['NOBS_G']
    rnobs = dat_i['NOBS_R']
    znobs = dat_i['NOBS_Z']
    maskbits = dat_i['MASKBITS']
    zfiberflux=dat_i['FIBERFLUX_Z']
    rfluxivar=dat_i['FLUX_IVAR_R']
    zfluxivar=dat_i['FLUX_IVAR_Z']
    w1fluxivar=dat_i['FLUX_IVAR_W1']
    if source_type == 'LRG':
        source_sel = cuts.isLRG(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                zfiberflux=zfiberflux, rfluxivar=rfluxivar, zfluxivar=zfluxivar, w1fluxivar=w1fluxivar,
                gnobs=gnobs, rnobs=rnobs, znobs=znobs, maskbits=maskbits, primary=None,
                south=south)
    else:
        lrg_opt, lrg_ir, lrg_sv_opt, lrg_sv_ir = sv1_cuts.isLRG(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                zfiberflux=zfiberflux, rfluxivar=rfluxivar, zfluxivar=zfluxivar, w1fluxivar=w1fluxivar,
                gnobs=gnobs, rnobs=rnobs, znobs=znobs, maskbits=maskbits, primary=None,
                south=south) 
    if source_type == 'LRG_OPT':
        source_sel = lrg_opt
    elif source_type == 'LRG_IR':
        source_sel = lrg_ir
    elif source_type == 'SV_LRG_OPT':
        source_sel = lrg_sv_opt
    elif source_type == 'SV_LRG_IR':
        source_sel = lrg_sv_ir
    elif source_type == 'LRG_OPT_EXT':
        source_sel = lrg_opt
    elif source_type == 'LRG_IR_EXT':
        source_sel = lrg_ir
    return source_sel

def get_ELG_sel(dat_i,south=True,source_type='elg'):
    assert(source_type in ['elg','elg_like','svgtot','svgfib','fdrgtot','fdrgfib'])
    gflux = dat_i['FLUX_G']/dat_i['MW_TRANSMISSION_G']
    rflux = dat_i['FLUX_R']/dat_i['MW_TRANSMISSION_R']
    zflux = dat_i['FLUX_Z']/dat_i['MW_TRANSMISSION_Z']
    w1flux = dat_i['FLUX_W1']/dat_i['MW_TRANSMISSION_W1']
    w2flux = dat_i['FLUX_W2']/dat_i['MW_TRANSMISSION_W2']
    gnobs = dat_i['NOBS_G']
    rnobs = dat_i['NOBS_R']
    znobs = dat_i['NOBS_Z']
    maskbits = dat_i['MASKBITS']
    zfiberflux=dat_i['FIBERFLUX_Z']
    rfluxivar=dat_i['FLUX_IVAR_R']
    zfluxivar=dat_i['FLUX_IVAR_Z']
    w1fluxivar=dat_i['FLUX_IVAR_W1']
    gsnr,rsnr,zsnr = get_snr(dat_i)
    gfiberflux = dat_i['FIBERFLUX_G']
    
    if  source_type=='elg':
                source_sel = cuts.isELG(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                        gsnr=gsnr,rsnr=rsnr,zsnr=zsnr,
                        gnobs=gnobs, rnobs=rnobs, znobs=znobs, maskbits=maskbits, primary=None,
                        south=south)
    elif source_type=='elg_like':
                source_sel = cuts.isELG_like(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                        gsnr=gsnr,rsnr=rsnr,zsnr=zsnr,
                        gnobs=gnobs, rnobs=rnobs, znobs=znobs, maskbits=maskbits, primary=None,
                        south=south)
    else:
        svgtot, svgfib, fdrgtot, fdrgfib = sv1_cuts.isELG(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                        gfiberflux = gfiberflux, gsnr=gsnr,rsnr=rsnr,zsnr=zsnr,
                        gnobs=gnobs, rnobs=rnobs, znobs=znobs, maskbits=maskbits, primary=None,
                        south=south)
    if source_type == 'svgtot':
        source_sel = svgtot
    elif source_type == 'svgfib':
        source_sel = svgfib
    elif source_type == 'fdrgtot':
        source_sel = fdrgtot
    elif source_type == 'fdrgfib':
        source_sel = fdrgfib
    return source_sel
    
     

def main(target = 'ELG_like'):
    i=0
    for one_set in sets:
        print(i)
        stacked_dat = None
        fns = glob.glob(one_set+'/tractor/*/tractor-*')
        for fn in fns:
            dat = fits.getdata(fn)
            if target == 'LRG':
                sel = get_LRG_sel(dat)
            if target == 'ELG':
                sel = get_ELG_sel(dat,source_type='elg')
            if target == 'ELG_like':
                sel = get_ELG_sel(dat,source_type='elg_like')
            if stacked_dat is None:
                    stacked_dat = dat[sel]
            else:
                    stacked_dat = np.hstack((stacked_dat,dat[sel]))
        columns = fits.ColDefs(stacked_dat)
        hdu = fits.BinTableHDU.from_columns(columns)
        hdu.writeto(savedir+"cosmos_%s_set%d.fits"%(target,i),overwrite=True)
        i+=1
    

def main_all_one(input_brickname):
    bricknames = [input_brickname]
    #collect all targets 
    savedir_all = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cosmos_all_sv3/'
    maskbits_dir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/maskbits/'
    
    
    for brickname in bricknames:
        dat = []
        mask_fn = maskbits_dir+'%s.fits'%brickname
        mask = fits.getdata(mask_fn)
        for one_set in sets:
            fn = os.path.join(one_set,'tractor',brickname[:3],'tractor-'+brickname+'.fits')
            dat.append(fits.getdata(fn))
        for i in range(len(sets)):
            print("%s %d"%(brickname,i))
            T=Table()
            bx_i = (dat[i]['BX']+0.5).astype(int)
            by_i = (dat[i]['BY']+0.5).astype(int)
            mask_flag_i = mask[(by_i),(bx_i)]
            sel_i = (mask_flag_i==1)
            for key in keys:
                T[key] = dat[i][key][sel_i]
            sel_elg_i = get_ELG_sel(dat[i][sel_i],source_type='elg')
            sel_lrg_i = get_LRG_sel(dat[i][sel_i])
            
            #LRG_OPT
            LRG_OPT = get_LRG_sel(dat[i][sel_i],source_type = 'LRG_OPT')
            #LRG_IR
            LRG_IR = get_LRG_sel(dat[i][sel_i],source_type = 'LRG_IR')
            #SV_LRG_OPT
            SV_LRG_OPT = get_LRG_sel(dat[i][sel_i],source_type = 'SV_LRG_OPT')
            #SV_LRG_IR
            SV_LRG_IR = get_LRG_sel(dat[i][sel_i],source_type = 'SV_LRG_IR')
            #LRG_OPT_EXT
            LRG_OPT_EXT = get_LRG_sel(dat[i][sel_i],source_type = 'LRG_OPT_EXT')
            #LRG_IR_EXT
            LRG_IR_EXT = get_LRG_sel(dat[i][sel_i],source_type = 'LRG_IR_EXT')
            #svgtot
            svgtot = get_ELG_sel(dat[i][sel_i],source_type = 'svgtot')
            #svgfib
            svgfib = get_ELG_sel(dat[i][sel_i],source_type = 'svgfib')
            #fdrgtot
            fdrgtot = get_ELG_sel(dat[i][sel_i],source_type = 'fdrgtot')
            #fdrgfib
            fdrgfib = get_ELG_sel(dat[i][sel_i],source_type = 'fdrgfib')
            #lrg sv3
            sv3_lrg, sv3_lrg_lowdens = get_LRG_sv3_sel(dat[i][sel_i])
            #elg sv3
            sv3_elg, sv3_elg_hip = get_ELG_sv3_sel(dat[i][sel_i])
            #faint_cut
            isfaint = faint_cut(dat[i][sel_i])
            #LRG_sv3 like
            lrg_sv3_like = get_LRG_sv3_like(dat[i][sel_i])
            
            T['set_%d_iselg'%i] = sel_elg_i
            T['set_%d_islrg'%i] = sel_lrg_i
            T['set_%d_lrgopt'%i] = LRG_OPT
            T['set_%d_lrgir'%i] = LRG_IR
            T['set_%d_svlrgopt'%i] = SV_LRG_OPT
            T['set_%d_svlrgir'%i] = SV_LRG_IR
            T['set_%d_lrgoptext'%i] = LRG_OPT_EXT
            T['set_%d_lrgirext'%i] = LRG_IR_EXT
            T['set_%d_elgsvgtot'%i] = svgtot
            T['set_%d_elgsvgfib'%i] = svgfib
            T['set_%d_elgfdrgtot'%i] = fdrgtot
            T['set_%d_elgfdrgfib'%i] = fdrgfib
            T['set_%d_lrg_sv3'%i] = sv3_lrg
            T['set_%d_lrg_sv3_lowdens'%i] = sv3_lrg_lowdens
            T['set_%d_elg_sv3'%i] = sv3_elg
            T['set_%d_elg_sv3_hip'%i] = sv3_elg_hip
            T['set_%d_isfaint'%i] = isfaint
            T['lrg_sv3_like'] = lrg_sv3_like
            
            for j in range(len(sets)):
                if i==j:
                    continue
                bx_j = (dat[j]['BX']+0.5).astype(int)
                by_j = (dat[j]['BY']+0.5).astype(int)
                mask_flag_j = mask[(by_j),(bx_j)]
                sel_j = (mask_flag_j==1)
                c1 = SkyCoord(ra=dat[i]['ra'][sel_i]*u.degree, dec=dat[i]['dec'][sel_i]*u.degree)
                c2 = SkyCoord(ra=dat[j]['ra'][sel_j]*u.degree, dec=dat[j]['dec'][sel_j]*u.degree)
                idx1, d2d, d3d = c1.match_to_catalog_sky(c2)
                sel_all = (np.array(d2d)<1.5/3600)
                #elg
                sel_elg_j = get_ELG_sel(dat[j][sel_j],source_type='elg')
                sel_elg_j_i = sel_elg_j[idx1]
                #lrg
                sel_lrg_j = get_LRG_sel(dat[j][sel_j],source_type = 'LRG')
                sel_lrg_j_i = sel_lrg_j[idx1]
                #LRG_OPT
                LRG_OPT = get_LRG_sel(dat[j][sel_j],source_type = 'LRG_OPT')[idx1]
                #LRG_IR
                LRG_IR = get_LRG_sel(dat[j][sel_j],source_type = 'LRG_IR')[idx1]
                #SV_LRG_OPT
                SV_LRG_OPT = get_LRG_sel(dat[j][sel_j],source_type = 'SV_LRG_OPT')[idx1]
                #SV_LRG_IR
                SV_LRG_IR = get_LRG_sel(dat[j][sel_j],source_type = 'SV_LRG_IR')[idx1]
                #LRG_OPT_EXT
                LRG_OPT_EXT = get_LRG_sel(dat[j][sel_j],source_type = 'LRG_OPT_EXT')[idx1]
                #LRG_IR_EXT
                LRG_IR_EXT = get_LRG_sel(dat[j][sel_j],source_type = 'LRG_IR_EXT')[idx1]
                #svgtot
                svgtot = get_ELG_sel(dat[j][sel_j],source_type = 'svgtot')[idx1]
                #svgfib
                svgfib = get_ELG_sel(dat[j][sel_j],source_type = 'svgfib')[idx1]
                #fdrgtot
                fdrgtot = get_ELG_sel(dat[j][sel_j],source_type = 'fdrgtot')[idx1]
                #fdrgfib
                fdrgfib = get_ELG_sel(dat[j][sel_j],source_type = 'fdrgfib')[idx1]
                #lrg sv3
                sv3_lrg, sv3_lrg_lowdens = get_LRG_sv3_sel(dat[j][sel_j])
                sv3_lrg = sv3_lrg[idx1]
                sv3_lrg_lowdens = sv3_lrg_lowdens[idx1]
                #elg sv3
                sv3_elg, sv3_elg_hip = get_ELG_sv3_sel(dat[j][sel_j])
                sv3_elg = sv3_elg[idx1]
                sv3_elg_hip = sv3_elg_hip[idx1]
                #isfaint
                isfaint = faint_cut(dat[j][sel_j])[idx1]
                
                T['set_%d_iselg'%j] = sel_elg_j_i
                T['set_%d_islrg'%j] = sel_lrg_j_i
                T['set_%d_matched'%j]=sel_all
                T['set_%d_flux_g'%j]=dat[j][sel_j]['flux_g'][idx1]
                T['set_%d_flux_r'%j]=dat[j][sel_j]['flux_r'][idx1]
                T['set_%d_flux_z'%j]=dat[j][sel_j]['flux_z'][idx1]
                T['set_%d_flux_w1'%j]=dat[j][sel_j]['flux_w1'][idx1]
                T['set_%d_flux_w2'%j]=dat[j][sel_j]['flux_w2'][idx1]
                T['set_%d_fiberflux_z'%j]=dat[j][sel_j]['fiberflux_z'][idx1]
                T['set_%d_fiberflux_g'%j]=dat[j][sel_j]['fiberflux_g'][idx1]
                T['set_%d_lrgopt'%j] = LRG_OPT
                T['set_%d_lrgir'%j] = LRG_IR
                T['set_%d_svlrgopt'%j] = SV_LRG_OPT
                T['set_%d_svlrgir'%j] = SV_LRG_IR
                T['set_%d_lrgoptext'%j] = LRG_OPT_EXT
                T['set_%d_lrgirext'%j] = LRG_IR_EXT
                T['set_%d_elgsvgtot'%j] = svgtot
                T['set_%d_elgsvgfib'%j] = svgfib
                T['set_%d_elgfdrgtot'%j] = fdrgtot
                T['set_%d_elgfdrgfib'%j] = fdrgfib
                T['set_%d_lrg_sv3'%j] = sv3_lrg
                T['set_%d_lrg_sv3_lowdens'%j] = sv3_lrg_lowdens
                T['set_%d_elg_sv3'%j] = sv3_elg
                T['set_%d_elg_sv3_hip'%j] = sv3_elg_hip
                T['set_%d_isfaint'%j] = isfaint
            basename = os.path.basename(sets[i])
            if not os.path.isdir(savedir_all+basename):
                subprocess.call(["mkdir","-p",savedir_all+basename])
            T.write(savedir_all+basename+'/%s.fits'%brickname,overwrite=True)

            
def main_all():
    bricknames = ['1498p017',  '1501p017',  '1503p017',  '1506p017', '1498p020',  '1501p020', '1503p020',  '1506p020', '1498p022', '1501p022',  '1503p022',  '1506p022', '1498p025',  '1501p025', '1503p025', '1506p025']
    import multiprocessing as mp
    p = mp.Pool(16)
    p.map(main_all_one, bricknames)

def stack_main_all():
    #stack all bricks altogether
    savedir_all = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cosmos_all_sv3/'
    if not os.path.isdir(savedir+'cosmos_all_stacked'):
        subprocess.call(["mkdir",savedir+'cosmos_all_stacked'])
    fns = glob.glob(savedir_all+'*')
    fns.sort()

    i=0
    for fn in fns:
        dat = Table()
        brick_dats = glob.glob(fn+'/*')
        flag = True
        for brick_dat in brick_dats:
            if flag is True:
                dat = Table.read(brick_dat)
                flag = False
            else:
                dat_i = Table.read(brick_dat)
                dat = vstack((dat,dat_i))
        dat.write(savedir+'cosmos_all_stacked/'+"cosmos_set%d.fits"%i,overwrite=True)
        i+=1

main_all()
stack_main_all()
