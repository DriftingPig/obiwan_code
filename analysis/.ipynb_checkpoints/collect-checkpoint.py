import os
import glob
import numpy as n
import numpy as np
from astropy.io import fits
from math import *
from astropy.table import Column, Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
import subprocess
from astropy.table import vstack,Table
from filesystem import LegacySimData
import subprocess
from SurveySource import BaseSource
import os

class Collect(LegacySimData):
    def __init__(self, survey_dir=None, outdir=None, subsection=None, brick=None, bricklist=None, region = None, **kwargs):
        super(Collect,self).__init__(survey_dir=survey_dir, outdir=outdir, subsection=subsection, brick=brick)
        self.region = region
    
    def brick_match(self, threads = None, bricklist = None, mp=None, subsection=None, startid=None, nobj=None, angle=1.5/3600, mode='sim',tracer=None, totslice = None, sliceidx = None, photo_z = False):
        inputs = []
        self.subsection = subsection
        bricklist_split = np.array_split(bricklist,threads)
        for i in range(threads):
            inputs.append((bricklist_split[i],startid, nobj, angle, mode, True,tracer))
        results = list(mp.map(self._brick_match_core, inputs))
        final_tab = None
        if len(results)>0:
            for tab in results:
                if tab is None:
                    continue
                if final_tab is not None:
                    final_tab = vstack((final_tab,tab))
                else:
                    final_tab = tab
            topdir = os.path.dirname(os.path.abspath(self.outdir))+'/subset'
            subprocess.call(["mkdir","-p",topdir])
            if totslice==1:
                output_fn = topdir+'/subset_%s.fits'%(self.subsection)
                final_tab.write(output_fn,overwrite=True)
            else:
                output_fn = topdir+'/subset_%s_part%d_of%d.fits'%(self.subsection,sliceidx,totslice)
                final_tab.write(output_fn,overwrite=True)
            print('written %s'%output_fn)
            if photo_z:
                from photo_z import photoz_main
                photoz_main(output_fn, 'input', region)
                photoz_main(output_fn, 'output', region)

    def collect_sweep(self, south = True,threads = None, mp=None,tracer=None):
        if south:
            region='south'
        else:
            region = 'north'
        dirs = "/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/%s/sweep/9.0/"%region
        fns = glob.glob(dirs+'sweep-*')
        inputs = []
        
        for fn in fns:
            brickname = os.path.basename(fn)
            inputs.append((south, brickname,tracer))
        mp.map(self._collect_sweep, inputs)

    def _collect_sweep(self, X):
        (south,brickname,tracer)=X
        assert(south)
        outdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/LRG_SV3/sweep/"
        topdir2 = "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0-photo-z/"
        if south:
            region='south'
        else:
            region = 'north'
        # collect tracers (now LRGs) from sweep files
        
        topdir = "/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/%s/sweep/9.0/"%region
        outdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/LRG_SV3/sweep/"
        catalog = BaseSource(filetype='sweep', survey_dir = topdir, outdir = topdir, subsection=None, brick=brickname)
        tracer_sel = catalog.target_selection(target=tracer,south=south)
        sweep_keys= ['BRICKNAME','RA','DEC','TYPE','OBJID','EBV','FLUX_G','FLUX_R','FLUX_Z','FLUX_W1','FLUX_W2','FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','MW_TRANSMISSION_W1','MW_TRANSMISSION_W2','NOBS_G','NOBS_R','NOBS_Z','NOBS_W1','NOBS_W2','SHAPE_R','SHAPE_E1','SHAPE_E2','FIBERFLUX_G','FIBERFLUX_R','FIBERFLUX_Z','MASKBITS','SERSIC','DCHISQ','PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z','GALDEPTH_G','GALDEPTH_R','GALDEPTH_Z','WISEMASK_W1','WISEMASK_W2','ANYMASK_G','ANYMASK_R','ANYMASK_Z','PSFDEPTH_W1','PSFDEPTH_W2']
        data = fits.getdata(topdir+brickname)
        photoz = fits.getdata(topdir2+brickname.replace('.fits','-pz.fits'))
        data_new = Table()
        for key in sweep_keys:
            data_new[key] = data[tracer_sel][key]
        data_new['Z_PHOT_MEAN'] = photoz[tracer_sel]['Z_PHOT_MEAN']
        data_new['Z_PHOT_STD'] = photoz[tracer_sel]['Z_PHOT_STD']
        data_new['Z_PHOT_L68'] = photoz[tracer_sel]['Z_PHOT_L68']
        data_new['Z_PHOT_U68'] = photoz[tracer_sel]['Z_PHOT_U68']
        data_new.write(outdir+brickname,overwrite=True)
        print("written %s"%brickname)

        

    def brick_match_dr9(self, bricklist, tracer = None,region='south',south=True, threads = None, mp=None, totslice = None, sliceidx = None):
        surveydir = "/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/%s/"%region
        outdir = "/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/%s/"%region
        final_tab = None
        bricklist_split = np.array_split(bricklist,threads)
        inputs = []
        for i in range(threads):
            inputs.append((bricklist_split[i],outdir,tracer,south))
        results = mp.map(self._brick_match_dr9, inputs)
        final_tab = None
        for result in results:
            if final_tab is None:
                final_tab = result
            else:
                final_tab = vstack((final_tab,result))
        topdir = os.path.dirname(os.path.abspath(self.outdir))+'/subset'
        final_tab.write(topdir+'/subset_dr9_%s_part%d_of%d.fits'%(tracer,sliceidx,totslice),overwrite=True)    
        print('written %s/subset_dr9_%s.fits'%(topdir,tracer))    
    
    def _brick_match_dr9(self, X):
        (bricklist, outdir,tracer,south) = X
        final_tab = None
        for brickname in bricklist:
            catalog = BaseSource(filetype='tractor', survey_dir=outdir, outdir=outdir,subsection=None, brick=brickname)
            tracer_sel = catalog.target_selection(target=tracer,south=south)
            tractor_fn = catalog.find_file('tractor')
            T = Table.read(tractor_fn)
            T_new = T[tracer_sel]
            if final_tab is not None:
                final_tab = vstack((final_tab,T_new))
            else:
                final_tab = T_new
        return final_tab
    
    def brick_match_input_random(self, bricklist, threads = None, mp = None):
        outdir = os.path.dirname(os.path.abspath(self.outdir))+'/divided_randoms/'
        final_tab = None
        bricklist_split = np.array_split(bricklist,threads)
        inputs = []
        for i in range(threads):
             inputs.append((bricklist_split[i],outdir))
        results = mp.map(self._brick_match_input_random, inputs)
        final_tab = None
        for result in results:
            if final_tab is None:
                final_tab = result
            else:
                final_tab = vstack((final_tab,result))
        topdir = os.path.dirname(os.path.abspath(self.outdir))+'/subset'
        final_tab.write(topdir+'/subset_intput_random.fits',overwrite=True)
        print('written %s/subset_intput_random.fits'%(topdir))
    
    def brick_match_random(self, bricklist, threads = None, mp = None, totslice = None, sliceidx = None):
        outdir = os.path.dirname(os.path.abspath(self.outdir))+'/randoms/'
        final_tab = None
        
        bricklist_split = np.array_split(bricklist,threads)
        
        
        inputs = []
        for i in range(threads):
            inputs.append((bricklist_split[i],outdir))
        results = mp.map(self._brick_match_random, inputs)
        final_tab = None
        for result in results:
            if final_tab is None:
                final_tab = result
            else:
                final_tab = vstack((final_tab,result))
        topdir = os.path.dirname(os.path.abspath(self.outdir))+'/subset'
        final_tab.write(topdir+'/subset_random_part%d_of%d.fits'%(sliceidx,totslice),overwrite=True)    
        print('written %s/subset_random.fits'%(topdir)) 
    
    def _brick_match_input_random(self, X):
        (bricklist, outdir) = X
        final_tab = None
        for brickname in bricklist:
            tractor_fn = outdir+'brick_%s.fits'%brickname
            T = Table.read(tractor_fn)[:300]
            T2 = Table.read(tractor_fn)[350:400]
            T = vstack((T,T2))
            if final_tab is not None:
                final_tab = vstack((final_tab,T))
            else:
                final_tab = T
        return final_tab

    def _brick_match_random(self, X):
        (bricklist, outdir) = X
        fn_random = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/LRG_SV3/truth/randoms_south_desi.fits"
        T = Table.read(fn_random)
        sel = np.zeros(len(T),dtype=np.bool)
        final_tab = None
        for brickname in bricklist:
            brick_sel = (T['BRICKNAME']==brickname)
            sel[brick_sel] = True
        return T[sel]

    def _brick_match_random_tmp(self, X):
        (bricklist, outdir) = X
        final_tab = None
        for brickname in bricklist:
            tractor_fn = outdir+'random-%s.fits'%brickname
            T = Table.read(tractor_fn)
            if final_tab is not None:
                final_tab = vstack((final_tab,T))
            else:
                final_tab = T
        return final_tab
        
    def _brick_match_core(self, X):
        '''
        mode:
        -- sim: match outputs to sim, output shares same dimension as sim
        -- tractor: match outputs to tractor, output shares same dimension as tractor
        '''
        (bricklist, startid, nobj, angle, mode, MS_star, tracer) = X
        final_tab = None
        #1783p350
        for brickname in bricklist:
            #print(brickname)
            sim = Table.read(self.find_file('simcat',brick=brickname))
            if len(sim)==0:
                #bricks with not injects (all inputs are masked)
                print(brickname)
                print('None')
                continue
            fn_tractor = self.find_file('tractor',brick=brickname)
            if os.path.isfile(fn_tractor) is False:
                # this happens when all sim inputs are outside ccd boundaries
                print('tractor None')
                continue
            catalog = BaseSource(filetype='tractor', survey_dir=self.survey_dir, outdir=self.outdir,subsection=self.subsection, brick=brickname)
            tracer_sel = catalog.target_selection(tracer,south=True)
            
            tractor = Table.read(self.find_file('tractor',brick=brickname))
            tractor[tracer] = tracer_sel
            #MS stars
            try:
                original_sim = Table.read(self.find_file('simorigin',brick=brickname))[startid:startid+nobj] 
            except:
                original_sim = Table.read(self.find_file('simorigin',brick=brickname))

            c1 = SkyCoord(ra=sim['ra']*u.degree, dec=sim['dec']*u.degree)
            c2 = SkyCoord(ra=np.array(tractor['ra'])*u.degree, dec=np.array(tractor['dec'])*u.degree)
            c3 = SkyCoord(ra=original_sim['ra']*u.degree, dec=original_sim['dec']*u.degree)
            if mode == 'sim':
                try:
                    idx1, d2d, d3d = c1.match_to_catalog_sky(c2)
                    idx2, d2d2, d3d2 = c1.match_to_catalog_sky(c3)
                except Exception as e: 
                    print(e)
                    print("bad brick %s"%brickname)
                    raise ValueError(brickname)

                matched = d2d.value <= angle
                distance = d2d.value
                tc = tractor[idx1]
                ors = original_sim[idx2]
            
            elif mode == 'tractor':
                idx1, d2d, d3d = c2.match_to_catalog_sky(c1)
                idx2, d2d, d3d = c2.match_to_catalog_sky(c3)
                matched = d2d.value <= angle
                distance = d2d.value
                tc = tractor
                sim = sim[idx1]
                ors = original_sim[idx2]
    
            tc.add_column(sim['ra'],name = 'sim_ra')
            tc.add_column(sim['dec'],name = 'sim_dec')
            tc.add_column(sim['gflux'],name = 'sim_gflux')
            tc.add_column(sim['rflux'],name='sim_rflux')
            tc.add_column(sim['zflux'],name='sim_zflux')
            tc.add_column(ors['w1'],name='sim_w1')
            tc.add_column(ors['w2'],name='sim_w2')
            tc.add_column(ors['id_sample'],name='id_sample')
            tc.add_column(ors['redshift'],name='sim_redshift')
            tc.add_column(sim['rhalf'],name='sim_rhalf')
            tc.add_column(sim['e1'],name='sim_e1')
            tc.add_column(sim['e2'],name='sim_e2')
            tc.add_column(sim['x'],name='sim_bx')
            tc.add_column(sim['y'],name='sim_by')
            tc['angle'] = np.array(d2d.value*3600.,dtype=np.float)
            tc['matched'] = np.array(matched,dtype=np.bool)
            tc.add_column(sim['n'],name='sim_sersic_n')
            #add depth records from dr9 map
            bx = (tc['bx']+0.5).astype(int)
            by = (tc['by']+0.5).astype(int)
            fn_pattern = "/global/cfs/cdirs/cosmo/work/legacysurvey/dr9/%s/coadd/%s/%s/legacysurvey-%s-depth-%s.fits.fz"
            psfdepth_g = fits.getdata(fn_pattern % (self.region, brickname[:3], brickname, brickname, 'g'))
            psfdepth_r = fits.getdata(fn_pattern % (self.region, brickname[:3], brickname, brickname, 'r'))
            psfdepth_z = fits.getdata(fn_pattern % (self.region, brickname[:3], brickname, brickname, 'z'))
            val_g = psfdepth_g[(by),(bx)]
            val_r = psfdepth_r[(by),(bx)]
            val_z = psfdepth_z[(by),(bx)]
            tc['dr9_psfdepth_g'] = val_g
            tc['dr9_psfdepth_r'] = val_r
            tc['dr9_psfdepth_z'] = val_z
            if MS_star:
                #match closest MS star to the output
                metric = fits.getdata(self.find_file('ref-sources',brick=brickname))
                c1 = SkyCoord(ra=tc['ra'],dec=tc['dec'])
                c2 = SkyCoord(ra=metric['ra']*u.degree, dec=metric['dec']*u.degree)
                idx1, d2d, d3d = c1.match_to_catalog_sky(c2)
                mt = metric[idx1]
                distance = d2d.value
                tc['star_distance'] = np.array(distance,dtype=np.float)
                tc['star_radius'] = np.array(mt['radius'], dtype=np.float)
                tc['MS_delta_ra'] = np.array(mt['ra']-tc['ra'],dtype=np.float)
                tc['MS_delta_dec']= np.array(mt['dec']-tc['dec'],dtype=np.float)
            if final_tab is None:
                final_tab = tc
            else:
                final_tab = vstack((final_tab,tc))
        return final_tab

    
    
def add_more_TS(prefix, survey_dir, outdir, startid, set_num, south=True):
    #TS on cards like 'flux_g%s'%prefix
    catalog = BaseSource(filetype='processed_one_rsp', survey_dir=survey_dir, outdir=outdir,subsection='rs%d_cosmos%d'%(startid,set_num),brick=None, force_construct=True)
    LRG_rsp = catalog.target_selection('LRG_sv3',south=south, prefix='_rsp')
    fn = catalog.find_file('processed_one_rsp')
    T = Table.read(fn)
    T['lrg_sv3%s'%prefix] = LRG_rsp
    T.write(fn, overwrite=True)
    print("written %s"%fn)
if __name__ == "__main__":
    survey_dir = "/global/project/projectdirs/cosmo/data/legacysurvey/dr9/" 
    outdir = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_resampled_seed/output/"
    startid = 9999
    set_num = 80
    
    add_more_TS(prefix = "_rsp",survey_dir = survey_dir, outdir = outdir, startid = startid, set_num = set_num)
