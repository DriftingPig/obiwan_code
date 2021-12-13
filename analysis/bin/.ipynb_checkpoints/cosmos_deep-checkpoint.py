"""
This is a script for collecting truth catalog from cosmos deep debiasing run
#TODO: 
1. filenames here need to be changed for a better structure. 
2. the data products here needs more description 
"""
import sys
sys.path.append("../")
from filesystem import LegacySimData 
import subprocess
from glob import glob
import astropy.io.fits as fits
from astropy.table import Table
from SurveySource import BaseSource
import os
import numpy as np
import  glob 
import os
from astropy.table import vstack,Table
from SurveySource import BaseSource
from astropy.coordinates import SkyCoord
from astropy import units as u
#keys to be collected to generate sweep files
sweep_keys= ['BRICKNAME','RA','DEC','TYPE','OBJID','EBV','FLUX_G','FLUX_R','FLUX_Z','FLUX_W1','FLUX_W2','FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','MW_TRANSMISSION_W1','MW_TRANSMISSION_W2','NOBS_G','NOBS_R','NOBS_Z','NOBS_W1','NOBS_W2','SHAPE_R','SHAPE_E1','SHAPE_E2','FIBERFLUX_G','FIBERFLUX_R','FIBERFLUX_Z','MASKBITS','SERSIC','DCHISQ','PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z','GALDEPTH_G','GALDEPTH_R','GALDEPTH_Z','WISEMASK_W1','WISEMASK_W2','ANYMASK_G','ANYMASK_R','ANYMASK_Z','BX','BY','GAIA_PHOT_G_MEAN_MAG','FIBERTOTFLUX_Z']

class CosmosDeep(object):
    def __init__(self):
        self.origin_outdir = '/global/project/projectdirs/cosmo/work/legacysurvey/dr9.1.1/'
        self.dr9_outdir = '/global/project/projectdirs/cosmo/work/legacysurvey/dr9/south/'
        self.survey_dir = self.dr9_outdir
        self.outdir = self.origin_outdir
        self.savedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_deep/'
        self.bricklist = self._get_bricklist()
        self.ccdnum_fn = os.path.join(self.savedir,'ccd_num.fits')
        self.cut_bricklist = np.loadtxt(os.path.join(self.savedir,'bricklist_cutted.txt'),dtype = np.str)
    def _get_bricklist(self,write=True):
        #get list of deep bricks by finding bricknames in coadd files
        if os.path.isfile(os.path.join(self.savedir,'bricklist.txt')):
            bricklist = np.loadtxt(os.path.join(self.savedir,'bricklist.txt'),dtype = np.str)
            return bricklist
        fns = glob.glob(os.path.join(self.outdir, 'coadd','*','*'))
        bricklist = []
        for fn in fns:
            bricklist.append(os.path.basename(fn))
        bricklist = np.array(bricklist, dtype = np.str)
        if write:
            np.savetxt(os.path.join(self.savedir,'bricklist.txt'), bricklist, fmt="%s")
        return bricklist
    def get_ccd_num(self):
        ccd1 = []
        ccd2 = []
        depthz1=[]
        depthz2=[]
        for brickname in self.bricklist:
            ccd_fn = os.path.join(self.origin_outdir, 'coadd', brickname[:3], brickname, 'legacysurvey-%s-ccds.fits'%brickname)
            depthz_fn = os.path.join(self.origin_outdir, 'coadd', brickname[:3], brickname, 'legacysurvey-%s-depth-z.fits.fz'%brickname)
            ccd_num1 = len(fits.getdata(ccd_fn))
            depthz_median = np.median(fits.getdata(depthz_fn).ravel())
            
            ccd_fn_dr9 = os.path.join(self.dr9_outdir, 'coadd', brickname[:3], brickname, 'legacysurvey-%s-ccds.fits'%brickname)
            depthz_fn_dr9 = os.path.join(self.dr9_outdir, 'coadd', brickname[:3], brickname, 'legacysurvey-%s-depth-z.fits.fz'%brickname)
            ccd_num2 = len(fits.getdata(ccd_fn_dr9))
            depthz_dr9_median = np.median(fits.getdata(depthz_fn_dr9).ravel())
            
            ccd1.append(ccd_num1)
            ccd2.append(ccd_num2)
            depthz1.append(depthz_median)
            depthz2.append(depthz_dr9_median)
        ccd1 = np.array(ccd1)
        ccd2 = np.array(ccd2)
        depthz1 = np.array(depthz1)
        depthz2 = np.array(depthz2)
        T = Table()
        T['brickname'] = self.bricklist
        T['ccd_deep'] = ccd1
        T['ccd_dr9'] = ccd2
        T['depthz_deep'] = depthz1
        T['depthz_dr9'] = depthz2
        T.write(os.path.join(self.savedir,'ccd_num.fits'), overwrite = True)
    def cut_bricks(self, scale = None, depthz_cut = None, write=True):
        #cut the bricks that does not have enough ccds, so deep_ccd_num>dr9_ccd_num*scale or using median galdepth cut, this is a rough cut, I use galdepth_z>300
        t = fits.getdata(self.ccdnum_fn)
        print(t['ccd_dr9'].max(),t['ccd_dr9'].min())
        if scale is not None:
            sel = t['ccd_deep']>scale*t['ccd_dr9']
        else:
            sel = t['depthz_deep']>depthz_cut
        print('total ccd: %d, after cut: %d'%(len(sel), sel.sum()))
        if write:
            np.savetxt(os.path.join(self.savedir,'bricklist_cutted.txt'),t[sel]['brickname'], fmt="%s")
        self.cut_bricklist = t[sel]['brickname']
    def get_cosmos_repeats_lists(self):
        #return a list of cosmos repeats bricks
        self.reference_outdir = '/global/cscratch1/sd/dstn/dr9-cosmos-subs/'
        fns = glob.glob(os.path.join(self.reference_outdir,'80','coadd','*','*'))
        bricklist = []
        for fn in fns:
            bricklist.append(os.path.basename(fn))
        self.repeat_bricklist = bricklist
        return bricklist
    def make_truth(self,TYPE='deep'):
        #make truth inputs from cosmos deep region
        #deep is collecting deep data, dr9 is corresponding dr9 data, both using self.cut_bricklist defined previously 
        assert(TYPE in ["deep","dr9"])
        if TYPE == "deep":
            output_fn = "truth.fits"
        if TYPE == "dr9":
            output_fn = "dr9_mirror.fits"
        tab = None
        for brickname in self.cut_bricklist:
            print(brickname)
            if TYPE == "deep":
                self.catalog = LegacySimData(survey_dir=self.survey_dir, outdir=self.outdir, brick=brickname)
            elif TYPE == "dr9":
                self.catalog = LegacySimData(survey_dir=self.survey_dir, outdir=self.dr9_outdir, brick=brickname)
            else:
                raise
            tractor_fn = self.catalog.find_file('tractor')
            dat_i = Table.read(tractor_fn)
            tab_i = Table()
            for key in sweep_keys:
                tab_i[key.lower()] = dat_i[key.lower()]
            if tab is None:
                tab = tab_i
            else:
                tab = vstack((tab, tab_i))
        tab.write(self.savedir+output_fn,overwrite=True)
        print("saved %s"%(self.savedir+output_fn))
    def add_cards(self, filetype, obj):
        #only add it to dr9 since w1 in deep does not work
        assert(filetype in ["cosmos_deep_dr9","cosmos_deep"])
        assert(obj in ["LRG_sv3_like","LRG_sv3"])
        catalog_i = BaseSource(filetype=filetype, survey_dir=self.survey_dir, outdir=self.outdir,force_construct=True)
        card = catalog_i.target_selection(obj)
        t_truth = Table.read(catalog_i.source_fn)
        t_truth[obj]=card
        t_truth.write(catalog_i.source_fn, overwrite=True)
        print(catalog_i.source_fn)
    def match(self, filetype1, filetype2):
        if filetype1 == "cosmos_deep" and filetype2 == "cosmos_deep_dr9":
            #match dr9 to truth
            prefix = "dr9"
        elif filetype1 == "cosmos_deep_dr9" and filetype2 == "cosmos_deep":
            #match truth to dr9
            prefix = "truth"
        catalog_1 = BaseSource(filetype=filetype1, survey_dir=self.survey_dir, outdir=self.outdir,force_construct=True)
        catalog_2 = BaseSource(filetype=filetype2, survey_dir=self.survey_dir, outdir=self.outdir,force_construct=True)
        cat1, cat2, matched = catalog_1.match_catalog(catalog_1.source, catalog_2.source)
        T = Table()
        T['matched'] = matched
        input_sweep_keys = sweep_keys.copy()
        input_sweep_keys.extend(['LRG_sv3_like','LRG_sv3'])
        for key in input_sweep_keys:
            T[key.lower()] = cat1[key.lower()]
            T["%s_%s"%(prefix, key.lower())] = cat2[key.lower()]
        names = catalog_1.source.columns.names
        for name in names:
            if name not in input_sweep_keys:
                T[key.lower()] = cat1[key.lower()]
        T.write(catalog_1.source_fn,overwrite=True)
    def truth_match_to_cosmos(self):
        bricknames = self.get_cosmos_repeats_lists()
        catalog_1 = BaseSource(filetype='cosmos_deep', survey_dir=self.survey_dir, outdir=self.outdir,force_construct=True)
        cat1_all = catalog_1.source
        sels = np.zeros(len(cat1_all),dtype=np.bool)
        
        for brickname in bricknames:
            sel = (cat1_all['brickname']==brickname)
            sels+=sel
        cat1 = cat1_all[sels]
        tot_sets = np.arange(0,10)
        topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cosmos_all_stacked/'
        T = Table()
        for one_set in tot_sets:
            print(one_set)
            cat2 = fits.getdata('/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/cosmos_all_stacked/cosmos_set%d.fits'%one_set)
            c1 = SkyCoord(ra=cat1['ra']*u.degree, dec=cat1['dec']*u.degree)
            c2 = SkyCoord(ra=cat2['ra']*u.degree, dec=cat2['dec']*u.degree)
            idx, d2d, d3d = c1.match_to_catalog_sky(c2)
            matched = d2d.value <= 1./3600
            distance = d2d.value
            cat2 = cat2[idx]
            T['set_%d_matched'%one_set] = matched
            T['set_%d_sv3_lrg'%one_set] = cat2['set_%d_lrg_sv3'%one_set]
            T['set_%d_sv3_elg_hip'%one_set] = cat2['set_%d_elg_sv3_hip'%one_set]
            for key in sweep_keys:
                T[key.lower()] = cat1[key.lower()]
                T["set%d_%s"%(one_set, key.lower())] = cat2[key.lower()]
        T['lrg_sv3'] = cat1['lrg_sv3']
        T['lrg_sv3_like'] = cat1['lrg_sv3_like']
        T.write(self.savedir+"/truth_cosmos_repeats.fits",overwrite=True)
    def run(self):
        cd = self
        
        cd.get_ccd_num()
        cd.cut_bricks(depthz_cut = 300,write=True)
        print("make_truth1")
        cd.make_truth(TYPE = "deep")
        print("make_truth2")
        cd.make_truth(TYPE="dr9")
        print("add_cards1")
        cd.add_cards(filetype="cosmos_deep_dr9",obj='LRG_sv3_like')
        print("add_cards2")
        cd.add_cards(filetype="cosmos_deep_dr9",obj='LRG_sv3')
        print("add_cards3")
        cd.add_cards(filetype="cosmos_deep",obj='LRG_sv3_like')
        print("add_cards4")
        cd.add_cards(filetype="cosmos_deep",obj='LRG_sv3')
        print("match1")
        cd.match(filetype1 = "cosmos_deep", filetype2="cosmos_deep_dr9")
        print("match2")
        cd.match(filetype1 = "cosmos_deep_dr9", filetype2="cosmos_deep")
        print("truth_match_to_cosmos")
        cd.truth_match_to_cosmos()
    def split(self, filetype, topdir, target):
        #split the files into per brick file needed for obiwan run
        assert(filetype in ["cosmos_deep_dr9","cosmos_deep"])
        assert(target in ["LRG_sv3_like","LRG_sv3"])
        catalog_i = BaseSource(filetype=filetype, survey_dir=self.survey_dir, outdir=self.outdir,force_construct=True)
        source = catalog_i.source
        ids = np.arange(len(source))
        for brickname in self.cut_bricklist:
            
            print(brickname)
            sel = (source['brickname']==brickname)&(source[target])
            print(sel.sum())
            T = Table()
            T['ra'] = source[sel]['ra']
            T['dec'] = source[sel]['dec']
            T['e1'] = source[sel]['shape_e1']
            T['e2'] = source[sel]['shape_e2']
            T['n'] = source[sel]['sersic']
            #some g band is nan, set it to a high mag
            T['g'] = 22.5 - 2.5*np.log10(source[sel]['flux_g']/source[sel]['mw_transmission_g'])
            idx = np.where(source[sel]['flux_g']<=0)
            T['g'][idx] = 30
            T['r'] = 22.5 - 2.5*np.log10(source[sel]['flux_r']/source[sel]['mw_transmission_r'])
            T['z'] = 22.5 - 2.5*np.log10(source[sel]['flux_r']/source[sel]['mw_transmission_z'])
            T['w1'] = 22.5 - 2.5*np.log10(source[sel]['flux_w1']/source[sel]['mw_transmission_w1'])
            T['w2'] = np.clip(0,30,22.5 - 2.5*np.log10(source[sel]['flux_w2']/source[sel]['mw_transmission_w2']))
            T['rhalf'] = source[sel]['shape_r']
            T['id'] = ids[sel]
            T.write(topdir+'/brick_%s.fits'%brickname,overwrite=True)
     
    def split_elgs(self, topdir):
        fn = "/global/cscratch1/sd/adematti/legacysim/dr9/cosmos/merged/truth_ELG_HIP.fits"
        source = fits.getdata(fn)
        ids = np.arange(len(source))
        for brickname in self.cut_bricklist:
            
            print(brickname)
            sel = (source['brickname']==brickname)
            print(sel.sum())
            T = Table()
            T['ra'] = source[sel]['ra']
            T['dec'] = source[sel]['dec']
            T['e1'] = source[sel]['shape_e1']
            T['e2'] = source[sel]['shape_e2']
            T['n'] = source[sel]['sersic']
            T['g'] = 22.5 - 2.5*np.log10(source[sel]['flux_g']/source[sel]['mw_transmission_g'])
            T['r'] = 22.5 - 2.5*np.log10(source[sel]['flux_r']/source[sel]['mw_transmission_r'])
            T['z'] = 22.5 - 2.5*np.log10(source[sel]['flux_r']/source[sel]['mw_transmission_z'])
            T['w1'] = 22.5*np.ones(sel.sum())
            T['w2'] = 22.5*np.ones(sel.sum())
            T['rhalf'] = source[sel]['shape_r']
            T['id'] = ids[sel]
            T.write(topdir+'/brick_%s.fits'%brickname,overwrite=True)
    def collect_tracers(self, tracer):
        assert(tracer in ['elg','lrg'])
        all_fn = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/output/rs0/tractor/*/*"
        all_fns = glob.glob(all_fn)
        elg_fns = []
        lrg_fns = []
        for fn in all_fns:
            if 'elg' in fn:
                elg_fns.append(fn)
            else:
                lrg_fns.append(fn)
        if tracer == 'elg':
                fns = elg_fns
        else:
                fns = lrg_fns
        
        samp = None
        for fn in fns:
            tt = fits.getdata(fn)
            if 'current_gflux' in tt.columns.names:
                tracer_i = Table.read(fn)
                if samp is None:
                    samp = tracer_i
                else:
                    samp = vstack((samp,tracer_i))
        samp['bad'] = np.zeros(len(samp),dtype = np.bool)
        sel = (samp['n']==-999)
        samp['bad'][sel] = np.ones(sel.sum(),dtype = np.bool)
        samp['n'] = np.clip(samp['n'],0.2,7)
        samp.write("/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/%s_v2.fits"%tracer,overwrite = True)
        print("written /global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/%s_v2.fits"%tracer)
        
        if tracer=='elg':
            topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/divided_randoms_elg/'
        else:
            topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/divided_randoms/'
        final = None
        for fn in fns:
            tracer_i = Table.read(fn)
            brickname = fn[-13:-5]
            randoms_i = Table.read(topdir+"brick_%s.fits"%brickname)
            assert(len(tracer_i)==len(randoms_i))
            if len(tracer_i)>0:
                tracer_i['bad'] = np.zeros(len(tracer_i),dtype = np.bool)
                sel = (tracer_i['n']==-999)
                tracer_i['bad'][sel] = np.ones(sel.sum(),dtype = np.bool)
                sel = (~tracer_i['fitted'])|(tracer_i['bad'])
                randoms_i['resampled_e1'] = tracer_i['e1']
                randoms_i['resampled_e1'][sel] = randoms_i['e1'][sel]
        
                randoms_i['resampled_e2'] = tracer_i['e2']
                randoms_i['resampled_e2'][sel] = randoms_i['e2'][sel]
        
                randoms_i['resampled_rhalf'] = tracer_i['rhalf']
                randoms_i['resampled_rhalf'][sel] = randoms_i['rhalf'][sel]
        
                randoms_i['resampled_n'] = tracer_i['n']
                randoms_i['resampled_n'][sel] = randoms_i['n'][sel]
        
                randoms_i['brickname'] = np.array([brickname]*len(randoms_i),dtype=np.str)
                 
                randoms_i['bad'] = tracer_i['bad']
                
        
                if final is None:
                    final = randoms_i
                else:
                    final = vstack((final,randoms_i))
        final.write("/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/%s_final_v2.fits"%tracer,overwrite=True)
        print("written /global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/%s_final_v2.fits"%tracer)
        
        if tracer == 'lrg':
            seed_fn = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_deep/truth.fits'
        if tracer == 'elg':
            seed_fn = "/global/cscratch1/sd/adematti/legacysim/dr9/cosmos/merged/truth_ELG_HIP.fits"
            
        fn1 = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/%s_final.fits"%tracer
        fn2 = seed_fn
        dat1 = fits.getdata(fn1)
        dat2 = fits.getdata(fn2)
        print("bad: %d"%dat1['bad'].sum())
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        c2 = SkyCoord(ra=dat1['ra']*u.degree, dec=dat1['dec']*u.degree)
        c1 = SkyCoord(ra=np.array(dat2['ra'])*u.degree, dec=np.array(dat2['dec'])*u.degree)
        idx, d2d, d3d = c1.match_to_catalog_sky(c2)
        matched = (d2d.value<1./3600)
        truth = Table.read(fn2)
        truth['matched'] = matched
        truth['resampled_e1'] = dat1['resampled_e1'][idx]
        truth['resampled_e2'] = dat1['resampled_e2'][idx]
        truth['resampled_rhalf'] = dat1['resampled_rhalf'][idx]
        truth['resampled_n'] = dat1['resampled_n'][idx]
        truth['bad'] = dat1['bad'][idx]

        truth.write("/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/%s_truth_v2.fits"%tracer,overwrite = True)
        print("written /global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/%s_truth_v2.fits"%tracer)
        
    def make_seed(self):
        fn = "/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/lrg_truth.fits"
        dat = fits.getdata(fn)
        sel = dat['LRG_sv3_like']
        print("matched: %d/%d"%((dat['matched']&sel).sum(),sel.sum()))
        sel = dat['matched']&dat['LRG_sv3_like']&(dat['galdepth_z']>1000)&(~dat['bad'])
        T = Table()
        T['ra'] = dat[sel]['ra']
        T['dec'] = dat[sel]['dec']
        gmag = 22.5 - 2.5*np.log10(dat[sel]['flux_g']/dat[sel]['mw_transmission_g'])
        g_sel = ~((gmag>0)&(gmag<30))
        gmag[g_sel] = 30
        rmag = 22.5 - 2.5*np.log10(dat[sel]['flux_r']/dat[sel]['mw_transmission_r'])
        zmag = 22.5 - 2.5*np.log10(dat[sel]['flux_z']/dat[sel]['mw_transmission_z'])
        w1mag = 22.5 - 2.5*np.log10(dat[sel]['flux_w1']/dat[sel]['mw_transmission_w1'])
        w2mag = 22.5 - 2.5*np.log10(dat[sel]['flux_w2']/dat[sel]['mw_transmission_w2'])
        T['g'] = gmag
        T['r'] = rmag
        T['z'] = zmag
        T['w1'] = w1mag
        T['w2'] = w2mag
        T['e1'] = dat[sel]['resampled_e1']
        T['e2'] = dat[sel]['resampled_e2']
        T['n'] = np.clip(dat[sel]['resampled_n'],0.2,7)
        T['rhalf'] = dat[sel]['resampled_rhalf']
        T['id_sample'] = np.arange(sel.sum())
        T.write("/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_deep/seed.fits",overwrite=True)
        
        
        
        
        
        
if __name__ == '__main__':
    cd = CosmosDeep()
    cd.run()
    topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/divided_randoms/'
    cd.split(filetype = "cosmos_deep", topdir=topdir, target="LRG_sv3_like")
    topdir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/sim_deep/divided_randoms_elg/'
    cd.split_elgs(topdir)
    cd.collect_tracers('lrg')
    cd.make_seed()
            
            
    
