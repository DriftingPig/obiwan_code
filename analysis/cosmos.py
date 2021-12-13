from filesystem import LegacySimData 
import subprocess
from glob import glob
import astropy.io.fits as fits
from astropy.table import Table,vstack
from SurveySource import BaseSource
import os
import numpy as np

class CosmosRepeats(LegacySimData):
    def __init__(self, survey_dir=None, outdir=None, subsection=None, brick=None, startid=None, **kwargs):
        
        self.reference_surveydir = '/global/cscratch1/sd/dstn/cosmos-subs-lsd/'
        self.reference_outdir_top = '/global/cscratch1/sd/dstn/dr9-cosmos-subs/'
        self.reference_outdir = '/global/cscratch1/sd/dstn/dr9-cosmos-subs/80/'
        super(CosmosRepeats,self).__init__(survey_dir=survey_dir, outdir=outdir, subsection=subsection, brick=brick)
        self.total_sets = ['80', '81', '82', '83', '84', '85', '86', '87', '88', '89']
        self.set_bricklist()
        #self.maskdir = self.find_file('masks')
        self.startid = startid
        
    def get_legacypipe_ver(self):
        #docker version used in this cosmos repeats run, documentation purpose
        return 'legacysurvey/legacypipe:DR9.7.1'
    
    def set_bricklist(self):
        fns = glob(os.path.join(self.reference_outdir,'coadd','*','*'))
        bricklist = []
        for fn in fns:
            bricklist.append(os.path.basename(fn))
        self.bricklist = bricklist
  
    def stack_all_rs(self):
        startids = ['0','50','100','150','200','250','300','350','400','450','500']
        for one_set in self.total_sets:
            print(one_set)
            final_tab = None
            for startid in startids:
                catalog = BaseSource(survey_dir=self.survey_dir, outdir=self.outdir,subsection = 'rs%s_cosmos%s'%(startid,one_set))
                fn = catalog.find_file("processed_one")
                tab = Table.read(fn)
                tab['startid'] = np.array([startid]*len(tab),dtype=np.str)
                if final_tab is None:
                    final_tab = tab
                else:
                    final_tab = vstack((final_tab,tab))
            #just setting 9999 as stacking 'all'
            catalog = BaseSource(survey_dir=self.survey_dir, outdir=self.outdir,subsection = 'rs9999_cosmos%s'%(one_set))
            fn = catalog.find_file("processed_one")
            
            final_tab.write(fn,overwrite=True)
            print("written fn %s"%fn)
                
   
    def set_maskbits_cross(self,mp=None):
        if mp is None:
            print('mp needs to be set')
            return None
        subprocess.call(["mkdir","-p",self.outdir+'/maskbits'])
        mp.map(self._set_maskbits_cross,self.bricklist)
        
    def _set_maskbits_cross(self,brickname):
        maskbits_img = np.zeros((3600,3600),dtype=np.int)
        for one_set in self.total_sets:
            cosmos_catalog = LegacySimData(survey_dir=self.survey_dir,outdir=self.outdir,brick=brickname, subsection='rs%d_cosmos%s'%(self.startid,one_set))
            maskbits_fn = cosmos_catalog.find_file('maskbits')
            maskbits = fits.getdata(maskbits_fn)
            maskbits_img  |= maskbits
        hdu = fits.ImageHDU(data=maskbits_img)
        fn = cosmos_catalog.find_file('maskbits_cross')
        fn = fn.replace('.fits','_rs%d.fits'%self.startid)
        hdu.writeto(fn,overwrite=True)
        print("written %s"%fn)
        
    def mask_tractor(self,mp=None):
        #add an additial colum to tractor file, 'matched_cosmos': the sources that's in the common maskbits==0 footprint
        if mp is None:
            print('mp needs to be set')
            return None
        mp.map(self._mask_tractor_core,self.bricklist)
        return True
        
    
    def _mask_tractor_core(self,brickname):
        if brickname is None:
            return None
        cosmos_catalog = LegacySimData(survey_dir=self.survey_dir,outdir=self.outdir,brick=brickname, subsection='rs%d_cosmos80'%(self.startid))
        cosmos_catalog.get_mask(brick=brickname)
        cosmos_catalog.get_maskbits_corss(brick=brickname,startid=self.startid)
        for one_set in self.total_sets:
                cosmos_catalog.subsection = 'rs%d_cosmos%s'%(self.startid, one_set)
                tractor_fn = cosmos_catalog.find_file('tractor')
                tractor = Table.read(tractor_fn)
                bx = (tractor['bx']+0.5).astype(int)
                by = (tractor['by']+0.5).astype(int)
                mask_flag = cosmos_catalog.mask[(by),(bx)]
                maskbits_corss_flag = cosmos_catalog.maskbits_cross[(by),(bx)]
                sel = (mask_flag==1)
                tractor['matched_cosmos'] = sel
                tractor['maskbits_cross'] = maskbits_corss_flag
                tractor.write(tractor_fn,overwrite=True)
        return True
    
    def match_catalog(self,mp=None,south=True):
        #adding additial columns to sources, showing the corresponding values in other sets
        if mp is None:
            #for debug
            X = (self.bricklist[0],self.total_sets[1],south)
            self._match_catalog_core(X)
            return None
        inputs = []
        for brickname in self.bricklist:
            for one_set in self.total_sets:
                inputs.append((brickname,one_set,south))
        mp.map(self._match_catalog_core,inputs)
        
    def _match_catalog_core(self,X):
        (brickname,set_num,south) = X
        self.brick = brickname
        self.subsection = 'rs%d_cosmos%s'%(self.startid,set_num)
        tractor_fn = self.find_file('tractor')
        T = Table.read(tractor_fn)
        catalog = BaseSource(filetype='tractor', survey_dir=self.survey_dir, outdir=self.outdir,subsection='rs%d_cosmos%s'%(self.startid,set_num),brick=brickname)
        LRG = catalog.target_selection('LRG_sv3',south=south)
        #lrg
        T['set_%s_lrg_sv3'%set_num] = LRG
        LRG_like = catalog.target_selection('LRG_sv3_like', south=south)
        T['set_%s_lrg_sv3_like'%set_num] = LRG_like
        for one_set in self.total_sets:
            #if one_set == set_num:
                #continue

            catalog_i = BaseSource(filetype='tractor', survey_dir=self.survey_dir, outdir=self.outdir,subsection='rs%d_cosmos%s'%(self.startid,one_set),brick=brickname)
            cat1, cat2, matched = catalog.match_catalog(catalog.source, catalog_i.source)
            catalog_i.source = cat2
            catalog_i._construct_class()
            LRG = catalog_i.target_selection('LRG_sv3',south=south)
            T['set_%s_lrg_sv3'%one_set] = LRG
            T['set_%s_matched'%one_set]=matched
            T['set_%s_flux_g'%one_set]=cat2['flux_g']
            T['set_%s_flux_r'%one_set]=cat2['flux_r']
            T['set_%s_flux_z'%one_set]=cat2['flux_z']
            T['set_%s_flux_w1'%one_set]=cat2['flux_w1']
            T['set_%s_flux_w2'%one_set]=cat2['flux_w2']
            T['set_%s_fiberflux_z'%one_set]=cat2['fiberflux_z']
            T['set_%s_fiberflux_g'%one_set]=cat2['fiberflux_g']
        tmp_fn = tractor_fn.replace('tractor-','tmp-tractor-')
        T.write(tmp_fn, overwrite=True)
        subprocess.call(["mv",tmp_fn,tractor_fn])
        
    def _match_catalog_core_old(self,X): #not used now
        (brickname,set_num,south) = X
        print(X)
        self.brick = brickname
        self.subsection = 'cosmos%s'%set_num
        tractor_fn = self.find_file('tractor')
        T = Table.read(tractor_fn)
        catalog = BaseSource(filetype='tractor', survey_dir=self.survey_dir, outdir=self.outdir,subsection='cosmos%s'%set_num,brick=brickname)
        #select ELG, LRG in this set
        LRG = catalog.target_selection('LRG',south=south)
        #lrg_opt, lrg_ir, lrg_sv_opt, lrg_sv_ir
        T['set_%s_lrgopt'%set_num] = LRG[0]
        T['set_%s_lrgir'%set_num] = LRG[1]
        T['set_%s_lrgsvopt'%set_num] = LRG[2]
        T['set_%s_lrgsvir'%set_num] = LRG[3]
        
        ELG = catalog.target_selection('ELG',south=south)
        #svgtot, svgfib, fdrgtot, fdrgfib
        T['set_%s_svgtot'%set_num] = ELG[0]
        T['set_%s_svgfib'%set_num] = ELG[1]
        T['set_%s_fdrgtot'%set_num] = ELG[2]
        T['set_%s_fdrgfib'%set_num] = ELG[3]
        
        for one_set in self.total_sets:
            #if one_set == set_num:
                #continue
            
            catalog_i = BaseSource(filetype='tractor', survey_dir=self.survey_dir, outdir=self.outdir,subsection='cosmos%s'%one_set,brick=brickname)
            cat1, cat2, matched = catalog.match_catalog(catalog.source, catalog_i.source)
            catalog_i.source = cat2
            catalog_i._construct_class()
            LRG_i = catalog_i.target_selection('LRG',south=south)
            ELG_i = catalog_i.target_selection('ELG',south=south)
            T['set_%s_lrgopt'%one_set] = LRG_i[0]
            T['set_%s_lrgir'%one_set] = LRG_i[1]
            T['set_%s_lrgsvopt'%one_set] = LRG_i[2]
            T['set_%s_lrgsvir'%one_set] = LRG_i[3]
            T['set_%s_svgtot'%one_set] = ELG_i[0]
            T['set_%s_svgfib'%one_set] = ELG_i[1]
            T['set_%s_fdrgtot'%one_set] = ELG_i[2]
            T['set_%s_fdrgfib'%one_set] = ELG_i[3]
            T['set_%s_matched'%one_set]=matched
            T['set_%s_flux_g'%one_set]=cat2['flux_g']
            T['set_%s_flux_r'%one_set]=cat2['flux_r']
            T['set_%s_flux_z'%one_set]=cat2['flux_z']
            T['set_%s_flux_w1'%one_set]=cat2['flux_w1']
            T['set_%s_flux_w2'%one_set]=cat2['flux_w2']
            T['set_%s_fiberflux_z'%one_set]=cat2['fiberflux_z']
            T['set_%s_fiberflux_g'%one_set]=cat2['fiberflux_g']
        tmp_fn = tractor_fn.replace('tractor-','tmp-tractor-')
        T.write(tmp_fn, overwrite=True)
        subprocess.call(["mv",tmp_fn,tractor_fn])
              
        
        
                     
        
        
