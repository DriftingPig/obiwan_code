from filesystem import LegacySimData
import astropy.io.fits as fits
from configs import *
from desitarget import cuts
from desitarget.sv3 import sv3_cuts
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys
sys.path.append('./cosmos_preprocess')
from dr9_tracer_like import isLRG_like
class BaseSourceBase(object):
    '''
    source on a catalog
    '''
    def __init__(self, filetype='tractor', band = 'g', force_construct=False,**kw):
        self.filetype = filetype
        self.band = band
        self.source_fn = self.find_file(filetype = self.filetype, brick=getattr(self,'brick',None), subsection = getattr(self, 'subsection',None), band = getattr(self,'band',None))
        brick = getattr(self,'brick',None)
        if brick is not None:
            self.source = fits.getdata(self.source_fn)
            self._construct_class()
        if brick is None and force_construct:
            self.source = fits.getdata(self.source_fn)
            self._construct_class()
        self.single_source = False
    def get_processed_one(self):
        fn = self.find_file('processed_one')
        print(fn)
        self.processed_one = fits.getdata(fn)
    def _construct_class(self):
        self.names = self.source.columns.names
        
        for name in self.names:
            setattr(self,name.lower(),self.source[name])
    def flux2mag(self,band):
        flux = getattr(self,'flux_%s'%band)
        mwtransmission = getattr(self,'mw_transmission_%s'%band)
        mag = 22.5 - 2.5 * np.log10(flux / mwtransmission)
        return mag
    def mag2flux(self,mag,band):
        mwtransmission = getattr(self,'mw_transmission_%s'%band)
        flux = 10**(-(mag-22.5)/2.5)*mwtransmission
        return flux
    def trueflux(self,band,prefix=''):
        flux = getattr(self,'flux_%s%s'%(band,prefix))
        mwtransmission = getattr(self,'mw_transmission_%s'%band)
        return flux/mwtransmission
    def set_flux(self,band,flux_array):
        setattr(self,'flux_%s'%band, flux_array)
    def SingleSource(self,idx=None):
        if idx is not None:
            self.idx = idx
            if self.single_source:
                print('this is already single source')
                print('using WholeSource to construct back...')
                self.WholeSource()
                self.source = self.source[self.idx]
            else:
                self.source = self.source[self.idx]
        else:
            if self.single_source:
                print('this is already single source, with idx %d'%self.idx)
            else:
                self.source = self.source[self.idx]
        self.single_source = True
        for name in self.names:
            setattr(self,name,self.source[name])
    def WholeSource(self):
        self.single_source = False
        self.source = fits.getdata(self.source_fn)
        self._construct_class()
    def set_idx(self,idx):
        self.idx = dix
    def find_file(self,filetype, **kwargs):
        return filetype
    
    @staticmethod
    def match_catalog(cat1,cat2,radius=1./3600):
        c1 = SkyCoord(ra=cat1['ra']*u.degree, dec=cat1['dec']*u.degree)
        c2 = SkyCoord(ra=cat2['ra']*u.degree, dec=cat2['dec']*u.degree)
        idx, d2d, d3d = c1.match_to_catalog_sky(c2)
        matched = d2d.value <= radius
        distance = d2d.value
        #cat1_new = cat1[matched]
        #cat2_new = cat2[idx1][matched]
        return cat1, cat2[idx], matched
    
    def target_selection(self,target,south=True,prefix=''):
        if target in ['ELG','LRG','LRG_sv3_like','lrg_sv3','elg_sv3'] is False:
            print('currently support ELG,LRG')
            return [-1]
        gflux = self.trueflux('g',prefix=prefix)
        rflux = self.trueflux('r',prefix=prefix)
        zflux = self.trueflux('z',prefix=prefix)
        w1flux = self.trueflux('w1',prefix=prefix)
        w2flux = self.trueflux('w2',prefix='')
        zfiberflux = self.fiberflux_z/self.mw_transmission_z
        gfiberflux = self.fiberflux_g/self.mw_transmission_g
        zfibertotflux = self.fibertotflux_z/self.mw_transmission_z
        if target == 'LRG':
            lrg = \
            cuts.isLRG(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                zfiberflux=zfiberflux, rfluxivar=self.flux_ivar_r, zfluxivar=self.flux_ivar_z, w1fluxivar=self.flux_ivar_w1,
                gnobs=self.nobs_g, rnobs=self.nobs_r, znobs=self.nobs_z, maskbits=self.maskbits, primary=None,
                south=south, gaiagmag = self.gaia_phot_g_mean_mag,zfibertotflux = self.fibertotflux_z)
            return lrg
        if target == 'ELG':
            svgtot, svgfib, fdrgtot, fdrgfib = cuts.isELG(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                        gfiberflux = gfiberflux, gsnr=(gflux>0),rsnr=(rflux>0),zsnr=(zflux>0),
                        gnobs=self.nobs_g, rnobs=self.nobs_r, znobs=self.nobs_z, maskbits=self.maskbits, primary=None,
                        south=south)
            return [svgtot, svgfib, fdrgtot, fdrgfib]
        if target == 'LRG_sv3_like':
            cut_method = isLRG_like(target)
            lrg = cut_method.isLRGlike_color(self.source)
            return lrg 
        if target == 'lrg_sv3':
            lrg, lrg_lowdens = sv3_cuts.isLRG(gflux = gflux,rflux=rflux,zflux=zflux,w1flux=w1flux,zfiberflux=zfiberflux,\
                         gnobs=self.nobs_g,rnobs=self.nobs_r,znobs=self.nobs_z,\
                        rfluxivar=self.flux_ivar_r,zfluxivar=self.flux_ivar_z,w1fluxivar=self.flux_ivar_w1,\
                        gaiagmag=self.gaia_phot_g_mean_mag,maskbits=self.maskbits,zfibertotflux=zfibertotflux,primary=None, south=south)
            return lrg
        if target == 'elg_sv3':
            elglop, elghip = sv3_cuts.isELG(gflux=gflux, rflux=rflux, zflux=zflux, w1flux=w1flux, w2flux=w2flux,
                        gfiberflux = gfiberflux, gsnr=(gflux>0),rsnr=(rflux>0),zsnr=(zflux>0),
                        gnobs=self.nobs_g, rnobs=self.nobs_r, znobs=self.nobs_z, maskbits=self.maskbits, primary=None,
                        south=south)
            return elghip
        
    def stack_all_rs(self,startids):
        #startids = ['0','50','100','150','200','250']
        final_tab = None
        from astropy.table import Table,vstack
        import numpy as np
        for startid in startids:
            catalog = BaseSource(survey_dir=self.survey_dir, outdir=self.outdir,subsection = 'rs%s'%(startid))
            fn = catalog.find_file("processed_one")
            tab = Table.read(fn)
            tab['startid'] = np.array([startid]*len(tab),dtype=np.str)
            if final_tab is None:
                    final_tab = tab
            else:
                    final_tab = vstack((final_tab,tab))
        #just setting 9999 as stacking 'all'
        catalog = BaseSource(survey_dir=self.survey_dir, outdir=self.outdir,subsection = 'rs9999')
        fn = catalog.find_file("processed_one")
        final_tab.write(fn,overwrite=True)
        print("written fn %s"%fn)
        
        
class BaseSource(LegacySimData, BaseSourceBase):
    def __init__(self, filetype='tractor', band='g',survey_dir=None, outdir=None, subsection=None, brick=None,**kw):
        super(BaseSource, self).__init__(filetype=filetype, band=band, survey_dir=survey_dir, outdir=outdir, subsection=subsection, brick=brick,**kw)
    

        
        
