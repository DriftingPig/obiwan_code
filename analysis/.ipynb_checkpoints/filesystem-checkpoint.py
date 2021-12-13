import os
import astropy.io.fits as fits
class LegacySimData(object):
    def __init__(self, survey_dir=None, outdir=None, subsection=None, brick=None, **kwargs):
        '''
        mainly ccopied from legacypipe/survey.py -- LegacySurveyData
        Create a LegacySurveyData object using data from the given
        *survey_dir* directory, or from the $LEGACY_SURVEY_DIR environment
        variable.

        Parameters
        ----------
        survey_dir : string
            Defaults to $LEGACY_SURVEY_DIR environment variable.  Where to look for
            files including calibration files, tables of CCDs and bricks, image data,
            etc.

        outdir : string
            setting to $obiwan_out if not specified
        '''

        if brick is not None or ~hasattr(self,'brick'):
            self.brick = brick
        if subsection is not None or ~hasattr(self,'subsection'):
            self.subsection = subsection
        if survey_dir is not None or ~hasattr(self,'survey_dir'):
            self.survey_dir = survey_dir
        if outdir is not None or ~hasattr(self,'outdir'):
            self.outdir = outdir
        
        if survey_dir is None:
            survey_dir = os.environ.get('LEGACY_SURVEY_DIR')
            if survey_dir is None:
                raise ValueError('Error: you should set the $LEGACY_SURVEY_DIR environment variable.')
        self.survey_dir = survey_dir
        
        if outdir is None:
            outdir = os.environ.get('obiwan_out')
            self.outdir = outdir
            if outdir is None:
                raise ValueError('Error: you should set the outdir variable.')
        else:
            self.outdir = outdir
        
        super().__init__(**kwargs)


    def __str__(self):
        return ('%s: dir %s, out %s' %
                (type(self).__name__, self.survey_dir, self.outdir))

    def find_file(self, filetype, brick=None, subsection=None, brickpre=None, band='%(band)s',
                  camera=None, expnum=None, ccdname=None,
                   **kwargs):
        '''
        Returns the filename of a Legacy Survey file.

        *filetype* : string, type of file to find, including:
             "tractor" -- Tractor catalogs
             "depth"   -- PSF depth maps
             "galdepth" -- Canonical galaxy depth maps
             "nexp" -- number-of-exposure maps

        *brick* : string, brick name such as "0001p000"

        *output*: True if we are about to write this file; will use self.outdir as
        the base directory rather than self.survey_dir.

        *subsection*: string, usually 'rs0', an obiwan addition  

        Returns: path to the specified file (whether or not it exists).
        '''
        from glob import glob
        if brick is None:
            if hasattr(self,'brick'):
                brick = self.brick
        if brick is None:
            brick = '%(brick)s'
            brickpre = '%(brick).3s'
        else:
            brickpre = brick[:3]
        if subsection is None:
            if hasattr(self,'subsection'):
                subsection = self.subsection

        basedir = self.survey_dir
        basedir_out = self.outdir

        
        if subsection is None:
             codir = os.path.join(basedir_out, 'coadd', brickpre, brick)
        else:
            codir = os.path.join(basedir_out, self.subsection, 'coadd',  brickpre, brick)


        sname = 'legacysurvey'

        if filetype == 'bricks':
            return os.path.join(basedir, 'survey-bricks.fits.gz')
        
        if filetype == 'sweep': 
            # the brick string for sweep file is actually its file name (because it contains multiple bricknames!) 
            return os.path.join(basedir, brick) 

        elif filetype == 'ccds':
            return glob(os.path.join(basedir, 'survey-ccds*.fits.gz'))

        elif filetype == 'ccd-kds':
            return glob(os.path.join(basedir, 'survey-ccds*.kd.fits'))

        elif filetype == 'tycho2':
            dirnm = os.environ.get('TYCHO2_KD_DIR')
            if dirnm is not None:
                fn = os.path.join(dirnm, 'tycho2.kd.fits')
                if os.path.exists(fn):
                    return fn
            return swap(os.path.join(basedir, 'tycho2.kd.fits'))

        elif filetype == 'large-galaxies':
            fn = os.environ.get('LARGEGALAXIES_CAT')
            if fn is None:
                return None
            if os.path.isfile(fn):
                return fn
            return None

        elif filetype == 'annotated-ccds':
            return glob(os.path.join(basedir, 'ccds-annotated-*.fits.gz'))

        elif filetype == 'tractor':
            if self.subsection is None:
                return os.path.join(basedir_out, 'tractor', brickpre, 
                                     'tractor-%s.fits' % brick)
            else:
                return os.path.join(basedir_out, subsection, 'tractor', brickpre, 
                                     'tractor-%s.fits' % brick)
        elif filetype == 'simcat':
            if subsection is None:
                return os.path.join(basedir_out, 'coadd', brickpre, brick, 
                                     sname+'-simcat-%s.fits' % brick)
            else:
                return os.path.join(basedir_out, subsection, 'coadd', brickpre, brick, 
                                     sname+'-simcat-%s.fits' % brick)
            
        elif filetype == 'simorigin':
            if brick is None:
                print('brick should not be None')
                return None
            fn = basedir_out.replace('output','')+'/divided_randoms'
            #fn = os.path.dirname(os.path.abspath(basedir_out))+'/divided_randoms'
            return os.path.join(fn,'brick_%s.fits' % brick)
            

        elif filetype == 'tractor-intermediate':
            if subsection is None:
                fn = os.path.join(basedir_out, 'tractor-i', brickpre, 'tractor-%s.fits' % brick)
                if os.path.isfile(fn):
                    return fn
                else:
                    return os.path.join(basedir_out, 'tractor-i', brickpre,
                                     'tractor-i-%s.fits' % brick)
            else:
                return os.path.join(basedir_out, subsection, 'tractor-i', brickpre, 
                                     'tractor-%s.fits' % brick)

        elif filetype in ['ccds-table', 'depth-table']:
            ty = filetype.split('-')[0]
            return os.path.join(codir, '%s-%s-%s.fits' % (sname, brick, ty))

        elif filetype in ['image-jpeg', 'model-jpeg', 'resid-jpeg',
                          'blobmodel-jpeg',
                          'imageblob-jpeg', 'simscoadd-jpeg','imagecoadd-jpeg',
                          'wise-jpeg', 'wisemodel-jpeg',
                          'galex-jpeg', 'galexmodel-jpeg',
                          ]:
            ty = filetype.split('-')[0]
            return os.path.join(codir, '%s-%s-%s.jpg' % (sname, brick, ty))

        elif filetype in ['outliers-pre', 'outliers-post',
                          'outliers-masked-pos', 'outliers-masked-neg']:
            if subsection is None:
                return os.path.join(basedir_out, 'metrics', brickpre,
                             '%s-%s.jpg' % (filetype, brick))
            else:
                return os.path.join(basedir_out, subsection, 'metrics', brickpre, 
                             '%s-%s.jpg' % (filetype, brick))
        #obiwan, adding 'sim-iamge', 'sim-invvar','sims'
        elif filetype in ['invvar', 'chi2', 'image', 'model', 'blobmodel',
                          'depth', 'galdepth', 'nexp', 'psfsize',
                          'copsf','sim-image','sim-invvar','sims']:
            return os.path.join(codir, '%s-%s-%s-%s.fits.fz' %
                                     (sname, brick, filetype, band))

        elif filetype in ['blobmap']:
            if self.subsection is None:
                return os.path.join(basedir_out, 'metrics', brickpre,
                                     'blobs-%s.fits.gz' % (brick))
            else:
                return os.path.join(basedir_out, subsection, 'metrics', brickpre,
                                     'blobs-%s.fits.gz' % (brick))

        elif filetype in ['maskbits']:
            return os.path.join(codir,
                                     '%s-%s-%s.fits.fz' % (sname, brick, filetype))

        elif filetype in ['all-models']:
            if self.subsection is None:
                return os.path.join(basedir_out, 'metrics', brickpre,
                                     'all-models-%s.fits' % (brick))
            else:
                return os.path.join(basedir_out, subsection, 'metrics', brickpre, 
                                     'all-models-%s.fits' % (brick))

        elif filetype == 'ref-sources':
            if self.subsection is None:
                return os.path.join(basedir_out, 'metrics', brickpre,
                                     'reference-%s.fits' % (brick))
            else:
                return os.path.join(basedir_out, subsection, 'metrics', brickpre, 
                                     'reference-%s.fits' % (brick))

        elif filetype == 'checksums':
            if self.subsection is None:
                return os.path.join(basedir_out, 'tractor', brickpre,
                                     'brick-%s.sha256sum' % brick)
            else:
                return os.path.join(basedir_out, subsection, 'tractor', brickpre, 
                                     'brick-%s.sha256sum' % brick)
        elif filetype == 'outliers_mask':
            if self.subsection is None:
                return os.path.join(basedir_out, 'metrics', brickpre,
                                     'outlier-mask-%s.fits.fz' % (brick))
            else:
                return os.path.join(basedir_out, subsection, 'metrics', brickpre, 
                                     'outlier-mask-%s.fits.fz' % (brick))
        #cosmos
        elif filetype == 'processed':
                fn = os.path.join(basedir_out, 'subset')
                if os.path.isdir(fn) is False:
                    fn = os.path.join(os.path.dirname(os.path.abspath(basedir_out)),'subset')
                fns = glob(os.path.join(fn,'*'))
                if len(fns)==1:
                    fns = fns[0]
                return fns
        elif filetype == 'processed_one': #made a new naming scheme
            if self.subsection is None:
                print('subsection need to be specified')
                return None
            else:
                fn = os.path.join(basedir_out, 'subset')
                if os.path.isdir(fn) is False:
                    fn = os.path.join(os.path.dirname(os.path.abspath(basedir_out)),'subset','subset_%s.fits'%subsection)
                return fn
        elif filetype == 'processed_one_rsp':
            fn = os.path.join(basedir_out, 'subset')
            if os.path.isdir(fn) is False:
                    fn = os.path.join(os.path.dirname(os.path.abspath(basedir_out)),'subset','subset_%s_rsp.fits'%subsection)
            return fn
            
        elif filetype == 'masks':
                #setting masks the same, made by matching all the depth plots, keeping non-zeros pixels across all sets    
                fn = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_subsets/maskbits/%s.fits'%self.brick
                return fn
        elif filetype == 'maskbits_cross':
            fn = os.path.join(basedir_out,'maskbits','%s.fits'%self.brick)
            return fn
        elif filetype == 'cosmos_deep':
            fn = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_deep/truth.fits'
            return fn
        elif filetype == 'cosmos_deep_dr9':
            fn = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/cosmos_deep/dr9_mirror.fits'
            return fn
        elif filetype == 'random':
            fn = os.path.join(os.path.dirname(os.path.abspath(basedir_out)), 'randoms', 'randoms.fits')
          
            return fn
        print('Unknown filetype "%s"' % filetype)
        assert(False)


    def get_calib_dir(self):
        '''
        Returns the directory containing calibration data.
        '''
        return os.path.join(self.outdir, 'calib')

    def get_image_dir(self):
        '''
        Returns the directory containing image data.
        '''
        return os.path.join(self.outdir, 'images')

    def get_survey_dir(self):
        '''
        Returns the base LEGACY_SURVEY_DIR directory.
        '''
        return self.survey_dir
    
    def get_model(self,band):
        self.model_fn = self.find_file('model',brick = self.brick, subsection = self.subsection, band=band)
        self.model = fits.getdata(self.model_fn)
    def get_image(self,band):
        self.image_fn = self.find_file('image',brick = self.brick, subsection = self.subsection, band=band)
        self.image = fits.getdata(self.image_fn)
    def get_sims(self,band):
        try:
            self.sim_fn = self.find_file('sims',brick = self.brick, subsection = self.subsection, band=band)
        except:
            self.sim_fn = self.find_file('sim-image',brick = self.brick, subsection = self.subsection, band=band)
        self.sim = fits.getdata(self.sim_fn)
    def get_blobmap(self):
        self.blobmap_fn = self.find_file('blobmap',brick = self.brick, subsection = self.subsection)
        self.blobmap = fits.getdata(self.blobmap_fn)
    def get_tractor(self,brick=None,subsection=None):
        if brick is not None:
            self.brick = brick
        if self.subsection is not None:
            self.subsection = subsection
        self.tractor_fn = self.find_file('tractor',brick = self.brick, subsection = self.subsection)
        self.tractor = fits.getdata(self.tractor_fn)
    def get_processed(self,brick=None,subsection=None):
        if brick is not None:
            self.brick = brick
        if self.subsection is not None:
            self.subsection = subsection
        self.processed_fn = self.find_file('processed',brick = self.brick, subsection = self.subsection)
        self.processed = fits.getdata(self.processed_fn)
    def get_mask(self,brick=None,subsection=None):
        #used ony for cosmos
        if brick is not None:
            self.brick = brick
        else:
            assert(False)
        self.mask_fn = self.find_file('masks')
        self.mask = fits.getdata(self.mask_fn)
    def get_maskbits_corss(self,brick=None,startid=None):
        if brick is not None:
            self.brick = brick
        else:
            assert(False)
        maskbits_fn = self.find_file('maskbits_cross')
        maskbitsdir = os.path.dirname(maskbits_fn)
        maskbits_fn = maskbitsdir+'/%s_rs%d.fits'%(brick,startid)
        self.maskbits_cross = fits.getdata(maskbits_fn)
        