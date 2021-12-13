from filesystem import LegacySimData
import astropy.io.fits as fits
from MakePlots import BasePlot
from SurveySource import BaseSource, BaseSourceBase
import matplotlib.pyplot as plt
class BrickViewer(LegacySimData,BasePlot):
	def __init__(self, survey_dir=None, outdir=None, subsection=None, brick=None, **kwargs):
		super(BrickViewer,self).__init__(survey_dir=survey_dir, outdir=outdir, subsection=subsection, brick=brick, **kwargs)
		self.blobmap_fn = self.find_file('blobmap',brick=self.brick,subsection=self.subsection)
		self.loadimg(self.blobmap_fn)

	def plot_src_over_blobmap(self,bx_cen=None, by_cen=None, radius_x=None, radius_y=None, wise=False, marker='ro'):
		self.src_fn = self.find_file('tractor-intermediate', brick=self.brick, subsection = self.subsection)
		src = fits.getdata(self.src_fn)
		self.src = src
		if bx_cen is not None:
			self.bx_cen = bx_cen
		if by_cen is not None:
			self.by_cen = by_cen
		if radius_x is not None:
			self.radius_x = radius_x
		if radius_y is not None:
			self.radius_y = radius_y
		keys = ['bx_cen','by_cen','radius_y','radius_x']
		for key in keys:
			assert(hasattr(self,key))
		if wise:
			xx = src['wise_x']
			yy = src['wise_y']
		else:
			xx = src['bx']
			yy = src['by']
		sel = (xx>self.bx_cen-self.radius_x)&(xx<self.bx_cen+self.radius_x)&(yy>self.by_cen-self.radius_y)&(yy<self.by_cen+self.radius_y)
		print("%d sources in total"%sel.sum())
		if wise:
			plt.plot(src['wise_x'][sel]-self.bx_cen+self.radius_x,src['wise_y'][sel]-self.by_cen+self.radius_y,maker)
		else:
			plt.plot(src['bx'][sel]-self.bx_cen+self.radius_x,src['by'][sel]-self.by_cen+self.radius_y,marker)

	def plot_img(self, img, band, bx_cen=None, by_cen=None, radius_x=None, radius_y=None, vmax=None,vmin=None):
		if bx_cen is not None:
			self.bx_cen = bx_cen
		if by_cen is not None:
			self.by_cen = by_cen
		if radius_x is not None:
			self.radius_x = radius_x
		if radius_y is not None:
			self.radius_y = radius_y
		keys = ['bx_cen','by_cen','radius_y','radius_x']
		for key in keys:
			assert(hasattr(self,key))
		by1 = self.by_cen - self.radius_y
		by2 = self.by_cen + self.radius_y
		bx1 = self.bx_cen - self.radius_x
		bx2 = self.bx_cen + self.radius_x
		img = img[by1:by2,bx1:bx2]
		self.imshow(img=img,vmax=vmax,vmin=vmin)
		return img

	def plot_radec(self,file_type='tractor',wise=False,marker='r.'):
		self.get_tractor()
		radec = self.tractor
		if hasattr(self,'bx_cen'):
			if wise:
				xx = radec['wise_x']
				yy = radec['wise_x']
			else:
				xx = radec['bx']
				yy = radec['by']
			sel = (xx>self.bx_cen-self.radius_x)&(xx<self.bx_cen+self.radius_x)&(yy>self.by_cen-self.radius_y)&(yy<self.by_cen+self.radius_y)
		else:
			sel=True
		plt.plot(radec['ra'][sel],radec['dec'][sel],marker)
	def init_bxy(self,bx,by,radius=10):
		self.bx = bx
		self.bx_cen = bx       
		self.by = by
		self.by_cen = by
		self.radius_x = radius
		self.radius_y = radius
        
        
class StampViewer(BrickViewer, BaseSource):
    def __init__(self, filetype='tractor', survey_dir=None, outdir=None, subsection=None, brick=None, idx = None, **kwargs):
        super(StampViewer,self).__init__(survey_dir=survey_dir, outdir=outdir, subsection=subsection, brick=brick, **kwargs)
        #super(BaseSource,self).__init__(filetype=filetype, brick=brick, subsection=subsection, survey_dir=survey_dir, outdir=outdir,**kw)
        if idx is not None:
            self.idx = idx
        else:
            print('idx is not set, default to 0')
            self.idx = 0
    def init_bxy_sim(self,idx=None,radius=15,wise=False):
        if idx is None:
            print('setting idx to 0')
            self.sim_idx = 0
        else:
            self.sim_idx = idx
        matched_source = BaseSourceBase(self.find_file('processed'))
 
        sel = (matched_source.brickname == self.brick)&(matched_source.detected)
        if wise:
            bx = np.array(matched_source.wise_x[sel])
            by = np.array(matched_source.wise_y[sel])
        else:
            bx = matched_source.bx[sel]
            by = matched_source.by[sel]
        self.bx_cen = int(bx[self.sim_idx]+0.5)
        self.by_cen = int(by[self.sim_idx]+0.5)
        self.radius_x = int(radius+0.5)
        self.radius_y = int(radius+0.5)
        
    def init_bxy_idx(self,idx=None,radius=15,wise=False):
        self.SingleSource(idx)
        if wise:
            self.bx_cen = int(self.wise_x+0.5)
            self.by_cen = int(self.wise_x+0.5)
            self.radius_x = radius
            self.radius_y = radius
        else:
            self.bx_cen = int(self.bx+0.5)
            self.by_cen = int(self.by+0.5)
            self.radius_x = radius
            self.radius_y = radius
        
    def plot_stamps_w_dr9(self,band, vmax=None,vmin=None,region=None):
        plt.figure(figsize = (8,10))
        plt.subplot(3,2,1)
        self.get_image(band)
        dat1 = self.plot_img(img=self.image,band = band, vmax=vmax,vmin=vmin)
        plt.xlabel('real image(%.4f)'%dat1.sum())

        plt.subplot(3,2,2)
        self.get_model(band)
        dat2 = self.plot_img(img=self.model, band = band, vmax=vmax,vmin=vmin)
        plt.xlabel('model image(%.4f)'%dat2.sum())

        plt.subplot(3,2,3)
        self.get_sims(band)
        dat3 = self.plot_img(img=self.sim, band = band, vmax=vmax,vmin=vmin)
        plt.xlabel('sim image(%.4f)'%dat3.sum())
        
        plt.subplot(3,2,4)
        dat4 = dat1-dat3
        plt.imshow(dat4,vmax=vmax,vmin=vmin)
        plt.xlabel('real-sim image(%.4f)'%dat4.sum())

        plt.subplot(3,2,5)
        dr9_dat = BrickViewerMatch(catalog=self,option='dr9',region=region)
        dr9_dat.match_bxy()
        dr9_dat.get_image(band)
        dat5 = dr9_dat.plot_img(img=dr9_dat.image, band = band, vmax=vmax,vmin=vmin)
        plt.imshow(dat5)
        plt.xlabel('original image(%.4f)'%dat5.sum())

        plt.subplot(3,2,6)
        dr9_dat.get_model(band)
        dat6 = dr9_dat.plot_img(img=dr9_dat.model, band = band,vmax=vmax,vmin=vmin)
        plt.imshow(dat6,vmax=vmax,vmin=vmin)
        plt.xlabel('tractor model image(%.4f)'%dat6.sum())

        
        
class BrickViewerMatch(BrickViewer):
    '''
    init a BrickView object with a matched catalog for comparison
    the catalog is a BrickViewer object that will get matched
    '''
    def __init__(self,catalog,option,region=None, **kw):
        if option == 'dr9':
            self.region = region
            if self.region is None:
                self.region = 'south'
                print('default region to dr9 south')
            self.match_dr9(region=self.region)
        self.brick = catalog.brick
        self.catalog = catalog
        super(BrickViewerMatch,self).__init__(brick=self.brick, subsection=self.subsection, survey_dir=self.survey_dir, outdir=self.outdir, **kw)
    
    def match_dr9(self,region='south'):
        self.region = region
        self.survey_dir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr9/'
        self.outdir = self.survey_dir+self.region
        self.subsection = None
        
    def match_bxy(self):
        self.bx_cen = self.catalog.bx_cen
        self.by_cen = self.catalog.by_cen
        self.radius_x = self.catalog.radius_x
        self.radius_y = self.catalog.radius_y
        
        
