import numpy as np
from legacypipe.survey import RexGalaxy, GaiaSource
from tractor.galaxy import ExpGalaxy, DevGalaxy
from tractor import PointSource
from tractor.sersic import SersicGalaxy, SersicIndex
import tractor.ellipses as ellipses
from legacypipe.survey import LogRadius
from legacypipe.survey import LegacySersicIndex
import tractor
from tractor import RaDecPos,NanoMaggies
import galsim
from astrometry.util.fits import fits_table
from tractor import *

#for sources inside the blob, we mask out using blobmap, keeping unmasked blobs
#sources outside maps are ref stars. We check if it has ref id in the catalog that we are interested, if not, we would add these stars
def reprocess_wcat(wcat, brickname,blobmap,maskbits,T,survey):
	dataset = survey.dataset
	if dataset == 'dr9':
		print('reprocessing wise src from dr9')
		fn_dr9_tractor = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/tractor-i/'+brickname[:3]+'/tractor-i-'+brickname+'.fits'
		dr9_tractor = fits_table(fn_dr9_tractor)
	if dataset == 'cosmos':
		print('reprocessing wise src from cosmos') 
		fn_dr9_tractor = '/global/cscratch1/sd/dstn/dr9-cosmos-subs/%s/tractor-i/%s/tractor-i-%s.fits'%(survey.cosmos_num,brickname[:3],brickname)
		dr9_tractor = fits_table(fn_dr9_tractor)

	#confirm on the code where I use maskbits to maskout cosmos repeats
	target_x = (dr9_tractor.bx+0.5).astype(int)
	target_y = (dr9_tractor.by+0.5).astype(int)
	infield = (target_x>=0)&(target_x<3600)&(target_y>=0)&(target_y<3600)
	target_x = np.clip(target_x,0,3599)
	target_y = np.clip(target_y,0,3599)
	targetxy = (target_y,target_x)
	outblob = (blobmap[targetxy]==-1)
	outblob[~infield] = True
	dr9_ref_id = dr9_tractor.ref_id
	idx_refstar = np.where(dr9_ref_id!=0)[0]
	dup_ref_star = []
	T_ref_id = T.ref_id[T.ref_id!=0]
	nkeep=0
	for idx in idx_refstar:
		if dr9_ref_id[idx] in T_ref_id:
			dup_ref_star.append(idx)
			print("deleting ref id %d"%dr9_ref_id[idx])
		else:
			nkeep+=1
	print("keeping %d stars added"%nkeep)
	#delete ref star that's already inside
	if len(dup_ref_star)>0:
		outblob[np.array(dup_ref_star)] = False
	incluster = dr9_tractor.iscluster
	idx = np.where(outblob&~incluster)[0]

	add_source = dr9_tractor[idx]
	for i in range(len(add_source)):
		ra = add_source.ra[i]
		dec = add_source.dec[i]
		brightness =  tractor.NanoMaggies(g=add_source.flux[i][0],r=add_source.flux[i][1],z=add_source.flux[i][2], order=['g','r','z'])
		r_half = np.clip(add_source.shape_r[i],0.001, 1e6)
		shape = ellipses.EllipseE(add_source.shape_r[i],add_source.shape_e1[i],add_source.shape_e2[i])
		if add_source.type[i] == 'DUP':
			#gaia source Can I simply set it as point source? --yes
			cat_i = GaiaSource(RaDecPos(ra,dec),brightness)
		if add_source.type[i] == 'PSF':
			cat_i = PointSource(RaDecPos(ra,dec),brightness)
		if add_source.type[i] == 'REX':
			cat_i = RexGalaxy(RaDecPos(ra,dec), brightness, LogRadius(np.log(r_half)))
		if add_source.type[i] == 'EXP':
			cat_i = ExpGalaxy(RaDecPos(ra,dec),brightness,shape)
		if add_source.type[i] == 'DEV':
			cat_i = DevGalaxy(RaDecPos(ra,dec),brightness,shape)
		if add_source.type[i] == 'SER':
			n = add_source.sersic[i]
			cat_i = SersicGalaxy(RaDecPos(ra,dec),brightness,shape, LegacySersicIndex(n))
		cat_i.setBrightness(NanoMaggies(w=1.))
		wcat.append(cat_i)
	N = len(add_source)
	'''
	#add the remaining stars from metrics, not used because I confirmed that additional metrics stars are also not used in all-blob mode
	metrics = fits_table('/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/metrics/'+brickname[:3]+'/reference-'+brickname+'.fits')
	dr9_ref_id = dr9_tractor.ref_id[dr9_tractor.ref_id!=0]
	metrics_ref_id = metrics.ref_id
	n_add = 0
	for i in range(len(metrics)):
		if metrics_ref_id[i] in dr9_ref_id:
			pass
		else:
			n_add+=1
			brightness = tractor.NanoMaggies(g=0,r=0,z=0, order=['g','r','z'])
			ra = metrics.ra[i]
			dec = metrics.dec[i]
			cat_i = GaiaSource(RaDecPos(ra,dec),brightness)
			cat_i.setBrightness(NanoMaggies(w=1.))
			wcat.append(cat_i)
	print("adding %d stars from metrics"%n_add)
	N+=n_add
	'''
	return wcat,N


class BuildStamp_wise():
    """
    wise version of buildstamp
    """
    def __init__(self, tim, band, targetwcs):
        self.targetwcs = targetwcs
        self.band = band
        self.nano2e = get_nano2e("w%s"%self.band)
        self.tim = tim
    def setlocal(self, obj):
        ra = obj.ra
        dec = obj.dec
        flag, xx,yy = self.targetwcs.radec2pixelxy(ra,dec)
        x = int(xx-1+0.5)
        y = int(yy-1+0.5)

        self.target_x=x
        self.target_y=y
        flag, xx, yy = self.tim.wcs.wcs.radec2pixelxy(*(self.targetwcs.pixelxy2radec(x+1,y+1)[-2:]))
        x_cen = xx-1
        y_cen = yy-1
        self.wcs=self.tim.getWcs().wcs
        x_cen_int,y_cen_int = round(x_cen),round(y_cen)
        self.sx0,self.sx1,self.sy0,self.sy1 = x_cen_int-32,x_cen_int+31,y_cen_int-32,y_cen_int+31
        (h,w) = self.tim.shape
        self.sx0 = np.clip(int(self.sx0), 0, w-1)
        self.sx1 = np.clip(int(self.sx1), 0, w-1) + 1
        self.sy0 = np.clip(int(self.sy0), 0, h-1)
        self.sy1 = np.clip(int(self.sy1), 0, h-1) + 1
        subslc = slice(self.sy0,self.sy1),slice(self.sx0,self.sx1)
        subimg = self.tim.getImage ()[subslc]
        subie  = self.tim.getInvError()[subslc]
        subwcs = self.tim.getWcs().shifted(self.sx0, self.sy0)
        subsky = self.tim.getSky().shifted(self.sx0, self.sy0)
        subpsf = self.tim.psf.constantPsfAt((self.sx0+self.sx1)/2., (self.sy0+self.sy1)/2.)
        new_tim = tractor.Image(data=subimg, inverr=subie, wcs=subwcs,psf=subpsf, photocal=self.tim.getPhotoCal(), sky=subsky, name=self.tim.name)
        return new_tim
    def galaxy(self, obj,bandname,fluxfactor,band):
        new_tim = self.setlocal(obj)
        n,ra,dec,r_half,e1,e2,flux, mw_transmission = int(obj.n),float(obj.ra),float(obj.dec),float(obj.rhalf),float(obj.e1),float(obj.e2), float(obj.get('w%sflux'%band)), float(obj.get('mw_transmission_w%s'%band))
        flux = fluxfactor*flux 
        print("Drawing galaxies on wise %s flux=%.2f n=%.2f e1=%.2f e2=%.2f" % \
                (bandname, flux, n, e1, e2))
        brightness =  tractor.NanoMaggies(w=flux, order=['w'])
        
        shape = ellipses.EllipseE(r_half,e1,e2)
        
        if n==0: 
             new_gal = PointSource(RaDecPos(ra,dec),brightness)
        if n==1:
             new_gal = ExpGalaxy(RaDecPos(ra,dec),brightness,shape)
        elif n==4:
             new_gal = DevGalaxy(RaDecPos(ra,dec),brightness,shape)
        else:
             new_gal = SersicGalaxy(RaDecPos(ra,dec),brightness,shape, LegacySersicIndex(n))
        new_tractor = Tractor([new_tim], [new_gal])
        mod0 = new_tractor.getModelImage(0)
        galsim_img = galsim.Image(mod0)
        galsim_img.bounds.xmin=self.sx0+1
        galsim_img.bounds.xmax=self.sx1-1
        galsim_img.bounds.ymin=self.sy0+1
        galsim_img.bounds.ymax=self.sy1-1

        return galsim_img        

def noise_for_galaxy(gal,nano2e,nims):
    """Returns numpy array of noise in Img count units for gal in image cnt units"""
    # Noise model + no negative image vals when compute noise
    one_std_per_pix= gal.array.copy() # nanomaggies
    one_std_per_pix[one_std_per_pix < 0]=0
    # rescale
    one_std_per_pix *= nano2e*nims.array.copy() # e-
    one_std_per_pix= np.sqrt(one_std_per_pix)
    num_stds= np.random.randn(one_std_per_pix.shape[0],one_std_per_pix.shape[1])
    #one_std_per_pix.shape, num_stds.shape
    noise= one_std_per_pix * num_stds
    # rescale
    nims_down = nims.copy()
    nims_down.fill(np.clip(nims_down.array,0.01,nims.array.max()))
    noise /= nano2e*nims_down.array.copy() #nanomaggies
    return noise

def ivar_for_galaxy(gal,nano2e,nims,band):
    """Adds gaussian noise to perfect source
    Args:
        gal: galsim.Image() for source, UNITS: nanomags
        nano2e: factor to convert to e- (gal * nano2e has units e-)
    Returns:
        galsim.Image() of invvar for the source, UNITS: nanomags
    """
    
    var= gal.copy() * nano2e*nims #e^2
    var.applyNonlinearity(np.abs)
    nims_down = nims.copy()
    nims_down.fill(np.clip(nims_down.array,0.01,nims.array.max()))
    var /= (nano2e*nims_down)**2 #nanomag^2
    poissons = {1: 0.15, 2: 0.3}
    #campatible to https://github.com/legacysurvey/legacypipe/blob/main/py/legacypipe/unwise.py L157
    #var+=gal.copy() * poissons[band]**2
    var.invertSelf()
    return var

def get_srcimg_invvar(stamp_ivar,img_ivar):
    """stamp_ivar, img_ivar -- galsim Image objects"""
    # Use img_ivar when stamp_ivar == 0, both otherwise
    return img_ivar
    use_img_ivar= np.ones(img_ivar.array.shape).astype(bool)
    use_img_ivar[ stamp_ivar.array > 0 ] = False
    # First compute using both
    ivar= np.power(stamp_ivar.array.copy(), -1) + np.power(img_ivar.array.copy(), -1)
    ivar= np.power(ivar,-1)
    keep= np.ones(ivar.shape).astype(bool)
    keep[ (stamp_ivar.array > 0)*\
          (img_ivar.array > 0) ] = False
    ivar[keep] = 0.
    # Now use img_ivar only where need to
    ivar[ use_img_ivar ] = img_ivar.array.copy()[ use_img_ivar ]
    # return
    obj_ivar = stamp_ivar.copy()
    obj_ivar.fill(0.)
    obj_ivar+= ivar
    return obj_ivar

def get_nano2e(bandname):

    vega_to_dn = {"w1":4.93,"w2":13.8}
    gain = {"w1":3.20,"w2":3.83}
    nano2e = 1/vega_to_dn[bandname]*gain[bandname]
    return nano2e

def get_nano2e_v2(bandname):
    '''
    These are numbers found on these websites, the numbers are close to the ones provided by Aaron, but not exactely the same
    test function based on:
    https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2ab
    https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4a.html#errmod
    '''
    # the nano2e I derived is 1e-9*f_0/c*g
    #f_0,c taken from https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#tbl1
    #vega to jy
    f_0 = dict(w1=306.682,
               w2=170.663,
               w3=29.0448,
               w4=8.2839)
    #jy to dn 
    c = dict(w1=1.9350e-6,
             w2=2.7048e-6,
             w3=1.832e-6,
             w4=5.2269e-5)
    #gain, taken from https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4a.html#errmod
    #dn to e-
    g = dict(w1=3.2,
             w2=3.83,
             w3=6.83,
             w4=24.5)

    nano2e = 1e-9*f_0[bandname]/c[bandname]*g[bandname]
    return nano2e


def add_source_to_tractor_tim(tim,band,simcat,targetwcs):
    #this is the kenobi for wise
    #make environment variable for brickname and surveydir for use here...
    import os
    if simcat is None:
        print('simcat is None, returning to original image')
        return tim
    if tim is None:
        return tim
    if band!=1 and band!=2:
        return tim
    #obiwan: shooting image to 0
    #tim.data = np.zeros(tim.data.shape)

    # Convert WISE fluxes from AB to Vega.
    #copied from https://github.com/legacysurvey/legacypipe/blob/master/py/legacypipe/runbrick.py#L2810
    vega_to_ab = dict(w1=2.699,
                          w2=3.339,
                          w3=5.174,
                          w4=6.620)
    bandname = 'w%d'%band
    
    dm = -vega_to_ab[bandname] #as we need ab to vega instead, thus a negative sign
    fluxfactor = 10.** (dm / -2.5)

    # Grab the data and inverse variance images [nanomaggies!]
    tim_image = galsim.Image(tim.getImage())
    tim_invvar = galsim.Image(tim.getInvvar())
    good = (tim.nims > 0)
    tim_dq = galsim.Image(np.array(good,dtype=np.int))
    tim_nims = galsim.Image(tim.nims).copy()
    # Also store galaxy sims and sims invvar
    sims_image = tim_image.copy()
    sims_image.fill(0.0)
    sims_ivar = sims_image.copy()
    
    
    objstamp = BuildStamp_wise(tim,band,targetwcs=targetwcs)
    #question:how do I add the sim noise?
    add_sim_noise =True
    for cat_i in simcat:
                stamp = objstamp.galaxy(cat_i,bandname,fluxfactor,band)
                # Add source if EVEN 1 pix falls on the CCD
                overlap = stamp.bounds & tim_image.bounds
                if overlap.area() > 0:
                    stamp = stamp[overlap]
                    ivarstamp= ivar_for_galaxy(stamp,objstamp.nano2e,tim_nims[overlap],band)
                    if add_sim_noise:
                          stamp += noise_for_galaxy(stamp[overlap],objstamp.nano2e,tim_nims[overlap])
                    print('Stamp overlaps tim: band=%s' % (objstamp.band))
                    stamp = stamp[overlap]
                    ivarstamp = ivarstamp[overlap]

                    # Zero out invvar where bad pixel mask is flagged (> 0)
                    #dq for wise image is set as at least 1 image touches with the pixel
                    keep = np.ones(tim_dq[overlap].array.shape)
                    #keep[ tim_dq[overlap].array > 0 ] = 0.
                    ivarstamp *= keep
                    # Add stamp to image
                    back= tim_image[overlap].copy()
                    tim_image[overlap] += stamp 
                    back_ivar= tim_invvar[overlap].copy()
                    tot_ivar= get_srcimg_invvar(ivarstamp, back_ivar)
                    tim_invvar[overlap] = tot_ivar.copy()

                    #Extra
                    sims_image[overlap] += stamp.copy()
                    sims_ivar[overlap] += ivarstamp.copy()
    tim.data = tim_image.array
    tim.inverr = np.sqrt(tim_invvar.array)
    tim.sim_image = sims_image.array
    tim.sim_image_ivar = sims_ivar.array
    return tim