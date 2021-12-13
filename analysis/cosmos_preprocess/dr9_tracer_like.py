import numpy as np
import astropy.io.fits as fits
import glob
import multiprocessing as mp
from astropy.table import Table,vstack
#also selecting dec>-30 for decals south only
region = 'south'
topdir = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/'+region+'/sweep/9.0/'
fns = glob.glob(topdir+'sweep*')
sweep_keys= ['BRICKNAME','RA','DEC','TYPE','OBJID','EBV','FLUX_G','FLUX_R','FLUX_Z','FLUX_W1','FLUX_W2','FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z','MW_TRANSMISSION_W1','MW_TRANSMISSION_W2','NOBS_G','NOBS_R','NOBS_Z','NOBS_W1','NOBS_W2','SHAPE_R','SHAPE_E1','SHAPE_E2','FIBERFLUX_G','FIBERFLUX_R','FIBERFLUX_Z','MASKBITS','SERSIC','DCHISQ','PSFSIZE_G','PSFSIZE_R','PSFSIZE_Z','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z','GALDEPTH_G','GALDEPTH_R','GALDEPTH_Z','WISEMASK_W1','WISEMASK_W2','ANYMASK_G','ANYMASK_R','ANYMASK_Z','GAIA_PHOT_G_MEAN_MAG','PSFDEPTH_W1','PSFDEPTH_W2']


class isLRG_like(object):
    def __init__(self,method):
        exec("self.%s()"%method)
    def LRG_basline_like(self):#1.4% LRG lost rate, south
        self.cut1 = 'zmag - w1mag > 0.8 * (rmag-zmag) - 0.6'
        self.cut2 = 'zfibermag < 21.5+0.13'
        self.cut3 = 'rmag - w1mag > 1.1-0.04'
        self.cut4 = 'rmag - w1mag > (w1mag - 17.22) * 1.8 -0.42'
        self.cut5 = 'rmag - w1mag > w1mag - 16.37-0.18'
        self.structure = '(%s)&(%s)&(%s)&(%s)&(%s)'%(self.cut1, self.cut2, self.cut3, self.cut4, self.cut5)
    def LRG_sv3_like(self):#south 
        #adding 0.01 to 0s
        self.cut1 = 'zmag - w1mag > 0.8 * (rmag - zmag) - 0.6-0.01'
        self.cut2 = 'zfibermag < 21.7+0.32'
        self.cut3_1 = 'gmag - rmag > 1.3-0.45'
        self.cut3_2 = '(gmag - rmag) > -1.55 * (rmag - w1mag)+3.13-0.13'
        self.cut3_3 = 'rmag - w1mag > 1.8-0.37'
        self.cut4_1 ='rmag - w1mag > (w1mag - 17.26) * 1.8-0.46'
        self.cut4_2 = 'rmag - w1mag > (w1mag - 16.36) * 1.-0.88'
        self.cut4_3 = 'rmag - w1mag > 3.29-1.65'
        self.structure = '(%s)&(%s)&((%s)&(%s)|(%s))&((%s)&(%s)|(%s))'%(self.cut1, self.cut2, self.cut3_1, self.cut3_2, self.cut3_3, self.cut4_1, self.cut4_2,  self.cut4_3)
    def LRG_sv3(self):
        self.cut1 = 'zmag - w1mag > 0.8 * (rmag - zmag) - 0.6'
        self.cut2 = 'zfibermag < 21.7'
        self.cut3_1 = 'gmag - rmag > 1.3'
        self.cut3_2 = '(gmag - rmag) > -1.55 * (rmag - w1mag)+3.13'
        self.cut3_3 = 'rmag - w1mag > 1.8'
        self.cut4_1 ='rmag - w1mag > (w1mag - 17.26) * 1.8'
        self.cut4_2 = 'rmag - w1mag > (w1mag - 16.36) * 1.'
        self.cut4_3 = 'rmag - w1mag > 3.29'
        self.structure = '(%s)&(%s)&((%s)&(%s)|(%s))&((%s)&(%s)|(%s))'%(self.cut1, self.cut2, self.cut3_1, self.cut3_2, self.cut3_3, self.cut4_1, self.cut4_2,  self.cut4_3)
    def isLRGlike_color(self,data):
        #assume the data are sweep files, things are in upper case
        rflux = data['FLUX_R']/data['MW_TRANSMISSION_R']
        zflux = data['FLUX_Z']/data['MW_TRANSMISSION_Z']
        w1flux = data['FLUX_W1']/data['MW_TRANSMISSION_W1']
        zfiberflux=data['FIBERFLUX_Z']/data['MW_TRANSMISSION_Z']
        gnobs = data['NOBS_G']
        rnobs = data['NOBS_R']
        znobs = data['NOBS_Z']
        rfluxivar=data['FLUX_IVAR_R']
        zfluxivar=data['FLUX_IVAR_Z']
        w1fluxivar=data['FLUX_IVAR_W1']
        gaiagmag=data['GAIA_PHOT_G_MEAN_MAG']
        maskbits = data['MASKBITS']
        zfibertotflux = data['FIBERTOTFLUX_Z']/data['MW_TRANSMISSION_Z']
        wisemask = data['WISEMASK_W1']
        R = locals()
        R.pop('self');R.pop('data')
        lrg = self.notinLRG_mask(primary=None, **R)
        assert(hasattr(self,'structure'))
        
        gflux = data['FLUX_G']/data['MW_TRANSMISSION_G']
        zmag =  22.5 - 2.5 * np.log10(zflux)
        w1mag = 22.5 - 2.5 * np.log10(w1flux)
        rmag  = 22.5 - 2.5 * np.log10(rflux)
        zfibermag = 22.5 - 2.5 * np.log10(zfiberflux)
        gmag = 22.5 - 2.5 * np.log10(gflux)

        lrg &= eval(self.structure)
        return lrg
    @staticmethod
    def notinLRG_mask(primary=None, rflux=None, zflux=None, w1flux=None,
                  zfiberflux=None, gnobs=None, rnobs=None, znobs=None,
                  rfluxivar=None, zfluxivar=None, w1fluxivar=None,
                  gaiagmag=None, maskbits=None, zfibertotflux=None,wisemask=None):
        """See :func:`~desitarget.cuts.isLRG` for details.
        Returns
        -------
        :class:`array_like`
            ``True`` if and only if the object is NOT masked for poor quality.
        """
        if primary is None:
            primary = np.ones_like(rflux, dtype='?')
        lrg = primary.copy()

        # ADM to maintain backwards-compatibility with mocks.
        if zfiberflux is None:
            log.warning('Setting zfiberflux to zflux!!!')
            zfiberflux = zflux.copy()

        lrg &= (rfluxivar > 0) & (rflux > 0)   # ADM quality in r.
        lrg &= (zfluxivar > 0) & (zflux > 0) & (zfiberflux > 0)   # ADM quality in z.
        lrg &= (w1fluxivar > 0) & (w1flux > 0)  # ADM quality in W1.

        lrg &= (gaiagmag == 0) | (gaiagmag > 18)  # remove bright GAIA sources

        # ADM remove stars with zfibertot < 17.5 that are missing from GAIA.
        lrg &= zfibertotflux < 10**(-0.4*(17.5-22.5))

        # ADM observed in every band.
        lrg &= (gnobs > 0) & (rnobs > 0) & (znobs > 0)

        # ADM default mask bits from the Legacy Surveys not set.
        lrg &= (maskbits==0)
    
        lrg &= (wisemask==0)

        return lrg


def process_one_set(X):
    (fns_i,set_num,rank,savedir,obj)=X
    cut_method = isLRG_like(obj)
    idx=0
    for fn in fns_i:
        data = Table.read(fn)
        sel = cut_method.isLRGlike_color(data)
        source = Table()
        for key in sweep_keys:
            source[key]=data[sel][key]
        source.write(savedir+'/desi_source_rank%d_set%d_%d.fits'%(rank,set_num,idx),overwrite=True)
        idx+=1
        del(data)
        del(source)
        
    

def collect(savedir, obj='LRG_sv3_like', threads=16):
    from mpi4py import MPI
    node_tot = MPI.COMM_WORLD.Get_size()
    print("total size:%d"%node_tot)
    rank = MPI.COMM_WORLD.Get_rank()
    if rank==0:
        print("last time it took 8 thread space to avoid memory failure")
    node_sub_fns = np.array_split(fns,node_tot)
    this_sub_fn = node_sub_fns[rank]
    sub_fns = np.array_split(this_sub_fn,threads)
    commands=[]
    print("running on rank %d"%rank)
    for i in range(threads):
        commands.append((sub_fns[i],i,rank, savedir,obj))
    with mp.Pool(threads) as p:
        p.map(process_one_set,commands)
        
def get_parser():
    import argparse
    de = ('a pipeline')

    ep = '''
    collect sources from costom defined colorcuts from dr9
    '''
    parser = argparse.ArgumentParser(description=de,epilog=ep)

    parser.add_argument('--collect', default=False,
                        action='store_true', help='collect expected targets from dr9 sweep catalog')
    parser.add_argument('--stack', default=False,
                        action='store_true', help='stack all the sources into a few chunks, chun_num=node_num')
    parser.add_argument('--depthcut', default=False,
                        action='store_true', help='do depth cut for stacked files')
    parser.add_argument('--stack2', default=False,
                        action='store_true', help='stack all the sources into a few chunks, chun_num=node_num')
    parser.add_argument('--threads', type=int, default=16, help='Run multi-threaded')
    parser.add_argument('--obj', type=str, default='LRG_sv3_like', help='type of object to run')
    return parser

def stack(savedir):
    from mpi4py import MPI
    node_tot = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    print(savedir)
    stack_fns = glob.glob(savedir+'/desi_source_*')
    node_sub_fns = np.array_split(stack_fns,node_tot)
    this_sub_fn = node_sub_fns[rank]
    print("this is rank %d, file num %d"%(rank,len(this_sub_fn)))
    dat = None
    N=len(this_sub_fn)
    for i in range(len(this_sub_fn)):
        tab_i = Table.read(this_sub_fn[i])
        print("%d/%d"%(i,N))
        if dat is not None:
             dat = vstack((dat, tab_i)) 
        else:
             dat = tab_i
    dat.write(savedir+"source_chunk%d.fits"%(rank),overwrite=True)

def depthcut(savedir):
    depth_g = 24.6
    depth_r = 24.6
    depth_z = 23.6
    depth_w1 = 21.55
    dec = -30
    chunk_fns = glob.glob(savedir+'source_chunk*')
    for i in range(len(chunk_fns)):
        dat = Table.read(chunk_fns[i])
        mag_z = -2.5*(np.log10(5./np.sqrt(dat['PSFDEPTH_Z']))-9)
        mag_r = -2.5*(np.log10(5./np.sqrt(dat['PSFDEPTH_R']))-9)
        mag_g = -2.5*(np.log10(5./np.sqrt(dat['PSFDEPTH_G']))-9)
        mag_w1 = -2.5*(np.log10(5./np.sqrt(dat['PSFDEPTH_W1']))-9)
        dec_val = dat['DEC']
        sel = (mag_g>depth_g)&(mag_z<1e9)&(mag_r>depth_r)&(mag_r<1e9)&(mag_z>depth_z)&(mag_z<1e9)&(mag_w1>depth_w1)&(mag_w1<1e9)&(dec_val>dec)
        dat_new = dat[sel]
        dat_new.write(savedir+'depth_cut_chunk%d.fits'%i,overwrite=True)
        del(dat)
        del(dat_new)
        
        

def stack_step2(savedir,writedir,set_num=1):
    chunk_fns = glob.glob(savedir+'depth_cut_chunk*')
    sub_chunk_fns = np.array_split(chunk_fns,set_num)
    for k in range(set_num):
       dat = None
       for i in range(len(sub_chunk_fns[k])):
           tab_i = Table.read(sub_chunk_fns[k][i])
           sel = (tab_i['DEC']>-30)
           tab_i = tab_i[sel]
           print(i)
           if dat is not None:
               if len(tab_i)>0:
                  dat = vstack((dat,tab_i))
           else:
               if len(tab_i)>0:
                   dat = tab_i
       dat.write(writedir+"/part%d.fits"%(k),overwrite=True)
       print("written %s"%(writedir+"/part%d.fits"%(k)))

def main():
    parser = get_parser()
    opt = parser.parse_args()
    optdict = vars(opt)
    if optdict['collect']:
        obj = optdict['obj']
        threads = optdict['threads']
        savedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/sweep_%s/'%obj
        import subprocess
        subprocess.call(["mkdir",'-p',savedir])
        collect(savedir=savedir, obj=obj, threads=threads)
    if optdict['stack']:
        obj = optdict['obj']
        savedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/sweep_%s/'%obj
        stack(savedir)
    if optdict['depthcut']:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        if rank==0:
            obj = optdict['obj']
            savedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/sweep_%s/'%obj
            depthcut(savedir)
    if optdict['stack2']:
        obj = optdict['obj']
        savedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/sweep_%s/'%obj
        writedir = '/global/cscratch1/sd/huikong/Obiwan/dr9_LRG/obiwan_out/truth/%s/'%obj
        import subprocess
        subprocess.call(["mkdir",'-p',writedir])
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        if rank==0:
            stack_step2(savedir,writedir)
        print("written %s"%writedir)


if __name__ == "__main__":
    main()
