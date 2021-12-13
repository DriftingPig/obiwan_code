"""
Arnaud's code of randomly sampling a set of grid inputs
#TODO:
    1. combine with the input random generation code
    2. add maskbits, eliminate targets with masked bits set 
"""

import numpy as np
class GlassDistribution(object):
    ndim = 2
    def __init__(self, npoints=1000, nmesh=256, strength=1e-4, seed=42):
        self.rng = np.random.RandomState(seed=seed)
        self.positions = np.array([self.rng.uniform(0.,1.,npoints) for idim in range(self.ndim)]).T
        self.nmesh = nmesh
        if np.ndim(nmesh) == 0:
            self.nmesh = (nmesh,)*self.ndim
        kmesh = [np.fft.fftfreq(nmesh) for nmesh in self.nmesh]
        kmesh[-1] = kmesh[-1][:self.nmesh[-1]//2 + 1]
        kmesh = np.meshgrid(*kmesh,indexing='ij')
        k2 = np.sum([km**2 for km in kmesh],axis=0)
        k2.flat[0] = 1.
        self.koverk2 = [-1j*km/k2 for km in kmesh]
        self.edges = [np.linspace(0.,1.,nmesh) for nmesh in self.nmesh]
        self.strength = strength
    def sample_cic(self):
        index,dindex = [],[]
        for edges,position in zip(self.edges,self.positions.T):
            #i = np.clip(np.searchsorted(g,d,side='left')-1,0,len(g)-2)
            ii = np.searchsorted(edges,position,side='right')-1
            assert np.all((ii >= 0) & (ii < len(edges)))
            index.append(ii)
            di = (position-edges[ii])/(edges[ii+1]-edges[ii])
            #assert ((di>=0.) & (di<=1.)).all()
            dindex.append(di)
        index = np.array(index).T
        dindex = np.array(dindex).T
        ishifts = np.array(np.meshgrid(*([[0,1]]*self.ndim),indexing='ij')).reshape((self.ndim,-1)).T
        for ishift in ishifts:
            sindex = index + ishift
            sweight = np.prod((1-dindex) + ishift*(-1+2*dindex),axis=-1)
            #print sweight
            np.add.at(self.delta,tuple(sindex.T),sweight)
    def shift_cic(self):
        index,dindex = [],[]
        for edges,position in zip(self.edges,self.positions.T):
            #i = np.clip(np.searchsorted(g,d,side='left')-1,0,len(g)-2)
            ii = np.searchsorted(edges,position,side='right')-1
            assert np.all((ii >= 0) & (ii < len(edges)))
            index.append(ii)
            di = (position-edges[ii])/(edges[ii+1]-edges[ii])
            #assert ((di>=0.) & (di<=1.)).all()
            dindex.append(di)
        index = np.array(index).T
        dindex = np.array(dindex).T
        ishifts = np.array(np.meshgrid(*([[0,1]]*self.ndim),indexing='ij')).reshape((self.ndim,-1)).T
        for idim,disp in enumerate(self.disps):
            #print('Displacements for dim {:d}: mean = {:.4f}, rms = {:.4f}'.format(idim,np.mean(disp),np.std(disp)))
            for ishift in ishifts:
                sindex = index + ishift
                sweight = disp[tuple(sindex.T)] * np.prod((1-dindex) + ishift*(-1+2*dindex),axis=-1)
                self.positions[:,idim] += sweight
        self.positions %= 1.
    def compute(self):
        self.delta = np.zeros(self.nmesh,dtype='f8')
        self.sample_cic()
        self.delta = self.delta/self.delta.mean() - 1.
        potential = self.strength * np.fft.rfftn(self.delta)
        self.disps = [np.fft.irfftn(potential * koverk2) for koverk2 in self.koverk2]
        self.shift_cic()
    def move_stats(self, prev):
        diff = self.positions - prev
        mask = np.all([(position >= edges[1]) & (position <= edges[-2]) for edges,position in zip(self.edges,self.positions.T)],axis=0)
        diff = diff[mask]
        return np.sum([d**2 for d in diff],axis=0).max()**0.5
    def __call__(self, max_iter=1000, max_move=1e-3):
        prev = self.positions.copy()
        for it in range(max_iter):
            self.compute()
            tmp = self.move_stats(prev)
            #print(tmp)
            if tmp < max_move:
                print('Converged after {:d}/{:d} iterations'.format(it,max_iter))
                break
            prev = self.positions.copy()
if __name__ == '__main__':
    distrib = GlassDistribution(npoints=300)
    from matplotlib import pyplot as plt
    def plot_pos(prev=None):
        if prev is not None:
            plt.scatter(prev[:,0],prev[:,1],marker='.')
        plt.scatter(distrib.positions[:,0],distrib.positions[:,1],marker='.')
        plt.show()
    def plot_disp():
        fig, lax = plt.subplots(1,len(distrib.disps),figsize=(10,4))
        for ii,ax in enumerate(lax):
            c = ax.pcolor(distrib.disps[ii])
            fig.colorbar(c, ax=ax)
    """
    plot_pos()
    prev = distrib.positions.copy()
    for it in range(3):
        distrib(max_iter=10)
        plot_pos(prev=prev)
        prev = distrib.positions.copy()
        plot_disp()
    """
    distrib()
    plot_pos()
