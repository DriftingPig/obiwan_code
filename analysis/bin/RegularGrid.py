"""This script, implementing hexagonal grid, is adapted from https://github.com/sweverett/Balrog-GalSim/blob/master/balrog/grid.py."""
import numpy as np
class BaseGrid(object):
    def __init__(self, spacing=1, shape=3600, rotation=0., degree=True):
        self.spacing = spacing
        self.shape = np.empty(2,dtype='f8')
        self.shape[:] = shape
        self.rotation_angle = rotation
        if degree:
            self.rotation_angle /= (180./np.pi)
        # If there is a rotation, enlarge box
        corners = np.array([(x,y) for x in [0,self.shape[0]] for y in [0,self.shape[1]]])
        corners = self._rotate(corners)
        self._rx = np.min(corners[:,0]),np.max(corners[:,0])
        self._ry = np.min(corners[:,1]),np.max(corners[:,1])
        self._make_raw_grid()
        self.positions = self._rotate(self.positions)
        self.positions = self._mask(self.positions)
    def _rotate(self, positions):
        c, s = np.cos(self.rotation_angle), np.sin(self.rotation_angle)
        mat = np.array(((c,-s),(s, c))).T
        offset = self.shape/2.
        return (positions - offset).dot(mat) + offset
    def _mask(self, positions):
        mask = np.all((positions >= 0) & (positions <= self.shape),axis=-1)
        return self.positions[mask,:]
class RectGrid(BaseGrid):
    def _make_raw_grid(self):
        # regular x, y
        x = np.arange(*self._rx,step=self.spacing)
        y = np.arange(*self._ry,step=self.spacing)
        # mesh
        self.positions = np.array(np.meshgrid(x,y,indexing='ij')).T.reshape((-1,2))
class HexGrid(BaseGrid):

    def __init__(self, *args, shift=0, **kwargs):
        self.shift = shift % 2
        super(HexGrid,self).__init__(*args,**kwargs)

    def _make_raw_grid(self):

        # Geometric factors of given hexagon
        # spacing = radius
        p = self.spacing * np.tan(np.pi / 6.) # side length / 2
        x = np.arange(self._rx[0],self._rx[1]+2*p,step=2*p) # first line
        xs = x - p # second line, shifted by half a side
        y = np.arange(*self._ry,step=2*p)
        x,y = np.meshgrid(x,y,indexing='ij')
        x[:,self.shift::2] = xs[:,None]
        self.positions = np.array([x,y]).T.reshape((-1,2))
        
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    grid = RectGrid(shape=1000)
    grid_rot = RectGrid(shape=1000,rotation=20.)
    plt.scatter(grid.positions[:,0],grid.positions[:,1],marker='.')
    plt.scatter(grid_rot.positions[:,0],grid_rot.positions[:,1],marker='.')
    plt.show()
    grid = HexGrid(shape=1000)
    plt.scatter(grid.positions[:,0],grid.positions[:,1],marker='.')
    #from grid_orig import HexGrid
    #ref = HexGrid.calc_hex_coords(0,0,1000,1000,60)
    #plt.scatter(ref[:,0],ref[:,1],marker='.')
    plt.show()