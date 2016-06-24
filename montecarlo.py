__author__ = 'xlibb'
import numpy as np
import scipy as sp


class lattice(object):
    """
    This is the class to state out the Heisenberg lattice. One can get square lattice, triangle lattice and honeycomb lattice.
    Dimension can be 1 or 2.
    Bc can be 1(periodic BC) or 0 (open BC)
    """
    def __init__(self,Nx,Ny,delta_the,delta_phi,siteclass="square lattice",dimension=2,bc=1):
        self.delta_phi=delta_phi
        self.delta_the=delta_the
        self.Nx=Nx
        self.Ny=Ny
        self.siteclass=siteclass
        self.dimension=dimension
        self.bc=bc

    def configuration(self):
        if self.siteclass=="square lattice":
            self.theta=np.pi*np.random.rand(self.Nx,self.Ny)
            self.phi=2*np.pi*np.random.rand(self.Nx,self.Ny)
            self.sx=np.cos(self.theta)*np.cos(self.phi)
            self.sy=np.cos(self.theta)*np.sin(self.phi)
            self.sz=np.sin(self.theta)
        else:
            pass

    def energy(self,Jx,Jy,Jz,Dx,Dy,Dz):
        pass

    def update(self,batch):
        x_index=np.random.random_integers(0,self.Nx,size=batch)
        y_index=np.random.random_integers(0,self.Ny,size=batch)
        for x,y in zip(x_index,y_index):
            self.theta[x,y]+=self.delta_the*np.random.rand(1)[0]
            self.phi[x,y]+=self.delta_phi*np.random.rand(1)[0]
            pass








