__author__ = 'xlibb'

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class lattice2D:
    """
    This class is to implement the lattice structure in Heisenberg model. The options are:
    siteclass: "square lattice","honeycomb lattice", "triangular lattice"
    Nx: number of vector 1
    Ny: number of vector 2
    Boundary condition: 0 for open boundary, 1 for periodic boundary
    """
    def __init__(self,Nx,Ny,delta_theta,delta_phi,Jx,Jy,Jz,Jx2,Jy2,Jz2,Dx,Dy,Dz,siteclass="square lattice",bc=1,hx=0,hy=0,hz=0):
        self.Nx=Nx
        self.Ny=Ny
        self.siteclass=siteclass
        self.bc=bc
        self.delta_theta=delta_theta
        self.delta_phi=delta_phi
        self.Jx=Jx
        self.Jy=Jy
        self.Jz=Jz
        self.Jx2=Jx2
        self.Jy2=Jy2
        self.Jz2=Jz2
        self.Dx=Dx
        self.Dy=Dy
        self.Dz=Dz
        self.hx=hx
        self.hy=hy
        self.hz=hz
        self.h=np.array([self.hx,self.hy,self.hz])



    def configuration(self):
        """
        Initialize the configuration
        """
        x1=2* random.rand(self.Nx, self.Ny)
        self.theta1=np.arccos(1-x1)
        self.phi1=2*np.pi*random.rand(self.Nx,self.Ny)
        self.sx1=np.sin(self.theta1)*np.cos(self.phi1)
        self.sy1=np.sin(self.theta1)*np.sin(self.phi1)
        self.sz1=np.cos(self.theta1)
        x2=2* random.rand(self.Nx, self.Ny)
        self.theta2=np.arccos(1-x2)
        self.phi2=2*np.pi*random.rand(self.Nx,self.Ny)
        self.sx2=np.sin(self.theta2)*np.cos(self.phi2)
        self.sy2=np.sin(self.theta2)*np.sin(self.phi2)
        self.sz2=np.cos(self.theta2)


    def energy(self):
        pass


    def new_config(self,batch):
        """
        Randomly generate a new configuration
        """
        x_index,y_index=random.random_integers(0,self.Nx-1,size=batch),random.random_integers(0,self.Ny-1,size=batch)
        x=(2 * random.rand(batch))
        thetaprime,phiprime=np.arccos(1-x),self.delta_phi*(2*random.rand(batch))
        layer=np.round(np.random.rand(batch))
        # thetaprime,phiprime=2*random.rand(batch),self.delta_phi*(2*random.rand(batch)-1)
        return np.vstack([x_index,y_index]).transpose(), np.vstack([thetaprime,phiprime]).transpose(),layer.transpose()

    def delta_energy(self,index,new_config,layers):
        """
        This is to calculate the energy difference between the new configuration and the older configuration
        """
        Jx,Jy,Jz,Dx,Dy,Dz=self.Jx,self.Jy,self.Jz,self.Dx,self.Dy,self.Dz
        Jx2,Jy2,Jz2=self.Jx2,self.Jy2,self.Jz2
        delta=0
        for site, angle,layer in zip(index,new_config,layers):
            x_index,y_index=site
            thetaprime, delphi = angle  # the change of the angles
            phiprime= delphi
            # thetaprime, phiprime = self.theta[x_index,y_index]+del_x/np.sin(self.theta[x_index,y_index]),\
            #                        self.phi[x_index, y_index] + delphi
            # thetaprime,phiprime=deltheta,delphi
            # angles at site(x_index,y_index) after change
            neighbors=self.neighbors(x_index,y_index)
            if layer==0:
                sxprime,sx0=np.sin(thetaprime)*np.cos(phiprime),self.sx1[x_index,y_index]
                syprime,sy0=np.sin(thetaprime)*np.sin(phiprime),self.sy1[x_index,y_index]
                szprime,sz0=np.cos(thetaprime),self.sz1[x_index,y_index]
                sdel=np.array([sxprime-sx0,syprime-sy0,szprime-sz0])
                delta+=np.dot(self.h,sdel)
                for index in neighbors:
                    if index[0] == 0 and x_index == self.Nx-1:
                        xvector = 1
                        yvector=0
                    elif index[0]==self.Nx-1 and x_index==0:
                        xvector=-1
                        yvector=0
                    elif index[1] == 0 and y_index==self.Ny-1:
                        xvector =0
                        yvector = 1
                    elif index[1]==self.Ny-1 and y_index==0:
                        xvector=0
                        yvector=-1
                    else:
                        xvector, yvector = index[0] - x_index, index[1] - y_index
                    vector = np.array([xvector, yvector])
                    vector = vector / np.linalg.norm(vector)
                    J=np.array([Jx.dot(np.abs(vector)),Jy.dot(np.abs(vector)), Jz.dot(np.abs(vector))])
                    J2=np.array([Jx2.dot(np.abs(vector)),Jy2.dot(np.abs(vector)), Jz2.dot(np.abs(vector))])
                    D=np.array([Dx.dot(vector),Dy.dot(vector),Dz.dot(vector)])
                    neighborS=np.array([self.sx1[index[0], index[1]],self.sy1[index[0], index[1]],self.sz1[index[0], index[1]]])
                    s2=np.array([self.sx2[index[0], index[1]],self.sy2[index[0], index[1]],self.sz2[index[0], index[1]]])
                    delta += J.dot(sdel*neighborS)+D.dot(np.cross(neighborS,sdel))
                    delta+=J2.dot(s2*sdel)
            else:
                sxprime,sx0=np.sin(thetaprime)*np.cos(phiprime),self.sx2[x_index,y_index]
                syprime,sy0=np.sin(thetaprime)*np.sin(phiprime),self.sy2[x_index,y_index]
                szprime,sz0=np.cos(thetaprime),self.sz2[x_index,y_index]
                sdel=np.array([sxprime-sx0,syprime-sy0,szprime-sz0])
                delta+=np.dot(self.h,sdel)
                for index in neighbors:
                    if index[0] == 0 and x_index == self.Nx-1:
                        xvector = 1
                        yvector=0
                    elif index[0]==self.Nx-1 and x_index==0:
                        xvector=-1
                        yvector=0
                    elif index[1] == 0 and y_index==self.Ny-1:
                        xvector =0
                        yvector = 1
                    elif index[1]==self.Ny-1 and y_index==0:
                        xvector=0
                        yvector=-1
                    else:
                        xvector, yvector = index[0] - x_index, index[1] - y_index
                    vector = np.array([xvector, yvector])
                    vector = vector / np.linalg.norm(vector)
                    J=np.array([Jx.dot(np.abs(vector)),Jy.dot(np.abs(vector)), Jz.dot(np.abs(vector))])
                    J2=np.array([Jx2.dot(np.abs(vector)),Jy2.dot(np.abs(vector)), Jz2.dot(np.abs(vector))])
                    D=np.array([Dx.dot(vector),Dy.dot(vector),Dz.dot(vector)])
                    neighborS=np.array([self.sx2[index[0], index[1]],self.sy2[index[0], index[1]],self.sz2[index[0], index[1]]])
                    s1=np.array([self.sx1[index[0], index[1]],self.sy1[index[0], index[1]],self.sz1[index[0], index[1]]])
                    delta += J.dot(sdel*neighborS)+D.dot(np.cross(neighborS,sdel))
                    delta+=J2.dot(s1*sdel)
        return delta


    def neighbors(self,x_index,y_index):
        if self.siteclass=="square lattice":
            if x_index not in [0,self.Nx-1] and y_index not in [0,self.Ny-1]:
                neighbors=[[x_index-1,y_index],[x_index+1,y_index],[x_index,y_index+1],[x_index,y_index-1]]
            elif x_index==0 and y_index not in [0,self.Ny-1]:
                if self.bc==1:
                    neighbors=[[self.Nx-1,y_index],[x_index+1,y_index],[x_index,y_index+1],[x_index,y_index-1]]
                else:
                    neighbors=[[x_index+1,y_index],[x_index,y_index+1],[x_index,y_index-1]]
            elif x_index==self.Nx-1 and y_index not in [0,self.Ny-1]:
                if self.bc == 1:
                    neighbors = [[x_index-1, y_index], [0, y_index], [x_index, y_index + 1],
                                 [x_index, y_index - 1]]
                else:
                    neighbors = [[x_index -11, y_index], [x_index, y_index + 1], [x_index, y_index - 1]]
            elif y_index==0 and x_index not in [0,self.Nx-1]:
                if self.bc == 1:
                    neighbors = [[x_index - 1, y_index], [x_index+1, y_index], [x_index, y_index + 1],
                                 [x_index, self.Ny-1]]
                else:
                    neighbors = [[x_index - 1, y_index], [x_index+1, y_index], [x_index, y_index + 1]]
            elif y_index==self.Ny-1 and x_index not in [0,self.Nx-1]:
                if self.bc == 1:
                    neighbors = [[x_index - 1, y_index], [x_index + 1, y_index], [x_index, 0],
                                 [x_index, y_index - 1]]
                else:
                    neighbors = [[x_index - 1, y_index], [x_index + 1, y_index], [x_index, y_index - 1]]
            elif y_index==0 and x_index==0:
                if self.bc == 1:
                    neighbors = [[self.Nx- 1, y_index], [x_index + 1, y_index], [x_index, y_index+1],
                                 [x_index, self.Ny - 1]]
                else:
                    neighbors = [[x_index + 1, y_index], [x_index, y_index+1]]
            elif y_index==0 and x_index==self.Nx-1:
                if self.bc == 1:
                    neighbors = [[x_index-1,y_index],[0,y_index],[x_index,y_index+1],[x_index,self.Ny-1]]
                else:
                    neighbors = [[x_index-1,y_index],[x_index,y_index+1]]
            elif y_index==self.Ny-1 and x_index==0:
                if self.bc==1:
                    neighbors = [[self.Nx-1, y_index], [x_index + 1, y_index], [x_index, 0], [x_index, y_index - 1]]
                else:
                    neighbors= [ [x_index + 1, y_index], [x_index, y_index - 1]]
            else:
                if self.bc == 1:
                    neighbors = [[x_index-1, y_index], [0, y_index], [x_index, 0], [x_index, y_index - 1]]
                else:
                    neighbors = [ [x_index-1, y_index], [x_index, y_index - 1]]
            return neighbors
        else:
            pass


    def update_config(self,index,new_config,layer):
        for site,angle,layer in zip(index,new_config,layer):
            x_index, y_index = site
            thetaprime, delphi = angle  # the change of the angles
            phiprime= delphi
            # thetaprime, phiprime = deltheta,  delphi

            # angles at site(x_index,y_index) after change
            if layer==0:
                self.theta1[x_index,y_index]=thetaprime
                self.phi1[x_index,y_index]=phiprime
                self.sx1[x_index,y_index]=np.sin(thetaprime)*np.cos(phiprime)
                self.sy1[x_index,y_index]=np.sin(thetaprime)*np.sin(phiprime)
                self.sz1[x_index,y_index]=np.cos(thetaprime)
            else:
                self.theta2[x_index,y_index]=thetaprime
                self.phi2[x_index,y_index]=phiprime
                self.sx2[x_index,y_index]=np.sin(thetaprime)*np.cos(phiprime)
                self.sy2[x_index,y_index]=np.sin(thetaprime)*np.sin(phiprime)
                self.sz2[x_index,y_index]=np.cos(thetaprime)


    def magnectization(self):
        return np.mean(np.mean(self.sx1)),np.mean(np.mean(self.sy1)),np.mean(np.mean(self.sz1)),\
               np.mean(np.mean(self.sx2)),np.mean(np.mean(self.sy2)),np.mean(np.mean(self.sz2))

    def stagger_magnetization(self):
        pass

    def plotconfig(self):
        fig=plt.figure()
        ax = Axes3D(fig)
        x=np.arange(1,self.Nx+1)
        y=np.arange(1,self.Ny+1)
        X,Y=np.meshgrid(y,x)
        Z=0*X
        plt.quiver(X, Y, Z, self.sx1, self.sy1, self.sz1,
                   length=0.8,# data
                   )
        Z2=0*X+3
        plt.quiver(X, Y, Z2, self.sx2, self.sy2, self.sz2,
                   length=0.8,color='green'
                   )
        ax.set_zlim3d([-1,5])
        fig2=plt.figure()
        ax = Axes3D(fig2)
        x=np.arange(1,self.Nx+1)
        y=np.arange(1,self.Ny+1)
        X,Y=np.meshgrid(y,x)
        Z=0*X
        plt.quiver(X, Y, Z, self.sx1-self.sx2, self.sy1-self.sy2, self.sz1-self.sz2,
                   length=0.8,# data
                   )
        ax.set_zlim3d([-1,1])
        plt.show()



"""
if __name__=="__main__":
    Nx, Ny, delta_theta, delta_phi=6,6,0.01,0.01
    Jx,Jy,Jz=np.array([1,1]),np.array([1,1]),np.array([0.5,0.5])
    Dx, Dy, Dz = np.array([1, 1]), np.array([1, 1]), np.array([0.5, 0.5])
    latticeA=lattice2D(Nx,Ny,delta_theta,delta_phi,Jx,Jy,Jz,Dx,Dy,Dz,siteclass="square lattice",bc=1)
    latticeA.configuration()
    latticeA.plotconfig()

"""


