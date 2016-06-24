from lattice3 import lattice2D
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os, time


def functimer(func):
    def wrapper(*args,**kwargs):
        now = time.time()
        a=func(*args,**kwargs)
        print("Running %s : %f s"%(func.__name__,time.time()-now))
        return a
    return wrapper

def mag_visualize(func):
    def wrapper(lattice,*args,**kwargs):
        pass
    pass


# pool = Pool(4)

class montecarlo:
    """
    This is the implementation of metropolis montecarlo algorithm
    """
    def __init__(self,batch,criteria,times,sample_num):
        self.batch=batch
        self.criteria=criteria
        self.times=times
        self.sample_num=sample_num

    @functimer
    def fix_temperature(self,lattice,beta):
        mx,my,mz=[],[],[]
        mx0,my0,mz0=lattice.magnectization()
        mx.append(mx0)
        my.append(my0)
        mz.append(mz0)
        for i in range(self.times):
            index,new_config=lattice.new_config(self.batch)
            delta_E=lattice.delta_energy(index,new_config)
            if delta_E<0:
                lattice.update_config(index,new_config)
                # lattice.plotconfig()
            else:
                rand_num=random.rand()
                if np.exp(-beta*delta_E)/(1+np.exp(-beta*delta_E))>rand_num:
                    lattice.update_config(index,new_config)
            mx0, my0, mz0 = lattice.magnectization()
            mx.append(mx0)
            my.append(my0)
            mz.append(mz0)
        plt.figure("Magnetization")
        plt.subplot(1,3,1)
        plt.plot(mx)
        plt.title("x direction")
        plt.subplot(1, 3, 2)
        plt.plot(my)
        plt.title("y direction")
        plt.subplot(1, 3, 3)
        plt.plot(mz)
        plt.title("z direction")



    @functimer
    def annealing(self,lattice,beta,delta_beta,ones_times):
        mx, my, mz = [], [], []
        mxs, mys, mzs = [], [], []
        mx0, my0, mz0 = lattice.magnectization()
        mxs0, mys0, mzs0 = lattice.stagger_magnetization()
        mx.append(mx0)
        my.append(my0)
        mz.append(mz0)
        mxs.append(mxs0)
        mys.append(mys0)
        mzs.append(mzs0)
        for i in range(self.times):
            if (i+1)%ones_times==0:
                beta+=delta_beta
            index,new_config = lattice.new_config(self.batch)
            delta_E = lattice.delta_energy(index,new_config)

            rand_num = random.rand()

            if np.exp(-beta * delta_E)  > rand_num:
                lattice.update_config(index,new_config)
            if (i+1)%100==0:
                mx0, my0, mz0 = lattice.magnectization()
                mxs0, mys0, mzs0 = lattice.stagger_magnetization()
                mx.append(mx0)
                my.append(my0)
                mz.append(mz0)
                mxs.append(mxs0)
                mys.append(mys0)
                mzs.append(mzs0)
        mxs,mys,mzs=np.array(mxs),np.array(mys),np.array(mzs)
        plt.figure("Magnetization and stagger magnetization")
        ax=plt.subplot(3, 3, 1)
        ax.plot(mx)
        ax.set_title('m_x')
        ax=plt.subplot(3, 3, 2)
        ax.plot(my)
        ax.set_title('m_y')
        ax=plt.subplot(3, 3, 3)
        ax.plot(mz)
        ax.set_title('m_z')
        ax=plt.subplot(3, 3, 4)
        ax.plot(mxs[:,0])
        ax.set_title('mxs_x')
        ax=plt.subplot(3, 3, 5)
        ax.plot(mys[:,0])
        ax.set_title('mxs_y')
        ax=plt.subplot(3, 3, 6)
        ax.plot(mzs[:,0])
        ax.set_title('mxs_z')
        ax = plt.subplot(3, 3, 7)
        ax.plot(mxs[:, 1])
        ax.set_title('mys_x')
        ax = plt.subplot(3, 3, 8)
        ax.plot(mys[:, 1])
        ax.set_title('mys_y')
        ax = plt.subplot(3, 3, 9)
        ax.plot(mzs[:, 1])
        ax.set_title('mys_z')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    # @functimer
    def run(self,beta,parameters,delta_beta=None,ones_times=None):
        for i in range(self.sample_num):
            latticeA=lattice2D(*parameters)
            latticeA.configuration()
            if delta_beta==None and ones_times==None:
                self.fix_temperature(latticeA,beta)
            else:
                self.annealing(latticeA,beta,delta_beta,ones_times)
            latticeA.savedata()
            latticeA.plotconfig()


    @functimer
    def phase_diagram(self,beta_list,parameters):
        def phase(beta):
            latticeA=lattice2D(*parameters)
            latticeA.configuration()
            self.fix_temperature(latticeA, beta)
            mx0, my0, mz0 = latticeA.magnectization()
            return [mx0,my0,mz0]
        # mag=pool.map(phase,beta_list)
        mag = np.array(list(map(phase, beta_list)))
        plt.figure()
        plt.plot(beta_list,mag[:,2])
        plt.show()

if __name__=="__main__":
    batch, criteria, times, sample_num=1, 1e-7, 10000, 1
    montecarlo=montecarlo(batch,criteria,times,sample_num)
    Nx, Ny, delta_theta, delta_phi = 6, 6, 1,2*np.pi
    Jx, Jy, Jz = np.array([0, 0]), np.array([1, 1]), np.array([0, 0])
    Dx, Dy, Dz = np.array([0, 0]), np.array([0, 0]), np.array([0, 0])
    parameters=(Nx,Ny,delta_theta,delta_phi,Jx,Jy,Jz,Dx,Dy,Dz,"square lattice",1)

    beta=.1
    delta_beta=0.01
    ones_times=1
    montecarlo.run(beta,parameters,delta_beta,ones_times)

