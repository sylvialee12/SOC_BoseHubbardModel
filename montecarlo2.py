__author__ = 'xlibb'
from hubbardTI import lattice2D
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
        mx, my, mz,mx1,my1,mz1 = [], [], [],[],[],[]
        mx0, my0, mz0,mx10,my10,mz10 = lattice.magnectization()
        mx.append(mx0)
        my.append(my0)
        mz.append(mz0)
        mx1.append(mx10)
        my1.append(my10)
        mz1.append(mz10)
        for i in range(self.times):
            if (i+1)%ones_times==0:
                beta+=delta_beta
            index,new_config,layers = lattice.new_config(self.batch)
            delta_E = lattice.delta_energy(index,new_config,layers)

            rand_num = random.rand()
            # if delta_E < 0:
            #     lattice.update_config(index,new_config,layers)
            #     # lattice.plotconfig()
            # else:
            #     rand_num = random.rand()
            #     if np.exp(-beta * delta_E) / (1 + np.exp(-beta * delta_E)) > rand_num:
            #         lattice.update_config(index,new_config,layers)

            if np.exp(-beta * delta_E)  > rand_num:
                lattice.update_config(index,new_config,layers)
            if (i+1)%100==0:
                mx0, my0, mz0,mx10,my10,mz10 = lattice.magnectization()
                mx.append(mx0)
                my.append(my0)
                mz.append(mz0)
                mx1.append(mx10)
                my1.append(my10)
                mz1.append(mz10)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(mx)
        plt.subplot(1, 3, 2)
        plt.plot(my)
        plt.subplot(1, 3, 3)
        plt.plot(mz)
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.plot(mx1)
        plt.subplot(1, 3, 2)
        plt.plot(my1)
        plt.subplot(1, 3, 3)
        plt.plot(mz1)

    # @functimer
    def run(self,beta,parameters,delta_beta=None,ones_times=None):
        for i in range(self.sample_num):
            latticeA=lattice2D(*parameters)
            latticeA.configuration()
            if delta_beta==None and ones_times==None:
                self.fix_temperature(latticeA,beta)
            else:
                self.annealing(latticeA,beta,delta_beta,ones_times)
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
    batch, criteria, times, sample_num=1, 1e-7, 100000, 1
    montecarlo=montecarlo(batch,criteria,times,sample_num)
    Nx, Ny, delta_theta, delta_phi = 6, 6, np.pi,np.pi
    Jx, Jy, Jz = -1*np.array([1, 1]), -1*np.array([1, 1]), 1*np.array([1, 1])
    Jx2, Jy2, Jz2 = -1/2*np.array([1, 1]), -1/2*np.array([1, 1]), 1/2*np.array([1, 1])
    Dx, Dy, Dz = np.array([0, 0]), np.array([0, 0]), np.array([0, 0])
    parameters=(Nx,Ny,delta_theta,delta_phi,Jx,Jy,Jz,Jx2, Jy2, Jz2,Dx,Dy,Dz,"square lattice",1)

    beta=1
    delta_beta=0.01
    ones_times=1
    montecarlo.run(beta,parameters,delta_beta,ones_times)
