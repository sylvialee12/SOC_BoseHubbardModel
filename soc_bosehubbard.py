import montecarlo1
import numpy as np

t=1
alpha=0.25*np.pi
lam=0.5
U=20
Jx= np.array([-4*t**2/(lam*U)*np.cos(2*alpha),-4*t**2/(lam*U) ])
Jy=np.array([-4*t**2/(lam*U) , -4*t**2/(lam*U)*np.cos(2*alpha)])
Jz=np.array([-4*t**2*(2*lam-1)/(lam*U)*np.cos(2*alpha), -4*t**2*(2*lam-1)/(lam*U)*np.cos(2*alpha)])
Dx=np.array([0,4*t**2/U*np.sin(2*alpha)])
Dy=np.array([-4*t**2/U*np.sin(2*alpha),0])
Dz=np.array([0,0])
batch, criteria, times, sample_num=1, 1e-7, 50000, 1
montecarlo=montecarlo1.montecarlo(batch,criteria,times,sample_num)
Nx, Ny, delta_theta, delta_phi = 6, 6, 1,np.pi
beta=1
delta_beta=0.01
ones_times=1
parameters=(Nx,Ny,delta_theta,delta_phi,Jx,Jy,Jz,Dx,Dy,Dz,"square lattice",1,0,0,0)
montecarlo.run(beta,parameters,delta_beta,ones_times)