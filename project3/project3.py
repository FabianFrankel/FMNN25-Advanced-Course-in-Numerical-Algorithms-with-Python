from mpi4py import MPI
import numpy as np
import sys
import matplotlib.pylab as plt
from scipy.linalg import block_diag
from numpy.linalg import inv

inv_stepsize = 20
proccess = 1
iterations = 20
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
WINDOW = 5
HEATER = 40
NORMAL = 15
u_0 = np.ones(6)*20
u_1 = np.ones(12)*20
u_2 = np.ones(6)*20

#Run this file with cmd 'mpirun -np 3 python project3.py'
def get_small_room_matrice(inv_stepsize):
    block1 = np.diag(-4*np.ones(inv_stepsize)) + np.diag(np.ones(inv_stepsize-1),1) +np.diag(np.ones(inv_stepsize-1),-1)
    block2 = np.diag(np.ones(inv_stepsize))
    A = np.row_stack([np.column_stack([block1,block2]),np.column_stack([block2,block1])])
    #print('small room')
    #print(A)
    return A

def get_big_room_matrice(inv_stepsize):
    dim = (inv_stepsize-1)*(inv_stepsize*2)
    print(dim)
    sub1_diag = np.ones(dim-1)
    for i in range(sub1_diag.size):
        if i % 2 == 1:
            sub1_diag[i] = 0
    A = np.diag(-4*np.ones(dim),0) + np.diag(np.ones(dim-2),2) + np.diag(np.ones(dim-2),-2) + np.diag(sub1_diag, 1) +np.diag(sub1_diag, -1)
    #print('big room')
    #print(A)
    return A

if (rank > 2):
    sys.exit()

for i in range(iterations):
    if(rank == 1):
        if i != 0:
            comm.Recv(u_0, source=0)
            comm.Recv(u_2, source=2)
        A = get_big_room_matrice(inv_stepsize)
        b = np.array([u_0[3] -WINDOW, -NORMAL -WINDOW, u_0[5], -NORMAL, -NORMAL, -NORMAL, -NORMAL, -NORMAL, -NORMAL, -u_1[2], -NORMAL -HEATER, u_1[5]-HEATER])
        u_1 = np.linalg.solve(A,b)
        comm.Send(u_1,dest=0)
        comm.Send(u_1,dest=2)
    elif(rank == 0):
        comm.Recv(u_1, source=1)
        A = get_small_room_matrice(inv_stepsize)
        b = np.array([-WINDOW -HEATER, -WINDOW, -WINDOW -u_1[0], -NORMAL -HEATER, -NORMAL, -NORMAL -u_1[2]])
        u_0 = np.linalg.solve(A,b)
        comm.Send(u_0, dest=1)
        
    elif rank == 2:
        comm.Recv(u_1, source=1)
        A = get_small_room_matrice(inv_stepsize)
        b = np.array([-NORMAL -HEATER, -NORMAL, -NORMAL -u_1[0], -HEATER -HEATER, -NORMAL, -HEATER -u_1[2]])
        u_2 = np.linalg.solve(A,b)
        comm.Send(u_2,dest=1)

if rank == 1:
    comm.Recv(u_0, source=0)
    comm.Recv(u_2, source=2)
    smallRoom_1 = np.zeros([inv_stepsize+1,inv_stepsize+1])
    smallRoom_2 = np.zeros([inv_stepsize+1,inv_stepsize+1])
    print(u_0)
    print(u_1)
    print(u_2)
elif rank == 2:
    comm.Send(u_2, dest=1)
elif rank == 0:
    comm.Send(u_0, dest=1)





