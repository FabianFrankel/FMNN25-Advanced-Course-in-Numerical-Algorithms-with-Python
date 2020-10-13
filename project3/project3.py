from mpi4py import MPI
import numpy as np
import sys
import matplotlib.pylab as plt
from scipy.linalg import block_diag
from numpy.linalg import inv

N = 3
proccess = 1
iterations = 20
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
WINDOW = 5
HEATER = 40
NORMAL = 15
u_0 = np.ones(6)*20
u_1_right = np.ones(12)*20
u_1_left = np.ones(12)*20
u_2 = np.ones(6)*20

#Run this file with cmd 'mpirun -np 3 python project3.py'
def get_small_room_matrice_left(N):
    A = -4*np.diag(np.ones(N*(N-1)))  + np.diag(np.ones(N*(N-1)-1),-1) + np.diag(np.ones(N*(N-1)-1),1) + np.diag(np.ones(N*(N-2)),N) + np.diag(np.ones(N*(N-2)),-N)
    #block2 = np.diag(np.ones(N+2))
    #A = np.row_stack([np.column_stack([block1,block2]),np.column_stack([block2,block1])])
    for i in range(N*(N-1)):
        if((i+1)%N == 0):
            A[i,:] = np.zeros(N*(N-1))
            A[i,i] = -3
            A[i,i-1] = 1
            if(i-N > 0):
                A[i,i-N] = 1
    print('small room')
    print(A, np.shape(A))
    return A

def get_small_room_matrice_right(N):
    A = -4*np.diag(np.ones(N*(N-1)))  + np.diag(np.ones(N*(N-1)-1),-1) + np.diag(np.ones(N*(N-1)-1),1) + np.diag(np.ones(N*(N-2)),N) + np.diag(np.ones(N*(N-2)),-N)
    #block2 = np.diag(np.ones(N+2))
    #A = np.row_stack([np.column_stack([block1,block2]),np.column_stack([block2,block1])])
    for i in range(N*(N-1)):
        if(i%N == 0):
            A[i,:] = np.zeros(N*(N-1))
            A[i,i] = -3
            if not (i-1):
                A[i,i-1] = 1
            if(i-N >= 0):
                A[i,i-N] = 1
    print('small room')
    print(A, np.shape(A))
    return A

def get_big_room_matrice(N):
    dim = (N-1)*(N*2)
    print(dim)
    sub1_diag = np.ones(dim-1)
    for i in range(sub1_diag.size):
        if i % 2 == 1:
            sub1_diag[i] = 0
    A = np.diag(-4*np.ones(dim),0) + np.diag(np.ones(dim-2),2) + np.diag(np.ones(dim-2),-2) + np.diag(sub1_diag, 1) +np.diag(sub1_diag, -1)
    print('big room')
    print(A)
    return A

if (rank > 2):
    sys.exit()

A_big = get_big_room_matrice(N)
A_small_right = get_small_room_matrice_right(N)
A_small_left = get_small_room_matrice_left(N)

for i in range(iterations):
    if(rank == 1):
        if i != 0:
            comm.Recv(u_0, source=0)
            comm.Recv(u_2, source=2)
        #b = np.array([u_0[3] -WINDOW, -NORMAL -WINDOW, u_0[5], -NORMAL, -NORMAL, -NORMAL, -NORMAL, -NORMAL, -NORMAL, -u_1[2], -NORMAL -HEATER, u_1[5]-HEATER])
        
        b = np.zeros((N-1)*2*N)
        for i in range(N-1):
            b[i] = -WINDOW
            b[-1-i] = -HEATER
        #LEFT
        b[0] = b[0] - u_0[0]
        b[N-2] = b[N-2] - NORMAL
        #RIGHT
        b[-1] = b[-1] - u_2[-1]
        b[1-N] = b[1-N] - NORMAL
        LeftBoundary = np.zeros(2*N)
        RightBoundary = np.zeros(2*N)
        for i in range(2*N):
            if i < N:
                LeftBoundary[i] = u_0[i]
                RightBoundary[i] = NORMAL
            else:
                LeftBoundary[i] = NORMAL
                RightBoundary[i] = u_2[i-N] #KANSKE
        
        for i in range(1,2*N-3):
            b[i*(N-1)] = -LeftBoundary[i]
            b[(i+1)*(N-1)-1] = -RightBoundary[i]

        u = np.linalg.solve(A_big,b)
        u_1_left = np.zeros(N-1)
        u_1_right = np.zeros(N-1)
        for i,j in zip(range((N-1)*2*N), range(N-1)):
            if i < (N-1)*N:
                if(i+1)%N:
                    u_1_left[j] = u[i]
            else:
                if(i+1)%N:
                    u_1_right[j] = u[i+N-1]

        comm.Send(u_1_left,dest=0)
        comm.Send(u_1_right,dest=2)

    elif(rank == 0):
        comm.Recv(u_1_left, source=1)
        b = np.zeros((N-1)*(N-1)+N-1)

        for i in range(N-1):
            b[i] = -WINDOW
            b[-1-i] = -NORMAL
        #LEFT
        b[0] = b[0] - HEATER
        #RIGHT
        b[1-N] = b[1-N] - HEATER
        LeftBoundary = np.zeros(N)
        RightBoundary = np.zeros(N)

        for i in range(N-3):
                LeftBoundary[i] = HEATER
        
        for i in range(N):
            RightBoundary[i] = u_1_left[i]

        for i in range(1,N-3):
            b[i*(N)] = -LeftBoundary[i]
            b[(i+1)*(N)-1] = b[(i+1)*(N)-1]-RightBoundary[i-1]
        u = np.linalg.solve(A_small_left,b)
        u_0 = np.zeros(N-1)
        for i,j in zip(range((N-1)*(N-1)+N-1),range(N-1)):
            if((i+1)%N == 0):
                u_0[j] = u[i]
        comm.Send(u_0, dest=1)

        
    elif rank == 2:
        comm.Recv(u_1_right, source=1)
        b = np.zeros((N-1)*(N-1)+N-1)

        for i in range(N-1):
            b[i] = -NORMAL
            b[-1-i] = -HEATER
        
        b[N-2] = b[N-2] - HEATER
        
        b[-1] = b[-1] - HEATER

        LeftBoundary = np.zeros(N)
        RightBoundary = np.zeros(N)

        for i in range(N-3):
                RightBoundary[i] = HEATER
        
        for i in range(N):
                LeftBoundary[i] = u_1_right[i]

        for i in range(1,N-3):
            b[i*(N)] = -LeftBoundary[i]
            b[(i+1)*(N)-1] = b[(i+1)*(N)-1]-RightBoundary[i-1]
        u = np.linalg.solve(A_small_right,b)
        u_2 = []
        for i,j in zip(range((N-1)*(N-1)+N-1),range(N-1)):
            if((i+1)%N == 0):
                u_2[j] = u[i]
        comm.Send(u_2,dest=1)

if rank == 1:
    comm.Recv(u_0, source=0)
    comm.Recv(u_2, source=2)
    smallRoom_1 = np.zeros([N+1,N+1])
    smallRoom_2 = np.zeros([N+1,N+1])
    print('u_0',u_0)
    print('u_1_left',u_1_left)
    print('u_1_right',u_1_right)
    print('u_2',u_2)
elif rank == 2:
    comm.Send(u_2, dest=1)
elif rank == 0:
    comm.Send(u_0, dest=1)





