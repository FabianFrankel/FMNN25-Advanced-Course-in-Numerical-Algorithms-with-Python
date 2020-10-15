from mpi4py import MPI
import numpy as np
import sys
import seaborn as sns
import matplotlib.pylab as plt
from scipy.linalg import block_diag
from numpy.linalg import inv


N = 20
proccess = 1
iterations = 10
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
WINDOW = 5
HEATER = 40
NORMAL = 15
OMEGA = 0.8
u_0 = np.ones(N-1)*20 #behövs i center från left
u_1_right = np.ones((N-1))*20 #behövs i right från center
u_1_left = np.ones((N-1))*20  #behövs i left från center
NC_left = np.ones((N-1))*20
NC_right = np.ones((N-1))*20 
u_2 = np.ones(N-1)*20   #behövs i center från right
u_left = np.ones((N*(N-1)))*20 #hela vänsterlösningen
u_right = np.ones((N*(N-1)))*20 #hela högerlösningen


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
    #print('small room')
    #print(A, np.shape(A))
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
    #print('small room')
    #print(A, np.shape(A))
    return A

def get_big_room_matrice(N):
    dim = (N-1)*(N*2)
    #print(dim)
    sub1_diag = np.ones(dim-1)
    for i in range(sub1_diag.size):
        if i % 2 == 1:
            sub1_diag[i] = 0
    A = np.diag(-4*np.ones(dim),0) + np.diag(np.ones(dim-2),2) + np.diag(np.ones(dim-2),-2) + np.diag(sub1_diag, 1) +np.diag(sub1_diag, -1)
    #print('big room')
    #print(A)
    return A

def buildRoom(u_left, u_center, u_right, u_1_left, u_1_right):

    l = np.zeros((N-1, N))
    r = np.zeros((N-1, N))

    for i in range(N-1):
        for j in range(N):
            l[i,j] = u_left[-j*(N-1)-i]
            r[i,j] = u_right[-j*(N-1)-i]


    c = np.zeros((2*N, N-1))
    for i in range(2*N):
        for j in range(N-1):
            c[i,j] = u_center[-i*(N-1)-j]

    

    window = np.ones(N-1)*WINDOW
    heater_v = np.ones(N-1)*HEATER
    normal = np.ones(2*N+2)*NORMAL
    normal = np.reshape(normal, ((2*N+2),1))

    c = np.vstack((heater_v,c))
    c = np.vstack((c,window))
    normal[0] = HEATER
    c = np.hstack((c,normal))
    normal[-1] = WINDOW
    c = np.hstack((normal,c))

    for i in range(N-1):
        c[-i-2,0] = u_1_left[-i]
        c[i+1,-1] = u_1_right[-i]


    window = np.ones(N)*WINDOW
    heater_v = np.ones(N)*HEATER
    heater_h = np.ones(N+1)*HEATER
    heater_h = np.reshape(heater_h,(N+1,1))
    normal = np.ones(N)*NORMAL

    r = np.vstack((r,normal))
    r = np.vstack((heater_v,r))
    r = np.hstack((r, heater_h))
    l = np.vstack((l,window))
    l = np.vstack((normal,l))
    l = np.hstack((heater_h,l))

    empty_small = np.zeros((N+1,N+1))

    r = np.vstack((r,empty_small))
    l = np.vstack((empty_small,l))

    Room = np.hstack((l,c))
    Room = np.hstack((Room,r))

    
    return Room

if (rank > 2):
    sys.exit()

A_big = get_big_room_matrice(N)
A_small_right = get_small_room_matrice_right(N)
A_small_left = get_small_room_matrice_left(N)

for itr in range(iterations):
    if(rank == 1):
        #print('rank:',rank)
        if itr != 0:
            #print('hej1', i)
            comm.Recv(u_0, source=0, tag=2)
            comm.Recv(u_2, source=2, tag=22)
            comm.Recv(u_left, source=0, tag=1)
            comm.Recv(u_right, source=2, tag=21)
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
        #print(np.shape(u_0))
        for i in range(1, 2*N-1):   #Changed
            if i < N:
                LeftBoundary[i-1] = u_0[i-1]    #Changed
                RightBoundary[i-1] = NORMAL #Changed
            else:   
                LeftBoundary[i-1] = NORMAL #Changed
                RightBoundary[i-1] = u_2[i-N-1] #Changed
        #print('hej2')
        for i in range(1,2*N-3):
            b[i*(N-1)] = -LeftBoundary[i]
            b[(i+1)*(N-1)-1] = -RightBoundary[i]
        #print('hej3')

        if itr > 0:
            u_center_old = u_center

        u_center = np.linalg.solve(A_big,b)


        if itr > 0:
            u_center = OMEGA*u_center + (1-OMEGA)*u_center_old


        u_1_left = []
        u_1_right = []
        NC_left = []
        NC_right = []

        for i in range((N-1)*2*N):
            if i < (N-1)*N:
                if(i+1)%N == 0:
                    u_1_left.append(u_center[i])
                    NC_left.append(u_center[i+1])
            else:
                if(i)%N == 0:
                    u_1_right.append(u_center[i+N-1])
                    NC_right.append(u_center[i+N-2])
        #print('hej4')
        u_1_left = np.array(u_1_left)
        u_1_right= np.array(u_1_right)
        NC_left = np.array(NC_left)
        NC_right = np.array(NC_right)
        #print('u_1_left', u_1_right, u)
        Room = buildRoom(u_left, u_center, u_right, u_1_left, u_1_right)
        

        comm.Send(u_1_left ,dest=0,tag=3)
        comm.Send(u_1_right ,dest=2, tag=4)
        comm.Send(NC_left, dest=0, tag=101)
        comm.Send(NC_right, dest=2, tag=100)

    elif(rank == 0): #SMALL LEFT
        comm.Recv(u_1_left, source=1, tag=3)
        comm.Recv(NC_left, source=1, tag=101)
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
        
        for i in range(N-1):           
            RightBoundary[i] = -u_1_left[i] + 2*NC_left[i]

        for i in range(1,N-3):
            b[i*(N)] = -LeftBoundary[i]
            b[(i+1)*(N)-1] = b[(i+1)*(N)-1]-RightBoundary[i-1]

        if itr > 0:
            u_left_old = u_left

        u_left = np.linalg.solve(A_small_left,b)

        if itr > 0:
            u_left = OMEGA*u_left + (1-OMEGA)*u_left_old

        u_0 = []
        for i in range((N-1)*(N-1)+N-1):
            if((i+1)%N == 0):
                u_0.append(u_left[i])
        u_0 = np.array(u_0)

        comm.Send(u_left, dest=1, tag=1)
        comm.Send(u_0, dest=1, tag=2)


        
    elif rank == 2: #RIGHT SMALL
        #print('rank:',rank)
        comm.Recv(u_1_right, source=1, tag=4)
        comm.Recv(NC_right, source=1, tag=100)
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
        
        for i in range(N-1):        #Changed
                LeftBoundary[i] = -u_1_right[i] + 2*NC_right[i]

        for i in range(1,N-3):
            b[i*(N)] = -LeftBoundary[i]
            b[(i+1)*(N)-1] = b[(i+1)*(N)-1]-RightBoundary[i-1]

        if itr > 0:
            u_right_old = u_right

        u_right = np.linalg.solve(A_small_right,b)

        u_2 = []

        if itr > 0:
            u_right = OMEGA*u_right + (1-OMEGA)*u_right_old

        for i in range((N-1)*(N-1)+N-1):
            if((i)%N == 0):
                u_2.append(u_right[i])
        u_2 = np.array(u_2)
        #print(u, u_2)

        
        comm.Send(u_right, dest=1, tag=21)
        comm.Send(u_2,dest=1, tag=22)
        



if rank == 1:
    comm.Recv(u_left, source=0)
    comm.Recv(u_right, source=2)
    fig , ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(Room, annot=False, linewidths=.5, ax=ax)             #sns.heatmap(Room,color="#03051A")     plt.show()
    plt.show()

    
elif rank == 2:
    comm.Send(u_right, dest=1)

elif rank == 0:
    comm.Send(u_left, dest=1)

sys.exit()



