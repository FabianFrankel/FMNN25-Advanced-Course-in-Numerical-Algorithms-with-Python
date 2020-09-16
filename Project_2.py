#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:20:59 2020

@author: Mikael
"""
from scipy import *
from numpy import *
from matplotlib.pyplot import *
import timeit


class Optimazation_problem:
    
    def __init__( self , object_function, gradient = 0):  
        self.object_function = object_function
        if not gradient:
            self.gradient = self.calculate_gradient()

        else:
            self.gradient = copy(gradient)

    def calculate_gradient(self):
        return

    def print(self, x):   #just to illustrate how we pass functions as arguments to the init function
        for i in x:
            print(i, self.object_function(i))
        

class Abstract_Newton:

    def __init__(self, optimization_problem):
        self.optimization_problem = copy(optimization_problem)


    def optimize(self, x_guess): 
        TOL = e-7
        x = x_guess
        f = self.optimization_problem.object_function
        g = self.optimization_problem.gradient
        g_x = g(x)
        hessian = 1

        while g_x < TOL:
            s = self.compute_s(hessian, g_x, x)
            alfa = self.compute_alfa()
            x = self.compute_next_x()
            hessian = self.compute_next_hessian()
        return x
    
    def compute_s(self, hessian, gradiant, x1): #This method is being handled by the classes children
        pass

    def compute_alfa(self): #This method is being handled by the classes children
        pass

    def line_search(self): #This method is being handled by the classes children
        pass

    def compute_next_x(self): #This method is being handled by the classes children
        pass


#In this class we can specify the different parts of how we do the optimization
class Newton(Abstract_Newton):
    def __init__(self, optimization_problem):
        super(Newton, self).__init__(self, optimization_problem)

    def compute_s(self, hessian, gradiant, x1): #Here is how we handle s calculation for this quasi newton method
        return

    def line_search(self): #Here is how we handle the line search for this quasi newton method
        return

    def compute_next_x(self):
        return



class Optimazation_methods2:
    
#    import Optimazation_problem
    def Newton( self , x_k , simplified = False , k_max = 100):
        
        ### INITIAL CONSTANTS/GUESS
        A = self.Jacobian( x_k )
        k = 0
        tol = 1e-5
        b = - self.F( x_k )
        x_kp1 = x_k.copy()
        ### INITIAL CONSTANTS
        
        while norm( b ) <= tol and k < k_max :   # Default 100 newton iterations
            
            k += 1
            x_k = x_kp1.copy()
            
            if simplified:
                
                x_delta = self.Cholesky_method( A , b )     # This should be fixed
                
                x_kp1 = x_delta + x_k
                b = - self.F( x_kp1 )
                
            else:
                
                x_delta = self.Cholesky_method( A , b )    # This should be fixed
                
                x_kp1 = x_delta + x_k
                A = self.Jacobian( x_kp1 )
                b = - self.F( x_kp1 )
        
        return x_np1
        
        
        
    def Grad_F2(self,X):
        
        l = shape(X)[0]
        
        
            

    '''
    def jac(self,u_n, u_0):
        h = 1e-4
        N = u_n.shape[0]
        JacF = zeros((N,N))
        H = zeros(N)
        
        for i in range(N):
            H[i] = h
            JacF[:,i] = (self.F(u_n + H, u_0) - self.F(u_n, u_0))/h
            H[i] = 0
        
        return JacF
    '''
        
    def Grad_F(self, x_1 , x_2 ):
        
        h = 1e-4
        
        
        return array( [ (self.F( x_1 + h , x_2 ) - self.F( x_1 , x_2 ) )/ h , (self.F( x_1 , x_2 + h) - self.F( x_1 ,x_2 ) ) / h ] )


        
    

#just to illustrate how we pass functions as arguments to the init function
def f(x):
    return x*x
op = Optimazation_problem(f)
x = [1,2,3,4,5,6]
op.print(x)
        
    
        
        



















