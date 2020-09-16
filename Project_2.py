#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:20:59 2020

@author: Mikael
"""
from scipy import *
from matplotlib.pyplot import *
import timeit


class Optimazation_problem:
    
    
    
    def __init__( self , Gradient = 0 ):
        
        if not Gradient:
            from import_function import * 
            
            self.Gradient_check = True
            
        else:
            from import_function import * 
            
            self.Gradient_check = False
    
    
    
class Optimazation_methods:
    
    import Optimazation_problem
    
    
    
    
    
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


        
    
        
    
    
        
    
        
        






















start_time = time.time()

print("--- %s seconds ---" % (time.time() - start_time))
