#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:20:59 2020

@author: Mikael
"""
from scipy import *
from matplotlib.pyplot import *
from numpy import *
from scipy import optimize as so
import timeit
from Imported_function import *

class Optimisation_problem:

    def __init__( self , f , Grad = False ):
        self.f = f
        if Grad:
            self.Grad_f = Grad
            self.Choice_grad = False
        else:
            self.Grad_f = Grad
            self.Choice_grad = True

class Optimazation_methods:

    def __init__( self, optimisation_problem , initial_x , tol = 1e-5):
        self.optimisation_problem = optimisation_problem
        self.initial_x = initial_x
        self.tol = tol
    
    def Grad_F( self , X ):
        
        h = 1e-10
        n = shape( X )[0]
        grad_vector = zeros( n )
        H = zeros( n )
   
        for i in range( n ):
            H[ i ] = h
            grad_vector[ i ] = ( self.optimisation_problem.f( X + H ) - self.optimisation_problem.f( X ) ) / h
            H[ i ] = 0
        
        return grad_vector
    
    def Hersian( self , X ):
        
        h = 1e-5
        n = X.shape[ 0 ]
        Hersian = zeros( (n,n) )
        H = zeros( n )
        
        for i in range( n ):
            H[ i ] = h
            Hersian[ : , i ] = ( self.Grad_choice( X + H ) - self.Grad_choice( X ) ) / h
            H[ i ] = 0
        
        Her_F = (1/2) * ( Hersian + Hersian.T )
        
        
        return Her_F
    
    def Compare_BFGS( self , H_k , x_k ):

        grad_f_k = self.Grad_choice( x_k )
        Her_f_k = self.Hersian( x_k )
        G_inv_k = linalg.inv( Her_f_k )

        Result =  H_k - G_inv_k
        return linalg.norm(Result)
        
    def Newton( self , Method_for_a , Method_for_H_update):
        
        x_k = self.initial_x.copy()
        grad_f_k = self.Grad_choice( x_k )
        Her_f_k = self.Hersian( x_k )
        H_k = linalg.inv( Her_f_k )
        k = 0
        kmax = 100000
        alpha = 1
        while linalg.norm( self.Grad_choice( x_k ) ) > self.tol and k < kmax:
            print(linalg.norm( self.Grad_choice( x_k ) ),'norm')

            k += 1
            s_k =  dot( - H_k , grad_f_k ) 
            alpha = self.alpha( Method_for_a , x_k , s_k , alpha)
            x_kp1 = x_k + alpha * ( s_k )
            x_km1 = x_k
            x_k = x_kp1
            grad_f_k = self.Grad_choice( x_k )
            grad_f_km1 = self.Grad_choice( x_km1 )
            H_k = self.Update_Hersian( Method_for_H_update ,  H_k , x_k , x_km1 , grad_f_k , grad_f_km1 )
            if Method_for_H_update == 'BFGS Compare':

                Comprare_Result = self.Compare_BFGS(H_k,x_k)
                print(Comprare_Result,'This is how good your update is at step k',k)
            
        if all( linalg.eigvals( H_k ) > 0 ) and k != kmax and linalg.norm( self.Grad_choice( x_k ) ) != nan:
            print('Newton converged')

            return x_k 
        else:
            #print(linalg.eigvals( H_k ))
            print(x_k)
            raise Exception ( "k = " ,k , " If k == kmax you did not converge, otherwise your hersian is not PD or Grad_f(x) is nan" )
            
      
    def Grad_choice( self , x_k ):

        if self.optimisation_problem.Choice_grad:
            return self.Grad_F( x_k )
        else:
            return self.optimisation_problem.Grad_f( x_k )

    def alpha( self , Method_for_a , x_k , s_k ,alpha):
        
        if Method_for_a == 'Classical':
            return  1
        elif Method_for_a == 'Inexact':
            return self.inexact_a( x_k , s_k ,alpha)
        elif Method_for_a == 'Exact':
            return self.exact_a( x_k , s_k ,alpha)
        else:
            raise Exception("You have not choosen appropriet method for alpha method . Use format: Method_for_a = 'Classical','Inexact' or 'Exact' ")
    
        
    def exact_a( self , x_k , s_k ,alpha):

        fa = lambda a : self.optimisation_problem.f( x_k + a * s_k )
        a = so.minimize(fa , alpha)
        print(a.x)

        return a.x
        
    def inexact_a( self , x_k , s_k , alpha):
        
        a_l = 0.
        a_r = 1e99
        sigma = 0.7
        rho = 0.1
        a_0 = alpha  #### jag vet ej vad a_0 skall vara, detta skall nog vara en bra gissning, vi vill ta typ a_0 steg
        
        fa = lambda a : self.optimisation_problem.f( x_k + a * s_k )
        fa_prim_a_0 = self.f_a_prim( a_0 , x_k , s_k ) 
        f_a_prim_a_l = self.f_a_prim( a_l , x_k , s_k )
        fa_a_0 = fa(a_0)
        f_a_a_l = fa(a_l)
        
        while  not ( ( fa_prim_a_0 >= sigma*f_a_prim_a_l  ) and ( fa_a_0 <= f_a_a_l + rho*(a_0 - a_l)*f_a_prim_a_l ) ) :
            
            if   ( fa_prim_a_0 >= sigma*f_a_prim_a_l  ):
                #print('Block1' ,a_0 , a_r , a_l )
                #pause(7)
                a_0 , a_r , a_l = self.Block1( a_0 , a_l , a_r , fa_a_0 , f_a_a_l , fa_prim_a_0 , f_a_prim_a_l ) 
                fa_a_0 = fa(a_0)
                f_a_a_l = fa(a_l)
                fa_prim_a_0 = self.f_a_prim( a_0 , x_k , s_k ) 
                f_a_prim_a_l = self.f_a_prim( a_l , x_k , s_k )
                
            else:
                #print('Block2',a_0 , a_r , a_l)
                #pause(7)
                a_0 , a_r , a_l = self.Block2( a_0 , a_l , a_r , fa_a_0 , f_a_a_l , fa_prim_a_0 , f_a_prim_a_l )
                fa_a_0 = fa(a_0)
                f_a_a_l = fa(a_l)
                fa_prim_a_0 = self.f_a_prim( a_0 , x_k , s_k ) 
                f_a_prim_a_l = self.f_a_prim( a_l , x_k , s_k )

        #print(a_0,'a_0 out')
        return a_0 
    
    def Block1( self , a_0 , a_l , a_r , fa_a_0 , fa_a_l , fa_prim_a_0 , f_a_prim_a_l ):
        
        tao = 0.1
        Chi = 9.
        
        delta_a_0 = ( a_0 - a_l ) * ( ( fa_prim_a_0 ) / ( f_a_prim_a_l - fa_prim_a_0)  )
        print(delta_a_0,'Hej')
        delta_a_0 = max( delta_a_0 , tao * ( a_0 - a_l ) )
        delta_a_0 = min( delta_a_0 , Chi * ( a_0 - a_l ) )
        a_l = a_0
        a_0 = a_0 + delta_a_0
        
        return a_0 , a_r , a_l
        
    def Block2( self , a_0 , a_l , a_r , fa_a_0 , fa_a_l , fa_prim_a_0 , f_a_prim_a_l ):
        
        tao = 0.1
        
        a_r = min( a_0 , a_r )
        a_0_bar =  (( ( a_0 - a_l )**2 ) *  f_a_prim_a_l ) / (2*( fa_a_l - fa_a_0 + ( a_0 - a_l )*f_a_prim_a_l ))
        a_0_bar = max ( a_0_bar , a_l + tao * ( a_r - a_l ) )
        a_0_bar = min( a_0_bar , a_r - tao * ( a_r - a_l ) )
        a_0 = a_0_bar
        
        return a_0 , a_r , a_l
            
    def f_a_prim( self , a , x_k , s_k ):
        
        h = 1e-10
        F_a = lambda a : self.optimisation_problem.f( x_k + a * s_k )
        return ( F_a( a - h ) - F_a( a ) ) / h
    
    def Update_Hersian( self , Method_for_H_update , H_k , x_k , x_km1 , grad_f_k , grad_f_km1  ):
                         
        delta_k = x_k - x_km1
        gamma_k = grad_f_k - grad_f_km1 
        
        if Method_for_H_update == 'Good Broyden':
            return self.Good_Broyden( H_k , delta_k , gamma_k )
        elif Method_for_H_update == 'Bad Broyden':
            return self.Bad_Broyden( H_k , delta_k , gamma_k )
        elif Method_for_H_update == 'Symmetric Broyden':
            return self.Symmetric_Broyden( H_k , delta_k , gamma_k )
        elif Method_for_H_update == 'DFP':
            return self.DFP( H_k , delta_k , gamma_k )
        elif Method_for_H_update == 'BFGS' or Method_for_H_update == 'BFGS Compare' :
            return self.BFGS( H_k , delta_k , gamma_k )
            
        else:
            raise Exception("You have not choosen appropriet method for alpha method . Use format: Method_for_H = 'Good Broyden','Bad Broyden','Symmetric Broyden' ,'DFP' ,'BFGS' or 'BFGS Compare' ")
    
    def Good_Broyden( self , H_k , delta_k , gamma_k ):     
        
        H_k_gamma_k = dot( H_k , gamma_k )
        H_k_delta_k = dot( H_k , delta_k )
        denumerator = inner(H_k_delta_k ,gamma_k)
        H_kp1 = H_k + outer( delta_k - H_k_gamma_k , H_k_delta_k ) / denumerator
        
        return H_kp1
    
    def Bad_Broyden( self , H_k , delta_k , gamma_k ): 
        
        H_kp1 = H_k + outer( ( delta_k - dot( H_k , gamma_k) ) / inner( gamma_k , gamma_k )   , gamma_k )
        
        return H_kp1

    def Symmetric_Broyden(self, H_k , delta_k , gamma_k ):

        u = delta_k - dot(H_k,gamma_k)
        a = 1 / inner(u,gamma_k)

        H_kp1 = H_k + a * outer(u,u) 
        return H_kp1

    def DFP( self , H_k , delta_k , gamma_k ): 
        
        a_1 = outer( delta_k , delta_k )/inner( delta_k , delta_k )
        a_2 = dot(H_k , dot( outer( gamma_k , gamma_k ), H_k)  ) / inner( gamma_k , dot( H_k , gamma_k ))
        H_kp1 = H_k + a_1 - a_2 
        
        return H_kp1
        
    def BFGS( self , H_k , delta_k , gamma_k ): 
       
        a_1 = (1 + inner( gamma_k , dot( H_k , gamma_k )  ) / inner( delta_k , gamma_k ) ) * ( outer( delta_k , delta_k ) / inner( delta_k , gamma_k ) )
        a_2 = ( dot( outer( delta_k , gamma_k ) , H_k ) + dot( H_k , outer( gamma_k , delta_k ) ) ) / inner( delta_k , gamma_k )
        H_kp1 = H_k + a_1 - a_2
        
        return H_kp1
  
    
        
        
























def f(X):
    return 100*(X[1]-X[0]**2)**2+(1-X[0])**2



'''
x_intital = array([1.00004,1.00002])
op = Optimisation_problem(f)
om = Optimazation_methods(op, x_intital)


Detta är Task 10, med n = 4  , Note: x_i in [0,1] 
'''

x_2_initial = linspace(0,1,11)

op_2 = Optimisation_problem(chebyquad,gradchebyquad)
om_2 = Optimazation_methods(op_2,x_2_initial,1.3e-8)
'''
Note : tol = 1.3e-5 for Good Broyden and Inexact 
Note : tol = 3e-5 for BFGS and Inxact
 
'''

print(om_2.Newton('Exact', 'BFGS Compare'))   # Method_for_H:'Good Broyden','Bad Broyden','Symmetric Broyden' ,'DFP' , 'BFGS', 'BFGS Compare' 
                                        ####  Check Good Broyden , BFGS



start_time = time.time()

print("--- %s seconds ---" % (time.time() - start_time))

#[0.11270161 0.37090054 0.62909946 0.88729839] x_min

