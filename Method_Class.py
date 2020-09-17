#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:20:59 2020

@author: Mikael
"""
from scipy import *
from matplotlib.pyplot import *
import timeit
    
class Optimazation_methods:
    
    def Grad_F( self , X ):
        
        h = 1e-4
        n = shape( X )[0]
        grad_vector = zeros(n)
        H = zeros(n)
   
        for i in range(n):
            H[i] = h
            grad_vector[i] = ( self.F( X + H ) - self.F( X ) ) /h
            H[i] = 0
        
        return grad_vector
    
    def Hersian( self , X ):
        h = 1e-4
        n = X.shape[0]
        Hersian = zeros( (n,n) )
        H = zeros(n)
        
        for i in range(n):
            H[i] = h
            Hersian[:,i] = ( self.Grad_F( X + H ) - self.Grad_F( X ) )/h
            H[i] = 0
        
        Her_F = (1/2) * ( Hersian + Hersian.T )
        
        if all( eigvals( Her_F ) > 0 ):
            
            return Her_F
        
        
        else:
            raise Exception( "Sorry The Hersian Is Not Positive Definite, Try Again" )
            
    def Newton( self , Method_for_a , Method_for_H ):
        
        x_k = self.initial_x.copy()
        grad_f_k = self.Grad_F(x_k)
        Her_f_k = self.Hersian(x_k)
        H_k = inv(Her_f_k)
        k = 0
        
        while norm( self.Grad_F( x_k ) ) > 1e-7 and k <= 100:
            
            k += 1
            s_k = - dot( H_k , grad_f_k ) 
            x_kp1 = x_k + self.alpha( Method_for_a , x_k , s_k ) * ( s_k )
            x_k = x_kp1.copy()
            grad_f_k = self.Grad_F( x_k )
            ####H_k = self.Update_Hersian( Method_for_h , x_k , H_k ,  )
            
        if all( eigvals( H_k ) > 0 ) and k != 100:
            
            return x_k
        else:
            raise Exception ("k = " ,k , " If k == 100 you did not converge, otherwise your hersian is not PD " )
            
        
    def alpha( self , Method_for_a , x_k , s_k ):
        
        Classical = [['Classical']]  ##### Detta kan man säkert göra snyggare, men det var typ såhär jag tänkte
        Inexact = [['Inexact']]      #####
        Exact = [['Exact']]          #####
        
        if Method_for_a in Classical:
            return - 1
        elif Method_for_a in Inexact:
            return self.inexact_a( x_k , s_k )
        elif Method_for_a in Exact:
            return self.exact_a( x_k , s_k )
        else:
            raise Exception("You have not choosen appropriet method for alpha method . Use format: Method_for_a = ['Classical'],['Inexact'] or ['Exact']")
    
        
    def exact_a( self , x_k , s_k ):
        fa = lambda a : self.F( x_k + a * s_k )
        a = scipy.optimize.minimize(fa)
        return a
        
    def inexact_a( self , x_k , s_k ):
        
        a_l = 0
        a_r = 1e7
        sigma = 0.7
        rho = 0.1
        a_0 = (a_l + a_r) / 2
        
        fa = lambda a : self.F( x_k + a * s_k )
        fa_prim_a_0 = self.Grad_F_a( a_0 , x_k , s_k ) 
        f_a_prim_a_l = self.Grad_F_a( a_l , x_k , s_k )
        fa_a_0 = fa(a_0)
        f_a_a_l = fa(a_l)
        
        while not ( fa_prim_a_0 >= f_a_prim_a_l  ) and not ( fa_a_0 <= fa_a_l + rho*f_a_prim_a_l ):
            
            if not ( fa_prim_a_0 >= f_a_prim_a_l  ):
                
                a_0 , a_r , a_l = self.Block1( a_0 , a_l , a_r , fa_a_0 , fa_a_l , fa_prim_a_0 , f_a_prim_a_l ) 
            
            else:
                
                a_0 , a_r , a_l = self.Block2( a_0 , a_l , a_r , fa_a_0 , fa_a_l , fa_prim_a_0 , f_a_prim_a_l )
                
        return a_0 
    
    def Block1( self , a_0 , a_l , a_r , fa_a_0 , fa_a_l , fa_prim_a_0 , f_a_prim_a_l ):
        
        tao = 0.1
        Chi = 9.
        
        delta_a_0 = ( a_0 - a_l ) * ( ( fa_prim_a_0 ) / fa_prim_a_l - fa_prim_a_0 )
        delta_a_0 = max( delta_a_0 , tao * ( a_0 - a_l ) )
        delta_a_0 = min( delta_a_0 , Chi* ( a_0 - a_l ) )
        a_l = a_0
        a_0 = a_0 + delta_a_0
        
        return a_0 , a_r , a_l
        
    def Block2( self , a_0 , a_l , a_r , fa_a_0 , fa_a_l , fa_prim_a_0 , f_a_prim_a_l ):
        
        tao = 0.1
        
        a_r = min( a_0 , a_r )
        a_0_bar =  ((( ( a_0 - a_l )**2 )*  f_a_prim_a_l ) / (2*( fa_a_l - fa_a_0 + ( a_0 - a_l )*f_a_prim_a_l )))  
        a_0_bar = max ( a_0_bar , a_l + tao * ( a_r - a_l ) )
        a_0_bar = min( a_0_bar , a_u - tao * ( a_r - a_l ) )
        a_0 = a_0_bar
        
        return a_0 , a_r , a_l
            
    def Grad_F_a( self , X , x_k , s_k ):
        
        h = 1e-4
        n = shape( X )[0]
        grad_vector = zeros(n)
        H = zeros(n)
        F_a = lambda a : self.F( x_k + a * s_k )
        for i in range(n):
            H[i] = h
            grad_vector[i] = ( Fa( X + H ) - F_a( X ) ) /h
            H[i] = 0
        
        return grad_vector
    

    def Update_Hersian( self , Method_for_h , H_k , x_k , x_km1 , grad_f_k , grad_f_km1  ):
        
        
        Good_Broyden = [['Good Broyden']]  ##### Detta kan man säkert göra snyggare, men det var typ såhär jag tänkte
        Bad_Broyden = [['Bad Broyden']]    #####
        DFP = [['DFP']]                    #####
        BFGS = [['BFGS']]                  #####
        delta_k = x_k - x_km1
        gamma_k = grad_f_k - grad_f_km1 
        
        if Method_for_h in Good_Broyden:
            return self.Good_Broyden( H_k , delta_k , gamma_k )
        elif Method_for_h in Bad_Broyden:
            return self.Bad_Broyden( H_k , delta_k , gamma_k )
        elif Method_for_h in DFP:
            return self.DFP( H_k , delta_k , gamma_k )
        elif Method_for_h in BFGS:
            return self.BFGS( H_k , delta_k , gamma_k )
        else:
            raise Exception("You have not choosen appropriet method for alpha method . Use format: Method_for_H = ['Good Broyden'],['bad Broyden'], ['DFP'] or ['BFGS'] ")
    
    def Good_Broyden( self , H_k , delta_k , gamma_k ):     ##### Jag tror detta är Bad Broyden, men jag kan blandat ihop dem
        
        H_k_gamma_k = dot( H_k , gamma_k )
        H_k_delta_k = dot( H_k , delta_k )
        denumerator = inner(H_k_delta_k ,gamma_k)
        
        H_kp1 = H_k + outer( gamma_k - H_k_gamma_k , H_k_delta_k )/denumerator
        
        return H_kp1
    
    def Bad_Broyden( self , H_k , delta_k , gamma_k ): 
        
        H_kp1 = H_k + outer( ( delta_k - dot( H_k , gamma_k) )/inner( gamma_k , gamma_k )   , gamma_k )
        
        return H_kp1
    
    def DFP( self , H_k , delta_k , gamma_k ): 
        
        a_1 = outer( delta_k , delta_k )/inner( delta_k , delta_k )
        a_2 = dot(H_k , dot( outer( gamma_k , gamma_k ), H_k)  ) / inner( gamma , dot( H_k , gamma_k ))
        H_kp1 = H_k + a_1 - a_2 
        
        return H_kp1
        
    def BFGS( self , H_k , delta_k , gamma_k ): 
       
        a_1 = (1 + inner( gamma_k , dot( H_k , gamma_k )  ) / inner( delta_k , gamma_k ) ) * ( outer( gamma_k , gamma_k ) / inner( delta_k , gamma_k ) )
        a_2 = ( dot( outer( delta_k , gamma_k ) , H_k ) + dot( H_k , outer( gamma_k , delta_k ) ) ) / inner( delta_k , gamma_k )
        H_kp1 = H_k + a_1 - a_2
        
        return H_kp1
    
    
        
    
        
    
    
        
    
        
        






















start_time = time.time()

print("--- %s seconds ---" % (time.time() - start_time))
