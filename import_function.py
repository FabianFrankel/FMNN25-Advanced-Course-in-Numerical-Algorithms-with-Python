#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 18:47:46 2020

@author: Mikael
"""
from scipy import *
from matplotlib.pyplot import *




class import_function:
    
    def F(self, x_1 , x_2 ):
        '''
        This is our imported function
        '''
        
        return 100 * ( x_2 - x_1 )**2 + ( 1 - x_1)**2
        
    
    
    
    def Grad_F(self, x_1 , x_2 ):
        
        
        
        return array( [ 202*x_1 - 200*x_2 - 2 , 200* (x_2 -x_1 ) ] )
    
    
    def Grad_F2(self, x_1 , x_2 ):
        
        h = 1e-4
        
        return array( [ (self.F( x_1 + h , x_2 ) - self.F( x_1 , x_2 ) )/ h , (self.F( x_1 , x_2 + h) - self.F( x_1 ,x_2 ) ) / h ] )



s = import_function()

s_1 = s.Grad_F(1,2)

print(s_1)

s_2 = s.Grad_F2(1,2)

print(s_2)









