# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 07:45:30 2019

@author: yoelr
"""

__all__ = ('Condition',)

class Condition:
    __slots__ = ('T', 'P')
    
    def __init__(self, T, P):
        self.T = T
        self.P = P
        
    def _ipython_display_(self):
        return f"{type(self).__name__}(T={self.T:.2f} K, P={self.P:.6g} Pa)"
        
    def __repr__(self):
        return f"{type(self).__name__}(T={self.T:.2f}, P={self.P:.6g})"
    