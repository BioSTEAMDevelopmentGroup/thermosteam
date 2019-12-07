# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 07:45:30 2019

@author: yoelr
"""

__all__ = ('ThermalCondition',)

class ThermalCondition:
    __slots__ = ('T', 'P')
    
    def __init__(self, T, P):
        self.T = T
        self.P = P
        
    def __iter__(self):
        yield self.T
        yield self.P
        
    def __repr__(self):
        return f"{type(self).__name__}(T={self.T:.2f}, P={self.P:.6g})"
    