# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 07:45:30 2019

@author: yoelr
"""

__all__ = ('ThermalCondition',)

class ThermalCondition:
    __slots__ = ('_T', '_P')
    
    def __init__(self, T, P):
        self.T = T
        self.P = P
    
    @property
    def T(self):
        return self._T
    @T.setter
    def T(self, T):
        self._T = float(T)
    @property
    def P(self):
        return self._P
    @P.setter
    def P(self, P):
        self._P = float(P)
    
    def copy(self):
        return self.__class__(self.T, self.P)
    
    def copy_like(self, other):
        self.T = other.T
        self.P = other.P
    
    @property
    def tuple(self):
        return self.T, self.P
    
    def __iter__(self):
        yield self.T
        yield self.P
        
    def __repr__(self):
        return f"{type(self).__name__}(T={self.T:.2f}, P={self.P:.6g})"