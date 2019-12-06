# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 07:45:30 2019

@author: yoelr
"""

__all__ = ('ThermalCondition',)

class ThermalCondition:
    __slots__ = ('phase', 'T', 'P')
    
    def __init__(self, phase, T, P):
        self.phase = phase
        self.T = T
        self.P = P
        
    def __iter__(self):
        yield self.phase
        yield self.T
        yield self.P
        
    def __repr__(self):
        return f"{type(self).__name__}(phase={repr(self.phase)}, T={self.T:.2f}, P={self.P:.6g})"
    