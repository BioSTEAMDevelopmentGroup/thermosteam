# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 07:11:01 2020

@author: yoelr
"""
from .fugacity_coefficients import LiquidFugacities, GasFugacities

__all__ = ('PartitionCoefficients',)

class PartitionCoefficients:
    __slots__ = ('f_g', 'f_l')
    
    def __init__(self, chemicals, thermo=None):
        self.f_g = GasFugacities(chemicals, thermo)
        self.f_l = LiquidFugacities(chemicals, thermo)
    
    def __call__(self, x, y, T, P):
        return self.f_g(y, T, P) / self.f_l(x, T)
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"