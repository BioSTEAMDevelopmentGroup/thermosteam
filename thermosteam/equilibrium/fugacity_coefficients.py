# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:40:24 2019

@author: yoelr
"""
__all__ = ('FugacityCoefficients', 
           'IdealFugacityCoefficients')

class FugacityCoefficients:
    __slots__ = ()
    
    def __init__(self, chemicals):
        self.chemicals = chemicals
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"<{type(self).__name__}([{chemicals}])>"


class IdealFugacityCoefficients(FugacityCoefficients):
    __slots__ = ('_chemicals')
    
    @property
    def chemicals(self):
        return self._chemicals
    @chemicals.setter
    def chemicals(self, chemicals):
        self._chemicals = tuple(chemicals)

    def __call__(self, y, T, P):
        return 1.

    
    
    
