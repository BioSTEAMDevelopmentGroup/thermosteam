# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:18:44 2019

@author: yoelr
"""

__all__ = ('PoyintingCorrectionFactor',
           'IdealPoyintingCorrectionFactor')

class PoyintingCorrectionFactor:
    __slots__ = ()
    
    def __init__(self, chemicals):
        self.chemicals = chemicals
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"<{type(self).__name__}([{chemicals}])>"


class IdealPoyintingCorrectionFactor(PoyintingCorrectionFactor):
    __slots__ = ('_chemicals')
    
    @property
    def chemicals(self):
        return self._chemicals
    @chemicals.setter
    def chemicals(self, chemicals):
        self._chemicals = tuple(chemicals)

    def __call__(self, y, T):
        return 1.
