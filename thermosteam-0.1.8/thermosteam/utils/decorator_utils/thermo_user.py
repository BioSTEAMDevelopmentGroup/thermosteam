# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:59:17 2019

@author: yoelr
"""
import thermosteam as tmo

__all__ = ('thermo_user',)

def thermo_user(cls):
    cls._load_thermo = _load_thermo
    cls.thermo = thermo
    cls.chemicals = chemicals
    cls.mixture = mixture
    return cls
    
def _load_thermo(self, thermo):
    self._thermo = thermo = tmo.settings.get_thermo(thermo)
    return thermo

@property
def thermo(self):
    return self._thermo
@property
def chemicals(self):
    return self._thermo.chemicals
@property
def mixture(self):
    return self._thermo.mixture