# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import thermosteam as tmo
import numpy as np

__all__ = ('LiquidFugacities', 'GasFugacities')

class LiquidFugacities:
    __slots__ = ('gamma', 'chemicals')
    
    def __init__(self, chemicals, thermo=None):
        thermo = tmo.settings.get_default_thermo(thermo)
        self.chemicals = chemicals = tuple(chemicals)
        self.gamma = thermo.Gamma(chemicals)
    
    def __call__(self, x, T):
        return x * self.gamma(x, T) * np.array([i.Psat(T) for i in self.chemicals], dtype=float)
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"
    
class GasFugacities:
    __slots__ = ('phi', 'pcf', 'chemicals')
    
    def __init__(self, chemicals, thermo=None):
        thermo = tmo.settings.get_default_thermo(thermo)
        self.chemicals = chemicals = tuple(chemicals)
        self.pcf = thermo.PCF(chemicals)
        self.phi = thermo.Phi(chemicals)
    
    def __call__(self, y, T, P):
        return P * self.pcf(y, T) * self.phi(y, T, P) * y
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"