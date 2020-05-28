# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from flexsolve import InfeasibleRegion

__all__ = ('ThermalCondition',)

class ThermalCondition:
    """
    Create a ThermalCondition object that contains temperature and pressure values.
    
    Parameters
    ----------
    T : float
        Temperature [K].
    P : float
        Pressure [Pa].
    
    """
    __slots__ = ('_T', '_P')
    
    def __init__(self, T, P):
        self.T = T 
        self.P = P 
    
    @property
    def T(self):
        """[float] Temperature in Kelvin."""
        return self._T
    @T.setter
    def T(self, T):
        T = float(T)
        if T < 0.: raise InfeasibleRegion('temperature')
        self._T = T
    
    @property
    def P(self):
        """[float] Pressure in Pascal."""
        return self._P
    @P.setter
    def P(self, P):
        P = float(P)
        if P < 0.: raise InfeasibleRegion('pressure')
        self._P = P
    
    def in_equilibrium(self, other):
        """Return whether thermal condition is in equilibrium with another
        (i. e. same temperature and pressure)."""
        return abs(self._T - other._T) < 1e-12 and abs(self._P - other._P) < 1e-12
    
    def copy(self):
        """Return a copy."""
        return self.__class__(self._T, self._P)
    
    def copy_like(self, other):
        """Copy the specifications of another ThermalCondition object."""
        self._T = other._T
        self._P = other._P
    
    @property
    def tuple(self):
        """tuple[float, float] Temperature and pressure"""
        return (self._T, self._P)
    
    def __getitem__(self, index):
        if index == 0:
            return self._T
        elif index == 1:
            return self._P
        else:
            raise IndexError('index out of range')
    
    def __setitem__(self, index, value):
        if index == 0:
            self.T = value
        elif index == 1:
            self.P = value
        else:
            raise IndexError('index out of range')
    
    def __iter__(self):
        return iter(self.tuple)
        
    def __repr__(self):
        return f"{type(self).__name__}(T={self.T:.2f}, P={self.P:.6g})"