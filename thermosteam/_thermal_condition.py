# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 07:45:30 2019

@author: yoelr
"""

__all__ = ('ThermalCondition',)

class ThermalCondition:
    """Create a ThermalCondition object that contains temperature and pressure values.
    
    Parameters
    ----------
    T : float
        Temperature [K].
    P : float
        Pressure [Pa].
    
    """
    __slots__ = ('T', 'P')
    
    def __init__(self, T, P):
        self.T = T #: [float] Temperature in Kelvin
        self.P = P #: [float] Pressure in Pascal
    
    def copy(self):
        """Return a copy."""
        return self.__class__(self.T, self.P)
    
    def copy_like(self, other):
        """Copy the specifications of another ThermalCondition object."""
        self.T = other.T
        self.P = other.P
    
    @property
    def tuple(self):
        """tuple[float, float] Temperature and pressure"""
        return (self.T, self.P)
    
    def __iter__(self):
        yield self.T
        yield self.P
        
    def __repr__(self):
        return f"{type(self).__name__}(T={self.T:.2f}, P={self.P:.6g})"