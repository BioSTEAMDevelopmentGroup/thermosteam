# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:40:24 2019

@author: yoelr
"""
__all__ = ('FugacityCoefficients', 
           'IdealFugacityCoefficients')

class FugacityCoefficients:
    """Abstract class for the estimation of fugacity coefficients. Non-abstract subclasses should implement the following methods:
        
    __init__(self, chemicals: Iterable[Chemicals]):
        Should use pure component data from chemicals to setup future calculations of fugacity coefficients.
    
    __call__(self, y: 1d array, T: float, P:float):
        Should accept an array of vapor molar compositions `y`, temperature `T` (in Kelvin), and pressure `P` (in Pascal), and return an array of fugacity coefficients. Note that the molar compositions must be in the same order as the chemicals defined when creating the FugacityCoefficients object.
        
    """
    __slots__ = ()
    
    def __init__(self, chemicals):
        self.chemicals = chemicals
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"<{type(self).__name__}([{chemicals}])>"


class IdealFugacityCoefficients(FugacityCoefficients):
    """Create an IdealFugacityCoefficients object that estimates all fugacity coefficients to be 1 when called with composition, temperature (K), and pressure (Pa).
    
    Parameters
    ----------
    
    chemicals : Iterable[Chemical]
    
    """
    __slots__ = ('_chemicals')
    
    @property
    def chemicals(self):
        """tuple[Chemical] All chemicals involved in the calculation of fugacity coefficients."""
        return self._chemicals
    @chemicals.setter
    def chemicals(self, chemicals):
        self._chemicals = tuple(chemicals)

    def __call__(self, y, T, P):
        return 1.

    
    
    
