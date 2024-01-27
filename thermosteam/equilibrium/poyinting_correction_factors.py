# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
__all__ = ('PoyintingCorrectionFactors',
           'MockPoyintingCorrectionFactors',
           'IdealGasPoyintingCorrectionFactors')
from ..constants import R
from numba import njit
import numpy as np

class PoyintingCorrectionFactors:
    """Abstract class for the estimation of Poyinting correction factors.
    Non-abstract subclasses should implement the following methods:
        
    __init__(self, chemicals: Iterable[:class:`~thermosteam.Chemical`]):
        Should use pure component data from chemicals to setup future 
        calculations of Poyinting correction factors.
    
    __call__(self, T: float, P: float):
        Should accept the temperature `T` (in Kelvin) and pressure `P` (in Pascal),
        and return an array of Poyinting correction factors. 
        
    """
    __slots__ = ()
    
    @property
    def chemicals(self):
        return self._chemicals
    @chemicals.setter
    def chemicals(self, chemicals):
        self._chemicals = tuple(chemicals)
    
    def __init__(self, chemicals):
        self.chemicals = chemicals
    
    def __reduce__(self):
        return type(self), (self.chemicals,)
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"<{type(self).__name__}([{chemicals}])>"


class MockPoyintingCorrectionFactors(PoyintingCorrectionFactors):
    """Create a MockPoyintingCorrectionFactors object 
    that estimates all poyinting correction factors to be 1 when
    called with a temperature (K) and pressure (Pa).
    
    Parameters
    ----------
    
    chemicals : Iterable[:class:`~thermosteam.Chemical`]
    
    """
    __slots__ = ('_chemicals',)
    
    def __call__(self, T, P, Psats=None):
        return 1.


@njit(cache=True)
def ideal_gas_poyinting_correction_factors(T, P, vls, Psats):
    dP = P - Psats
    dP[dP < 0.] = 0.
    return np.exp(vls / (R * T) * dP)

class IdealGasPoyintingCorrectionFactors(PoyintingCorrectionFactors):
    """Create an IdealGasPoyintingCorrectionFactors object that estimates 
    poyinting correction factors assuming ideal gas when called with 
    a temperature (K) and pressure (Pa).
    
    Parameters
    ----------
    
    chemicals : Iterable[:class:`~thermosteam.Chemical`]
    
    """
    __slots__ = ('_chemicals',)

    def __call__(self, T, P, Psats=None):
        vls = []
        chemicals = []
        index = []
        for i, chemical in enumerate(self._chemicals):
            try: vls.append(chemical.V('l', T, P))
            except: continue
            chemicals.append(chemical)
            index.append(i)
        vls = np.array(vls, dtype=float)
        if Psats is None:
            Psats = np.array([i.Psat(T) for i in chemicals], dtype=float)
        else:
            Psats = Psats[index]
        pcf = np.ones(len(self._chemicals))
        pcf[index] = ideal_gas_poyinting_correction_factors(T, P, vls, Psats)
        return pcf
        