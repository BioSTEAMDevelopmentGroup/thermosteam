# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import thermosteam as tmo
import numpy as np

__all__ = ('LiquidFugacities', 'GasFugacities', 'Fugacities',
           'fugacities_by_phase')

class LiquidFugacities:
    """
    Create a LiquidFugacities capable of computing fugacities of chemicals
    in the liquid phase when called with a composition vector (1d array)
    and a temperature.
    
    Parameters
    ----------
    chemicals : Iterable[:class:`~thermosteam.Chemical`]
        Chemicals to compute fugacities.
    thermo : :class:`~thermosteam.Thermo`, optional
        Thermodynamic property package.
    
    Examples
    --------
    >>> import thermosteam as tmo
    >>> import numpy as np
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'])
    >>> tmo.settings.set_thermo(chemicals)
    >>> # Create a LiquidFugacities object
    >>> F_l = tmo.equilibrium.LiquidFugacities(chemicals)
    >>> F_l
    LiquidFugacities([Water, Ethanol])
    >>> # Compute liquid fugacities
    >>> liquid_molar_composition = np.array([0.72, 0.28])
    >>> f_l = F_l(x=liquid_molar_composition, T=355, P=101325.)
    >>> f_l
    array([43338.226, 57731.001])
    
    """
    __slots__ = ('gamma', 'chemicals', 'pcf')
    
    def __init__(self, chemicals, thermo=None):
        thermo = tmo.settings.get_default_thermo(thermo)
        self.chemicals = chemicals = tuple(chemicals)
        self.gamma = thermo.Gamma(chemicals)
        self.pcf = thermo.PCF(chemicals)
    
    def unweighted(self, x, T, P=101325.):
        Psats = np.array([i.Psat(T) for i in self.chemicals], dtype=float)
        return self.gamma(x, T, P) * self.pcf(T, P, Psats) * Psats
    
    def __call__(self, x, T, P=101325., reduce=False):
        f_reduced = x * self.gamma(x, T)
        if reduce: return f_reduced
        Psats = np.array([i.Psat(T) for i in self.chemicals], dtype=float)
        return f_reduced * Psats * self.pcf(T, P, Psats)
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"


class GasFugacities:
    """
    Create a GasFugacities capable of computing fugacities of chemicals
    in the liquid phase when called with a composition vector (1d array)
    and a temperature.
    
    Parameters
    ----------
    chemicals : Iterable[Chemicals]
        Chemicals to compute fugacities.
    thermo : Thermo, optional
        Thermodynamic property package.
    
    Examples
    --------
    >>> import thermosteam as tmo
    >>> import numpy as np
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'])
    >>> tmo.settings.set_thermo(chemicals)
    >>> # Create a GasFugacities object
    >>> F_g = tmo.equilibrium.GasFugacities(chemicals)
    >>> F_g
    GasFugacities([Water, Ethanol])
    >>> # Compute gas fugacities
    >>> gas_molar_composition = np.array([0.43, 0.57])
    >>> f_g = F_g(y=gas_molar_composition, T=355, P=101325)
    >>> f_g
    array([43569.7, 57755.2])

    """
    __slots__ = ('phi', 'chemicals')
    
    def __init__(self, chemicals, thermo=None):
        thermo = tmo.settings.get_default_thermo(thermo)
        self.chemicals = chemicals = tuple(chemicals)
        self.phi = thermo.Phi(chemicals)
    
    def unweighted(self, y, T, P):
        return P * self.phi(y, T, P)
    
    def __call__(self, y, T, P, reduce=False):
        f_reduced = self.phi(y, T, P) * y
        if reduce: return f_reduced
        return P * f_reduced
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"


class Fugacities:
    __slots__ = ('fugacities_by_phase',)
    
    def __init__(self, chemicals, thermo=None):
        self.fugacities_by_phase = fugacities_by_phase(chemicals, thermo)
        
    def unweighted(self, phase, z, T, P):
        return self.fugacities_by_phase[phase].unweighted(z, T, P)
        
    def __call__(self, phase, z, T, P, reduce=False):
        return self.fugacities_by_phase[phase](z, T, P, reduce)
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"
        
def fugacities_by_phase(chemicals, thermo):
    thermo = tmo.settings.get_default_thermo(thermo)
    G = GasFugacities(chemicals, thermo)
    L = LiquidFugacities(chemicals, thermo)
    return {
        'g': G,
        'l': L,
        'L': L,
    }