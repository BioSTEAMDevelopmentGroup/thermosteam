# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
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
    >>> f_l = F_l(x=liquid_molar_composition, T=355)
    >>> f_l
    array([43338.226, 58056.67 ])
    
    """
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
