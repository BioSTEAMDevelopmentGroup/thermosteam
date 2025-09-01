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
from .._phase import check_phase

__all__ = ('PartitionCoefficients',)


class PartitionCoefficients:
    """
    Create a PartitionCoefficients capable of computing partition coefficients 
    when called with a composition vectors (1d arrays) for the two phases, the temperature,
    and the pressure.
    
    Parameters
    ----------
    phases : str, optional
        Phases in equilibrium.
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
    >>> # Create a PartitionCoefficients object
    >>> K_gl = tmo.equilibrium.PartitionCoefficients('gl', chemicals)
    >>> K_gl
    PartitionCoefficients('gl', [Water, Ethanol])
    >>> # Compute partition coefficients
    >>> liquid_molar_composition = np.array([0.72, 0.28])
    >>> vapor_molar_composition = np.array([0.28, 0.72])
    >>> k_gl = K_gl(y=vapor_molar_composition, x=liquid_molar_composition, T=355, P=101325)
    >>> k_gl
    array([0.594, 2.035])
    
    """
    __slots__ = ('phases', 'fugacities_by_phase')
    
    def __init__(self, phases, chemicals, thermo=None):
        if len(phases) != 2: raise ValueError('phases argument must consist of two phases')
        for i in phases: check_phase(i)
        self.phases = phases
        self.fugacities_by_phase = tmo.equilibrium.fugacities_by_phase(chemicals, thermo)
    
    def __call__(self, y, x, T, P):
        top_phase, bottom_phase = self.phases
        return (
            self.fugacities_by_phase[bottom_phase].unweighted(x, T, P)
            / self.fugacities_by_phase[top_phase].unweighted(y, T, P)
        )
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.fugacities[self.phases[0]].chemicals])
        return f"{type(self).__name__}({self.phases!r}, [{chemicals}])"

