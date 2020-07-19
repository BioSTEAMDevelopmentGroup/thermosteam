# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import flexsolve as flx
from ..exceptions import InfeasibleRegion
from ..utils.decorators import thermo_user
from . import binary_phase_fraction as binary
from .dew_point import DewPointCache
from .bubble_point import BubblePointCache
from .fugacity_coefficients import IdealFugacityCoefficients
from .._thermal_condition import ThermalCondition
from ..indexer import MolarFlowIndexer
from .. import functional as fn
from ..utils import Cache
import numpy as np

__all__ = ('SLE',)

@thermo_user
class SLE:
    __slots__ = ('_x', # [float] Fraction of solute as a solid.
                 '_gamma', # [ActivityCoefficients] Estimates activity coefficients of a liquid.
                 '_liquid_mol', # [1d array] Liquid molar data.
                 '_solid_mol', # [1d array] Solid molar data.
                 '_phase_data', # tuple[str, 1d array] Phase-data pairs.
                 '_index', # [1d array] Index of chemicals in equilibrium.
                 '_chemical', # [Chemical] Single chemical in equilibrium.
                 '_nonzero', # [1d array(bool)] Chemicals present in the mixture
                 '_mol_solute', # [float] Solute molar data.
    )
    
    def __init__(self, imol=None, thermal_condition=None, thermo=None):
        super().__init__(imol, thermal_condition, thermo)
        imol = self._imol
        self._phase_data = tuple(imol)
        self._liquid_mol = liquid_mol = imol['l']
        self._solid_mol = imol['s']
        self._nonzero = np.zeros(liquid_mol.shape, dtype=bool)
        self._index = ()
        self._chemical = None
        self._x = None
    
    def _setup(self, solute):
        # Get flow rates
        liquid_mol = self._liquid_mol
        vapor_mol = self._vapor_mol
        mol = liquid_mol + vapor_mol
        nonzero = mol > 0
        chemicals = self.chemicals
        solute_index = chemicals.get_index(solute)
        self._mol_solute = mol_solute = nonzero[solute_index]
        if not mol_solute:
            raise RuntimeError('no solute available')
        if (self._nonzero == nonzero).all():
            index = self._index
        else:
            # Set up indices for both equilibrium and non-equilibrium species
            index = chemicals.get_lle_indices(nonzero)   
            N = len(index)
            if N < 2:
                raise RuntimeError('at least 2 chemicals are required for SLE')
            else:
                # Set equilibrium objects
                eq_chems = chemicals.tuple
                eq_chems = [eq_chems[i] for i in index]
                self._nonzero = nonzero
                self._index = index
                thermo = self._thermo
                self._gamma = thermo.Gamma(eq_chems)
        if mol.sum() != 0: 
            raise RuntimeError('no chemicals to perform equilibrium')
        self._mol = mol[index]
        
        