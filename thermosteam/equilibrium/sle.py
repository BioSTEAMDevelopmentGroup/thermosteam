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
from ..utils import Cache
from .equilibrium import Equilibrium
from chemicals import solubility_eutectic
import numpy as np

__all__ = ('SLE', 'SLECache')

class SLE(Equilibrium, phases='ls'):
    """
    Create an SLE object that performs solid-liquid equilibrium for a given solute
    when called with a temperature and pressure.
        
    Parameters
    ----------
    imol=None : MaterialIndexer, optional
        Molar chemical phase data is stored here.
    thermal_condition=None : ThermalCondition, optional
        The temperature and pressure used in calculations are stored here.
    thermo=None : Thermo, optional
        Themodynamic property package for equilibrium calculations.
        Defaults to `thermosteam.settings.get_thermo()`.
    
    Examples
    --------
    Solve SLE of tetradecanol in octanol:
        
    >>> from thermosteam import indexer, equilibrium, settings
    >>> settings.set_thermo(['Methanol', 'Tetradecanol'], cache=True)
    >>> imol = indexer.MolarFlowIndexer(l=[('Methanol', 10), ('Tetradecanol', 30)], phases=('s', 'l'))
    >>> sle = equilibrium.SLE(imol)
    >>> sle('Tetradecanol', T=300)
    >>> sle
    SLE(imol=MolarFlowIndexer(
            l=[('Methanol', 10), ('Tetradecanol', 5.721)],
            s=[('Tetradecanol', 24.28)]),
        thermal_condition=ThermalCondition(T=300.00, P=101325))
    
    Solve SLE of pure tetradecanol:
    
    >>> from thermosteam import indexer, equilibrium, settings
    >>> settings.set_thermo(['Octanol', 'Tetradecanol'], cache=True)
    >>> imol = indexer.MolarFlowIndexer(l=[('Tetradecanol', 30)], phases=('s', 'l'))
    >>> sle = equilibrium.SLE(imol)
    >>> sle('Tetradecanol', T=300) # Under melting point
    >>> sle
    SLE(imol=MolarFlowIndexer(phases=('l', 's'),
            s=[('Tetradecanol', 30)]),
        thermal_condition=ThermalCondition(T=300.00, P=101325))
    >>> sle('Tetradecanol', T=320) # Over melting point
    >>> sle
    SLE(imol=MolarFlowIndexer(phases=('l', 's'),
            l=[('Tetradecanol', 30)]),
        thermal_condition=ThermalCondition(T=320.00, P=101325))
    
    
    """
    __slots__ = ('_x', # [float] Fraction of solute as a solid.
                 '_gamma', # [ActivityCoefficients] Estimates activity coefficients of a liquid.
                 '_liquid_mol', # [1d array] Liquid molar data.
                 '_solid_mol', # [1d array] Solid molar data.
                 '_phase_data', # tuple[str, 1d array] Phase-data pairs.
                 '_index', # [1d array] Index of chemicals in equilibrium.
                 '_chemical', # [Chemical] Single chemical in equilibrium.
                 '_nonzero', # [1d array(bool)] Chemicals present in the mixture
                 '_mol_solute', # [float] Solute molar data.
                 '_solute_index', # [int] Solute index
                 '_solute_gamma_index', # [int] Solute index for activity coefficients
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
    
    def _setup(self):
        # Get flow rates
        liquid_mol = self._liquid_mol
        solid_mol = self._solid_mol
        mol = liquid_mol + solid_mol
        solute_index = self._solute_index
        self._mol_solute = mol_solute = mol[solute_index]
        if not mol_solute:
            raise RuntimeError('no solute available')
        nonzero = mol > 0
        if (self._nonzero == nonzero).all():
            index = self._index
        else:
            chemicals = self.chemicals
            # Set up indices for both equilibrium and non-equilibrium species
            index = chemicals.get_lle_indices(nonzero)   
            N = len(index)
            if N == 1:
                assert solute_index == index[0], "solute is not valid"
                self._chemical = chemicals.tuple[solute_index]
            else:
                # Set equilibrium objects
                eq_chems = chemicals.tuple
                eq_chems = [eq_chems[i] for i in index]
                self._nonzero = nonzero
                self._index = index
                thermo = self._thermo
                self._gamma = thermo.Gamma(eq_chems)
                self._solute_gamma_index = self._index.index(solute_index)
        
    def __call__(self, solute, T, P=None):
        thermal_condition = self.thermal_condition
        if P: thermal_condition.P = P
        thermal_condition.T = T
        chemicals = self.chemicals
        self._solute_index = solute_index = chemicals.get_index(solute)
        self._setup()
        liquid_mol = self._liquid_mol
        solid_mol = self._solid_mol
        if self._chemical:
            Tm = self._chemical.Tm
            if T > Tm:
                liquid_mol[solute_index] = self._mol_solute
                solid_mol[solute_index] = 0.
            else:
                liquid_mol[solute_index] = 0.
                solid_mol[solute_index] = self._mol_solute
        else:
            x = self._solve_x(T)
            self._update_solubility(x)
    
    def _update_solubility(self, x):
        solute_index = self._solute_index
        liquid_mol = self._liquid_mol
        solid_mol = self._solid_mol
        F_mol_liquid = liquid_mol[self._index].sum() - liquid_mol[solute_index]
        mol_solute = self._mol_solute
        x_max = mol_solute / (F_mol_liquid + mol_solute)
        if x < 0.:
            liquid_mol[solute_index] = 0.
            solid_mol[solute_index] = mol_solute
        elif x >= x_max:
            liquid_mol[solute_index] = mol_solute
            solid_mol[solute_index] = 0.
        else:
            liquid_mol[solute_index] = mol_solute_liquid = x * F_mol_liquid
            solid_mol[solute_index] = mol_solute - mol_solute_liquid 
    
    def _solve_x(self, T):
        solute_chemical = self.chemicals.tuple[self._solute_index]
        Tm = solute_chemical.Tm
        if Tm is None: raise RuntimeError(f"solute {solute_chemical} does not have a melting temperature, Tm")
        Cpl = solute_chemical.Cn.l(T)
        Cps = solute_chemical.Cn.s(T)
        Hm = solute_chemical.Hfus
        if Tm is None: raise RuntimeError(f"solute {solute_chemical} does not have a heat of fusion, Hfus")
        gamma = 1.
        x = solubility_eutectic(T, Tm, Hm, Cpl, Cps, gamma) # Initial guess
        args = (T, Tm, Hm, Cpl, Cps)
        return flx.wegstein(self._x_iter, x, xtol=1e-6, args=args)
        
    def _x_iter(self, x, T, Tm, Hm, Cpl, Cps):
        self._update_solubility(x)
        liquid_mol = self._liquid_mol[self._index]
        F_mol_liquid = liquid_mol.sum()
        x_l = liquid_mol / F_mol_liquid
        gamma = self._gamma(x_l, T)
        return solubility_eutectic(T, Tm, Hm, Cpl, Cps, gamma[self._solute_gamma_index])
        
        
class SLECache(Cache): load = SLE
del Cache, Equilibrium     