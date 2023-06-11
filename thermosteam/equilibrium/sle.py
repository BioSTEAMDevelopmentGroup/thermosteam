# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import flexsolve as flx
from ..utils import Cache
from .equilibrium import Equilibrium
from .activity_coefficients import IdealActivityCoefficients
from chemicals import solubility_eutectic
import numpy as np

__all__ = ('SLE', 'SLECache')

class SLE(Equilibrium, phases='ls'):
    """
    Create an SLE object that performs solid-liquid equilibrium for a given solute
    when called with a temperature and pressure.
        
    Parameters
    ----------
    imol=None : :class:`~thermosteam.indexer.MaterialIndexer`, optional
        Molar chemical phase data is stored here.
    thermal_condition=None : :class:`~thermosteam.ThermalCondition`, optional
        The temperature and pressure used in calculations are stored here.
    thermo=None : :class:`~thermosteam.Thermo`, optional
        Themodynamic property package for equilibrium calculations.
        Defaults to `thermosteam.settings.get_thermo()`.
    activity_coefficient=None : float
        Activity coefficient of solute in the liquid; only valid if
        `thermo.Gamma` is :class:`~thermosteam.equilibrium.activity_coefficients.IdealActivityCoefficients`.
    
    Examples
    --------
    Solve SLE of glucose in water:
    
    >>> from thermosteam import Chemical, indexer, equilibrium, settings
    >>> Glucose = Chemical('Glucose', Tm=419.15, Hfus=19930)
    >>> Glucose.Cn.s.add_model(224.114064, top_priority=True)
    >>> Glucose.Cn.l.add_model(360.312, top_priority=True) # More or less in solution
    >>> settings.set_thermo(['Water', Glucose], cache=True)
    >>> imol = indexer.MolarFlowIndexer(l=[('Water', 10), ('Glucose', 1)], phases=('s', 'l'))
    >>> sle = equilibrium.SLE(imol)
    >>> sle('Glucose', T=298.15) # Given T
    >>> sle
    SLE(imol=MolarFlowIndexer(
            l=[('Water', 10), ('Glucose', 0.01308)],
            s=[('Glucose', 0.9869)]),
        thermal_condition=ThermalCondition(T=298.15, P=101325))
    >>> sle('Glucose', H=0.) # Given H
    >>> sle    
    SLE(imol=MolarFlowIndexer(
            l=[('Water', 10), ('Glucose', 0.01306)],
            s=[('Glucose', 0.9869)]),
        thermal_condition=ThermalCondition(T=298.06, P=101325))
    
    Results may not be too accurate sometimes, but the solubility 
    (mol fraction of solute in solvent) may be specified:
    
    >>> sle('Glucose', T=298.15, solubility=0.0833) # Given T
    >>> sle
    SLE(imol=MolarFlowIndexer(
            l=[('Water', 10), ('Glucose', 0.9087)],
            s=[('Glucose', 0.09131)]),
        thermal_condition=ThermalCondition(T=298.15, P=101325))
    >>> sle('Glucose', H=90, solubility=0.0833) # Given H
    >>> sle
    SLE(imol=MolarFlowIndexer(
            l=[('Water', 10), ('Glucose', 0.9087)],
            s=[('Glucose', 0.09131)]),
        thermal_condition=ThermalCondition(T=292.41, P=101325))
    
    Solve SLE of tetradecanol in octanol:
        
    >>> from thermosteam import indexer, equilibrium, settings
    >>> settings.set_thermo(['Methanol', 'Tetradecanol'], cache=True)
    >>> imol = indexer.MolarFlowIndexer(l=[('Methanol', 10), ('Tetradecanol', 30)], phases=('s', 'l'))
    >>> sle = equilibrium.SLE(imol)
    >>> sle('Tetradecanol', T=300) # Given T
    >>> sle
    SLE(imol=MolarFlowIndexer(
            l=[('Methanol', 10), ('Tetradecanol', 19.07)],
            s=[('Tetradecanol', 10.93)]),
        thermal_condition=ThermalCondition(T=300.00, P=101325))
    >>> sle('Tetradecanol', H=0.) # Given H
    >>> sle
    SLE(imol=MolarFlowIndexer(
            l=[('Methanol', 10), ('Tetradecanol', 6.116)],
            s=[('Tetradecanol', 23.88)]),
        thermal_condition=ThermalCondition(T=287.31, P=101325))
    
    Solve SLE of pure tetradecanol:
    
    >>> from thermosteam import indexer, equilibrium, settings
    >>> settings.set_thermo(['Octanol', 'Tetradecanol'], cache=True)
    >>> imol = indexer.MolarFlowIndexer(l=[('Tetradecanol', 30)], phases=('s', 'l'))
    >>> sle = equilibrium.SLE(imol)
    >>> sle('Tetradecanol', T=300) # Under melting point given T
    >>> sle
    SLE(imol=MolarFlowIndexer(phases=('l', 's'),
            s=[('Tetradecanol', 30)]),
        thermal_condition=ThermalCondition(T=300.00, P=101325))
    >>> sle('Tetradecanol', T=320) # Over melting point given T
    >>> sle
    SLE(imol=MolarFlowIndexer(phases=('l', 's'),
            l=[('Tetradecanol', 30)]),
        thermal_condition=ThermalCondition(T=320.00, P=101325))
    >>> sle('Tetradecanol', H=0.) # Under melting point given H
    >>> sle
    SLE(imol=MolarFlowIndexer(phases=('l', 's'),
            s=[('Tetradecanol', 30)]),
        thermal_condition=ThermalCondition(T=298.15, P=101325))
    >>> sle('Tetradecanol', H=1000000) # Over melting point given H
    >>> sle
    SLE(imol=MolarFlowIndexer(phases=('l', 's'),
            l=[('Tetradecanol', 30)]),
        thermal_condition=ThermalCondition(T=317.59, P=101325))
    >>> sle('Tetradecanol', H=500000) # At melting point given H
    >>> sle
    SLE(imol=MolarFlowIndexer(
            l=[('Tetradecanol', 13.2)],
            s=[('Tetradecanol', 16.8)]),
        thermal_condition=ThermalCondition(T=312.65, P=101325))
    
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
                 '_phase_data',
                 'activity_coefficient', # [float] Activity coefficient of solute in the liquid; only valid thermo.Gamma is IdealActivityCoefficients.
    )
    
    def __init__(self, imol=None, thermal_condition=None, thermo=None, 
                 solubility_weight=None, activity_coefficient=None):
        super().__init__(imol, thermal_condition, thermo)
        self._phase_data = tuple(imol)
        self._nonzero = None
        self._index = ()
        self._chemical = None
        self._x = None
        self.activity_coefficient = activity_coefficient
    
    def _setup(self):
        # Get flow rates
        imol = self._imol
        self._phase_data = tuple(imol)
        self._liquid_mol = liquid_mol = imol['l']
        self._solid_mol = solid_mol = imol['s']
        mol = liquid_mol + solid_mol
        solute_index = self._solute_index
        self._mol_solute = mol_solute = mol[solute_index]
        if not mol_solute:
            raise RuntimeError('no solute available')
        nonzero = frozenset(mol.nonzero_keys())
        if self._nonzero == nonzero:
            index = self._index
        else:
            chemicals = self.chemicals
            # Set up indices for both equilibrium and non-equilibrium species
            index = chemicals.get_lle_indices(nonzero)   
            N = len(index)
            if N == 1:
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
        
    def __call__(self, solute, T=None, P=None, H=None, solubility=None):
        """
        Perform solid-liquid equilibrium.

        Parameters
        ----------
        T : float, optional
            Operating temperature [K].
        P : float, optional
            Operating pressure [Pa].
        H : float, optional
            Operating enthalpy [kJ]
        solubility : float, optional
            Mol fraction of solute at maximum solubility.
            
        """
        thermal_condition = self.thermal_condition
        chemicals = self.chemicals
        mixture = self.mixture
        phase_data = self._phase_data
        self._solute_index = solute_index = chemicals.get_index(solute)
        T_given = T is not None
        H_given = H is not None
        if not (T_given or H_given):
            raise ValueError('must pass T or H; none passed')
        if T_given and H_given:
            raise ValueError('must pass either T or H; not both')
        if P: thermal_condition.P = P
        else: P = thermal_condition.P 
        if T_given: thermal_condition.T = T
        else: T = thermal_condition.T
        if solubility is not None:
            solute_index = self._solute_index
            self._mol_solute = (
                self._solid_mol[solute_index] + self._liquid_mol[solute_index]
            )
            self._index = slice(None)
            self._update_solubility(solubility)
            if T_given:
                thermal_condition.T = T
            elif H_given:
                thermal_condition.T = mixture.xsolve_T_at_HP(phase_data, H, T, P)
            else:
                raise Exception('Unknown')
            return
        self._setup()
        mol_solute = self._mol_solute
        liquid_mol = self._liquid_mol
        solid_mol = self._solid_mol
            
        if T_given: 
            if self._chemical:
                thermal_condition.T = T
                Tm = self._chemical.Tm
                if T > Tm:
                    liquid_mol[solute_index] = mol_solute
                    solid_mol[solute_index] = 0.
                else:
                    liquid_mol[solute_index] = 0.
                    solid_mol[solute_index] = mol_solute
            else:
                solubility = self._solve_x(T)
                self._update_solubility(solubility)
        elif H_given:
            if self._chemical:
                # Set temperature in equilibrium
                self._thermal_condition.T = T = Tm = self._chemical.Tm
                
                # Check if liquid
                liquid_mol[solute_index] = mol_solute
                solid_mol[solute_index] = 0.
                H_liq = mixture.xH(phase_data, T, P)
                if H >= H_liq:
                    self._thermal_condition.T = mixture.xsolve_T_at_HP(phase_data, H, T, P)
                    return
    
                # Check if subcooled liquid
                liquid_mol[solute_index] = 0.
                solid_mol[solute_index] = mol_solute
                H_sol = mixture.xH(phase_data, T, P)
                if H <= H_sol:
                    self._thermal_condition.T = mixture.xsolve_T_at_HP(phase_data, H, T, P)
                    return
                
                # Adjust liquid fraction accordingly
                L = (H - H_sol)/(H_liq - H_sol)
                liquid_mol[solute_index] = L * mol_solute
                solid_mol[solute_index] = mol_solute - liquid_mol[solute_index]
            elif H_given:
                def f(T):
                    solubility = self._solve_x(T)
                    self._update_solubility(solubility)
                    return mixture.xsolve_T_at_HP(phase_data, H, T, P)
                
                self._thermal_condition.T = T = flx.aitken(
                    f, mixture.xsolve_T_at_HP(phase_data, H, T, P),
                    1e-3, (), 50, checkiter=False
                )
            else:
                raise Exception('unknown')
    
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
            liquid_mol[solute_index] = mol_solute_liquid = F_mol_liquid * x / (1 - x)
            solid_mol[solute_index] = mol_solute - mol_solute_liquid 
    
    def _solve_x(self, T):
        solute_chemical = self.chemicals.tuple[self._solute_index]
        Tm = solute_chemical.Tm
        if Tm is None: raise RuntimeError(f"solute {solute_chemical} does not have a melting temperature, Tm")
        Cpl = solute_chemical.Cn.l(T)
        Cps = solute_chemical.Cn.s(T)
        Hm = solute_chemical.Hfus
        if Hm is None: raise RuntimeError(f"solute {solute_chemical} does not have a heat of fusion, Hfus")
        gamma = 1.
        x = solubility_eutectic(T, Tm, Hm, Cpl, Cps, gamma) # Initial guess
        
        args = (T, Tm, Hm, Cpl, Cps)
        if isinstance(self._gamma, IdealActivityCoefficients):
            return solubility_eutectic(T, Tm, Hm, Cpl, Cps, self.activity_coefficient or 1.)
        return flx.aitken(self._x_iter, x, xtol=1e-6, args=args, checkiter=False, maxiter=100)
        
    def _x_iter(self, x, T, Tm, Hm, Cpl, Cps):
        self._update_solubility(x)
        liquid_mol = self._liquid_mol[self._index]
        F_mol_liquid = liquid_mol.sum()
        x_l = liquid_mol / F_mol_liquid
        gamma = self._gamma(x_l, T)
        return solubility_eutectic(T, Tm, Hm, Cpl, Cps, gamma[self._solute_gamma_index])
        
        
class SLECache(Cache): load = SLE
del Cache, Equilibrium     
