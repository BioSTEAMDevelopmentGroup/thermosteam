# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from numba import njit
from ..utils import Cache
from .equilibrium import Equilibrium
from .binary_phase_fraction import phase_fraction
import numpy as np

__all__ = ('LLE', 'LLECache')

@njit(cache=True)
def liquid_activities(mol_L, T, f_gamma, gamma_args):
    total_mol_L = mol_L.sum()
    if total_mol_L:
        x = mol_L / total_mol_L
        gamma = f_gamma(x, T, *gamma_args)
        xgamma = x * gamma
    else:
        xgamma = np.ones_like(mol_L)
    return xgamma

@njit(cache=True)
def gibbs_free_energy_of_liquid(mol_L, xgamma):
    xgamma[xgamma <= 0] = 1
    g_mix = (mol_L * np.log(xgamma)).sum()
    return g_mix

@njit(cache=True)
def lle_objective_function(mol_L, mol, T, f_gamma, gamma_args):
    mol_l = mol - mol_L
    xgamma_l = liquid_activities(mol_l, T, f_gamma, gamma_args)
    xgamma_L = liquid_activities(mol_L, T, f_gamma, gamma_args)
    g_mix_l = gibbs_free_energy_of_liquid(mol_l, xgamma_l)
    g_mix_L = gibbs_free_energy_of_liquid(mol_L, xgamma_L)
    g_mix = g_mix_l + g_mix_L
    return g_mix

def solve_lle_liquid_mol(mol, T, f_gamma, gamma_args, **differential_evolution_options):
    from scipy.optimize import differential_evolution
    args = (mol, T, f_gamma, gamma_args)
    bounds = np.zeros([mol.size, 2])
    bounds[:, 1] = mol
    result = differential_evolution(lle_objective_function, bounds, args,
                                    **differential_evolution_options)
    return result.x

class LLE(Equilibrium, phases='lL'):
    """
    Create a LLE object that performs liquid-liquid equilibrium when called.
    Differential evolution is used to find the solution that globally minimizes
    the gibb's free energy of both phases.
        
    Parameters
    ----------
    imol=None : :class:`~thermosteam.indexer.MaterialIndexer`, optional
        Molar chemical phase data is stored here.
    thermal_condition=None : :class:`~thermosteam.ThermalCondition`, optional
        The temperature and pressure used in calculations are stored here.
    thermo=None : :class:`~thermosteam.Thermo`, optional
        Themodynamic property package for equilibrium calculations.
        Defaults to `thermosteam.settings.get_thermo()`.
    
    Examples
    --------
    >>> from thermosteam import indexer, equilibrium, settings
    >>> settings.set_thermo(['Water', 'Ethanol', 'Octane', 'Hexane'], cache=True)
    >>> imol = indexer.MolarFlowIndexer(
    ...             l=[('Water', 304), ('Ethanol', 30)],
    ...             L=[('Octane', 40), ('Hexane', 1)]
    ... )
    >>> lle = equilibrium.LLE(imol)
    >>> lle(T=360)
    >>> lle
    LLE(imol=MolarFlowIndexer(
            L=[('Water', 2.67), ('Ethanol', 2.28), ('Octane', 39.9), ('Hexane', 0.988)],
            l=[('Water', 301.), ('Ethanol', 27.7), ('Octane', 0.0788), ('Hexane', 0.0115)]),
        thermal_condition=ThermalCondition(T=360.00, P=101325))
    
    """
    __slots__ = ('composition_cache_tolerance',
                 'temperature_cache_tolerance',
                 '_z_mol',
                 '_T',
                 '_lle_chemicals',
                 '_IDs',
                 '_K',
                 '_phi'
    )
    differential_evolution_options = {'seed': 0,
                                      'popsize': 12,
                                      'tol': 1e-6}
    
    def __init__(self, imol=None, thermal_condition=None, thermo=None,
                 composition_cache_tolerance=1e-5,
                 temperature_cache_tolerance=1e-3):
        super().__init__(imol, thermal_condition, thermo)
        self.composition_cache_tolerance = composition_cache_tolerance
        self.temperature_cache_tolerance = temperature_cache_tolerance
        self._lle_chemicals = None
    
    def __call__(self, T, P=None, top_chemical=None, update=True):
        """
        Perform liquid-liquid equilibrium.

        Parameters
        ----------
        T : float
            Operating temperature [K].
        P : float, optional
            Operating pressure [Pa].
        top_chemical : str, optional
            Identifier of chemical that will be favored in the "LIQUID" phase.
        update : bool, optional
            Whether to update material flows, temperature and pressure. If False,
            returns the chemicals in liquid-liquid equilibrium, 
            partition coefficients, and phase fraction.
            
        """
        if update:
            thermal_condition = self._thermal_condition
            thermal_condition.T = T
            if P: thermal_condition.P = P
        imol = self._imol
        mol, index, lle_chemicals = self.get_liquid_mol_data()
        F_mol = mol.sum()
        mol = mol / F_mol # Normalize first
        if F_mol:
            z_mol = mol / F_mol
            if (self._lle_chemicals == lle_chemicals 
                and T - self._T < self.temperature_cache_tolerance 
                and (self._z_mol - z_mol < self.composition_cache_tolerance).all()):
                K = self._K 
                self._phi = phi = phase_fraction(z_mol, K, self._phi)
                y = z_mol * K / (phi * K + (1 - phi))
                mol_l = y * phi
                mol_L = mol - mol_l
            else:
                gamma = self.thermo.Gamma(lle_chemicals)
                mol_L = solve_lle_liquid_mol(mol, T, gamma.f, gamma.args,
                                             **self.differential_evolution_options)
                mol_l = mol - mol_L
                if top_chemical:
                    MW = self.chemicals.MW[index]
                    mass_L = mol_L * MW
                    mass_l = mol_l * MW
                    IDs = {i.ID: n for n, i in enumerate(lle_chemicals)}
                    top_chemical_index = IDs[top_chemical]
                    C_L = mass_L[top_chemical_index] / mass_L.sum()
                    C_l = mass_l[top_chemical_index] / mass_l.sum()
                    if C_L < C_l: mol_l, mol_L = mol_L, mol_l
                F_mol_l = mol_l.sum()
                z_mol_l = mol_l / F_mol_l
                F_mol_L = mol_L.sum()
                z_mol_L = mol_L / F_mol_L
                z_mol_l[z_mol_l < 1e-16] = 1e-16
                K = z_mol_L / z_mol_l
                self._K = K
                self._phi = F_mol_L / (F_mol_L + F_mol_l)
                self._lle_chemicals = lle_chemicals
                self._z_mol = z_mol
                self._T = T
            if not update: return self._lle_chemicals, self._K, self._phi
            imol['l'][index] = mol_l * F_mol
            imol['L'][index] = mol_L * F_mol
        
    def get_liquid_mol_data(self):
        # Get flow rates
        imol = self._imol
        imol['L'] = mol =  imol['l'] + imol['L']
        imol['l'] = 0
        index = self.chemicals.get_lle_indices(mol > 0)
        mol = mol[index]
        chemicals = self.chemicals.tuple
        lle_chemicals = [chemicals[i] for i in index]
        return mol, index, lle_chemicals

class LLECache(Cache): load = LLE
del Cache, njit, Equilibrium