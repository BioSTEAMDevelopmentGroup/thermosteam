# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
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
from scipy.optimize import shgo, differential_evolution
import numpy as np

__all__ = ('LLE', 'LLECache')

# TODO: SUBMIT ISSUE TO NUMBA
@njit(cache=True)
def liquid_activities(mol_L, T, f_gamma, gamma_args):
    total_mol_L = mol_L.sum()
    x = mol_L / total_mol_L
    gamma = f_gamma(x, T, *gamma_args)
    xgamma = x * gamma
    return xgamma

@njit(cache=True)
def gibbs_free_energy_of_liquid(mol_L, xgamma):
    xgamma[xgamma <= 0] = 1
    g_mix = (mol_L * np.log(xgamma)).sum()
    return g_mix

@njit(cache=True)
def lle_objective_function(mol_L, mol, T, f_gamma, gamma_args):
    mol_l = mol - mol_L
    if mol_l.sum() == 0.:
        g_mix_l = 0.
    else:
        xgamma_l = liquid_activities(mol_l, T, f_gamma, gamma_args)
        g_mix_l = gibbs_free_energy_of_liquid(mol_l, xgamma_l)
    if mol_L.sum() == 0.:
        g_mix_L = 0.
    else:
        xgamma_L = liquid_activities(mol_L, T, f_gamma, gamma_args)
        g_mix_L = gibbs_free_energy_of_liquid(mol_L, xgamma_L)
    g_mix = g_mix_l + g_mix_L
    return g_mix

class LLE(Equilibrium, phases='lL'):
    """
    Create a LLE object that performs liquid-liquid equilibrium when called.
    The SHGO (simplicial homology global optimization) alogorithm [1]_ is used to find the 
    solution that globally minimizes the gibb's free energy of both phases.
        
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
    >>> lle(T=360, top_chemical='Octane')
    >>> lle
    LLE(imol=MolarFlowIndexer(
            L=[('Water', 2.67), ('Ethanol', 2.28), ('Octane', 39.9), ('Hexane', 0.988)],
            l=[('Water', 301.), ('Ethanol', 27.7), ('Octane', 0.0788), ('Hexane', 0.0115)]),
        thermal_condition=ThermalCondition(T=360.00, P=101325))
    
    References
    ----------
    .. [1] Endres, SC, Sandrock, C, Focke, WW (2018) “A simplicial homology 
           algorithm for lipschitz optimisation”, Journal of Global Optimization.
    
    """
    __slots__ = ('composition_cache_tolerance',
                 'temperature_cache_tolerance',
                 'method',
                 '_z_mol',
                 '_T',
                 '_lle_chemicals',
                 '_IDs',
                 '_K',
                 '_phi'
    )
    default_method = 'shgo'
    shgo_options = dict(f_tol=1e-6, minimizer_kwargs=dict(f_tol=1e-6))
    differential_evolution_options = {'seed': 0,
                                      'popsize': 12,
                                      'tol': 1e-6}
    
    def __init__(self, imol=None, thermal_condition=None, thermo=None,
                 composition_cache_tolerance=1e-5,
                 temperature_cache_tolerance=1e-3):
        super().__init__(imol, thermal_condition, thermo)
        self.composition_cache_tolerance = composition_cache_tolerance
        self.temperature_cache_tolerance = temperature_cache_tolerance
        self.method = self.default_method
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
        if F_mol and len(lle_chemicals) > 1:
            z_mol = mol = mol / F_mol # Normalize first
            if (self._lle_chemicals == lle_chemicals 
                and T - self._T < self.temperature_cache_tolerance 
                and (self._z_mol - z_mol < self.composition_cache_tolerance).all()):
                K = self._K 
                self._phi = phi = phase_fraction(z_mol, K, self._phi)
                if phi >= 1.:
                    mol_l = mol
                    mol_L = 0.
                else:
                    y = z_mol * K / (phi * K + (1 - phi))
                    mol_l = y * phi
                    mol_L = mol - mol_l
            else:
                mol_L = self.solve_lle_liquid_mol(mol, T, lle_chemicals)
                mol_l = mol - mol_L
                if top_chemical:
                    MW = self.chemicals.MW[index]
                    mass_L = mol_L * MW
                    mass_l = mol_l * MW
                    IDs = {i.ID: n for n, i in enumerate(lle_chemicals)}
                    try: top_chemical_index = IDs[top_chemical]
                    except: pass
                    else:
                        C_L = mass_L[top_chemical_index] / mass_L.sum()
                        C_l = mass_l[top_chemical_index] / mass_l.sum()
                        if C_L < C_l: mol_l, mol_L = mol_L, mol_l
                F_mol_l = mol_l.sum()
                F_mol_L = mol_L.sum()
                if not F_mol_L:
                    self._K = np.zeros_like(mol)
                    self._phi = 0.
                elif not F_mol_l:
                    self._K = 1e16 * np.ones_like(mol)
                    self._phi = 1.
                else:
                    x_mol_l = mol_l / F_mol_l
                    x_mol_L = mol_L / F_mol_L
                    x_mol_l[x_mol_l < 1e-16] = 1e-16
                    K = x_mol_L / x_mol_l
                    self._K = K
                    self._phi = F_mol_L / (F_mol_L + F_mol_l)
                self._lle_chemicals = lle_chemicals
                self._z_mol = z_mol
                self._T = T
            if not update: return self._lle_chemicals, self._K, self._phi
            imol['l'][index] = mol_l * F_mol
            imol['L'][index] = mol_L * F_mol
        elif not update: 
            mol_l = mol
            mol_L = np.zeros_like(mol_l)
            if top_chemical:
                MW = self.chemicals.MW[index]
                IDs = {i.ID: n for n, i in enumerate(lle_chemicals)}
                try: top_chemical_index = IDs[top_chemical]
                except: pass
                else:
                    C_L = mol_L[top_chemical_index]
                    C_l = mol_l[top_chemical_index]
                    if C_L < C_l: mol_l, mol_L = mol_L, mol_l
            F_mol_L = mol_L.sum()
            if F_mol_L:
                K = 1e16 * np.ones_like(mol)
                phi = 1.
            else:
                K = np.zeros_like(mol)
                phi = 0.
            return lle_chemicals, K, phi
        
    def solve_lle_liquid_mol(self, mol, T, lle_chemicals):
        gamma = self.thermo.Gamma(lle_chemicals)
        args = (mol, T, gamma.f, gamma.args)
        bounds = np.zeros([mol.size, 2])
        bounds[:, 1] = mol
        bounds[0, 1] = 0.5 * mol[0] # Remove symmetry
        method = self.method
        if method == 'shgo':
            result = shgo(lle_objective_function, bounds, args, options=self.shgo_options)
            if not result.success or (result.x == 0.).all():
                result = differential_evolution(lle_objective_function, bounds, args,
                                                **self.differential_evolution_options)
        elif method == 'differential evolution':
            result = differential_evolution(lle_objective_function, bounds, args,
                                            **self.differential_evolution_options)
        else:
            raise ValueError(f"invalid method {repr(method)}")
        return result.x
        
    def get_liquid_mol_data(self):
        # Get flow rates
        imol = self._imol
        imol['L'] = mol =  imol['l'] + imol['L']
        imol['l'] = 0
        index = self.chemicals.get_lle_indices(mol.nonzero_keys())
        mol = mol[index]
        chemicals = self.chemicals.tuple
        lle_chemicals = [chemicals[i] for i in index]
        return mol, index, lle_chemicals

class LLECache(Cache): load = LLE
del Cache, njit, Equilibrium