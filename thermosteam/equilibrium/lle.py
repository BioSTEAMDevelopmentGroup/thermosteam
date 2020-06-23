# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from flexsolve import njitable
from ..utils import thermo_user, Cache
from scipy.optimize import differential_evolution
from .._thermal_condition import ThermalCondition
from .vle import VLE
import numpy as np

__all__ = ('LLE', 'LLECache')

def liquid_activities(mol_L, T, f_gamma):
    total_mol_L = mol_L.sum()
    if total_mol_L:
        x = mol_L / total_mol_L
        gamma = f_gamma(x, T)
        xgamma = x * gamma
    else:
        xgamma = np.ones_like(mol_L)
    return xgamma

@njitable(cache=True)
def gibbs_free_energy_of_liquid(mol_L, xgamma):
    xgamma[xgamma <= 0] = 1
    g_mix = (mol_L * np.log(xgamma)).sum()
    return g_mix

def lle_objective_function(mol_L, mol, T, f_gamma):
    mol_l = mol - mol_L
    xgamma_l = liquid_activities(mol_l, T, f_gamma)
    xgamma_L = liquid_activities(mol_L, T, f_gamma)
    g_mix_l = gibbs_free_energy_of_liquid(mol_l, xgamma_l)
    g_mix_L = gibbs_free_energy_of_liquid(mol_L, xgamma_L)
    g_mix = g_mix_l + g_mix_L
    return g_mix

def solve_lle_liquid_mol(mol, T, f_gamma, **differential_evolution_options):
    args = (mol, T, f_gamma)
    bounds = np.zeros([mol.size, 2])
    bounds[:, 1] = mol
    result = differential_evolution(lle_objective_function, bounds, args,
                                    **differential_evolution_options)
    return result.x

@thermo_user
class LLE:
    """
    Create a LLE object that performs liquid-liquid equilibrium when called.
    Differential evolution is used to find the solution that globally minimizes
    the gibb's free energy of both phases.
        
    Parameters
    ----------
    imol : MaterialIndexer
        Chemical phase data is stored here.
    thermal_condition=None : ThermalCondition, optional
        The temperature and pressure used in calculations are stored here.
    thermo=None : Thermo, optional
        Themodynamic property package for equilibrium calculations.
        Defaults to `thermosteam.settings.get_thermo()`.
    
    Examples
    --------
    >>> from thermosteam import indexer, equilibrium, settings
    >>> settings.set_thermo(['Water', 'Ethanol', 'Octane', 'Hexane'])
    >>> imol = indexer.MolarFlowIndexer(
    ...             l=[('Water', 304), ('Ethanol', 30)],
    ...             L=[('Octane', 40), ('Hexane', 1)])
    >>> lle = equilibrium.LLE(imol)
    >>> lle(T=360)
    >>> lle
    LLE(imol=MolarFlowIndexer(
            L=[('Water', 2.671), ('Ethanol', 2.284), ('Octane', 39.92), ('Hexane', 0.9885)],
            l=[('Water', 301.3), ('Ethanol', 27.72), ('Octane', 0.07884), ('Hexane', 0.01154)]),
        thermal_condition=ThermalCondition(T=360.00, P=101325))
    
    """
    __slots__ = ('_thermo', # [float] Thermo object for estimating mixture properties.
                 '_imol', # [MaterialIndexer] Stores vapor and liquid molar data.
                 '_thermal_condition', # [ThermalCondition] T and P values are stored here.
)
    differential_evolution_options = {'seed': 0,
                                      'popsize': 12,
                                      'tol': 0.002}
    
    def __init__(self, imol, thermal_condition=None, thermo=None):
        self._load_thermo(thermo)
        self._thermal_condition = thermal_condition or ThermalCondition(298.15, 101325.)
        self._imol = imol
    
    def __call__(self, T, P=None, top_chemical=None):
        """
        Perform liquid-liquid equilibrium.

        Parameters
        ----------
        T : float
            Operating temperature [K].
        P : float, optional
            Operating pressure [Pa].
        top_chemical : str, optional
            Identifier of chemical that will be favored in the "liquid" phase.
            
        """
        thermal_condition = self._thermal_condition
        thermal_condition.T = T
        if P: thermal_condition.P = P
        imol = self._imol
        mol, index, lle_chemicals = self.get_liquid_mol_data()
        total_mol = mol.sum()
        if total_mol:
            gamma = self.thermo.Gamma(lle_chemicals)
            mol_L = solve_lle_liquid_mol(mol, T, gamma,
                                         **self.differential_evolution_options)
            mol_l = mol - mol_L
            if top_chemical:
                MW = self.chemicals.MW[index]
                mass_L = mol_L * MW
                mass_l = mol_l * MW
                top_chemical_index = self.chemicals.index(top_chemical)
                C_L = mass_L[top_chemical_index] / mass_L.sum()
                C_l = mass_l[top_chemical_index] / mass_l.sum()
                top_L = C_L > C_l
                if top_L: mol_l, mol_L = mol_L, mol_l
            imol['l'][index] = mol_l
            imol['L'][index] = mol_L
    
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
    
    imol = VLE.imol
    thermal_condition = VLE.thermal_condition
    __format__ = VLE.__format__
    __repr__ = VLE.__repr__

class LLECache(Cache): load = LLE
del Cache    