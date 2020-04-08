# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:40:05 2019

@author: yoelr
"""
from ..utils.decorator_utils import thermo_user
from scipy.optimize import differential_evolution
from .._thermal_condition import ThermalCondition
from .vle import VLE
import numpy as np

__all__ = ('LLE',)

def gibbs_free_energy_of_liquid(l, T, f_gamma):
    l_sum = l.sum()
    if l_sum:
        x = l / l_sum
        gamma = f_gamma(x, T)
        xgamma = x * gamma
        xgamma[xgamma <= 0] = 1
        g_mix = (l * np.log(xgamma)).sum()
    else:
        g_mix = 0
    return g_mix

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
            l=[('Water', 301.3), ('Ethanol', 27.72), ('Octane', 0.07885), ('Hexane', 0.01154)]),
        thermal_condition=ThermalCondition(T=360.00, P=101325))
    
    """
    __slots__ = ('_thermo', # [float] Thermo object for estimating mixture properties.
                 '_gamma', # [ActivityCoefficients] Estimates activity coefficients of a liquid.
                 '_imol', # [MaterialIndexer] Stores vapor and liquid molar data.
                 '_TP', # [ThermalCondition] T values are stored here.
                 '_index', # [1d array] Index of chemicals in equilibrium.
                 '_mol') # [Chemical] Single chemical in equilibrium.
    
    differential_evolution_options = {'seed': 0}
    
    def __init__(self, imol, thermal_condition=None, thermo=None):
        self._load_thermo(thermo)
        self._TP = thermal_condition or ThermalCondition(298.15, 101325.)
        self._imol = imol
        self._index = ()
    
    def __call__(self, T, P=None):
        """
        Perform liquid-liquid equilibrium.

        Parameters
        ----------
        T : float
            Operating temperature [K].
        P : float, optional
            Operating pressure [Pa].
            
        """
        self._setup()
        TP = self._TP
        TP.T = T
        if P: TP.P = P
        if self._mol.sum():
            l = self._solve_l(T)
            index = self._index
            imol = self._imol
            imol['l'][index] = l
            imol['L'][index] = self._mol - l
    
    def _setup(self):
        # Get flow rates
        imol = self._imol
        mol = imol['l'] + imol['L']
        
        # Set up indices for both equilibrium and non-equilibrium species
        chemicals = self.chemicals
        index = chemicals.get_lle_indices(mol > 0)
        chemicals = chemicals.tuple
        lle_chemicals = [chemicals[i] for i in index]
        self._gamma = self.thermo.Gamma(lle_chemicals)
        self._index = index
        self._mol = mol[index]
     
    def _gibbs_free_energy(self, l, T):
        f_gamma = self._gamma
        L = self._mol - l
        return (gibbs_free_energy_of_liquid(L, T, f_gamma)
                + gibbs_free_energy_of_liquid(l, T, f_gamma))
    
    def _solve_l(self, T):
        mol = self._mol
        args = (T,)
        bounds = np.zeros([mol.size, 2])
        bounds[:, 1] = mol
        result = differential_evolution(self._gibbs_free_energy, bounds, args,
                                        **self.differential_evolution_options)
        return result.x
    
    imol = VLE.imol
    thermal_condition = VLE.thermal_condition
    __format__ = VLE.__format__
    __repr__ = VLE.__repr__
