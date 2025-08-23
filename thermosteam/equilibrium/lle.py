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
from ..exceptions import NoEquilibrium
from .binary_phase_fraction import phase_fraction
from scipy.optimize import shgo, differential_evolution
import flexsolve as flx
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

@njit(cache=True)
def psuedo_equilibrium_inner_loop(logKgammay, z, T, n, f_gamma, gamma_args, phi):
    logKgammay_new = logKgammay.copy()
    K = np.exp(logKgammay[:n])
    x = z/(1. + phi * (K - 1.))
    x[x < 0] = 1e-16
    x = x / x.sum()
    gammay = logKgammay[n:]
    gammax = f_gamma(x, T, *gamma_args)
    K = gammax / gammay 
    y = K * x
    y /= y.sum()
    gammay = f_gamma(y, T, *gamma_args)
    K = gammax / gammay
    logKgammay_new[:n] = np.log(K)
    logKgammay_new[n:] = gammay
    return logKgammay_new

def pseudo_equilibrium_outer_loop(logKgammayphi, z, T, n, f_gamma, gamma_args, inner_loop_options):
    logKgammayphi_new = logKgammayphi.copy()
    logKgammay = logKgammayphi[:-1]
    phi = logKgammayphi[-1]
    args=(z, T, n, f_gamma, gamma_args)
    logKgammay = flx.aitken(
        psuedo_equilibrium_inner_loop, logKgammay, 
        args=(*args, phi), **inner_loop_options,
    )
    K = np.exp(logKgammay[:n])
    try:
        phi = phase_fraction(z, K, phi)
    except (ZeroDivisionError, FloatingPointError):
        raise NoEquilibrium
    if np.isnan(phi): raise NoEquilibrium
    if phi > 1: phi = 1 - 1e-16
    if phi < 0: phi = 1e-16
    logKgammayphi_new[:-1] = logKgammay
    logKgammayphi_new[-1] = phi
    return logKgammayphi_new

def pseudo_equilibrium(K, phi, z, T, n, f_gamma, gamma_args, inner_loop_options, outer_loop_options):
    phi = phase_fraction(z, K, phi)
    try:
        x = z/(1. + phi * (K - 1.))
    except:
        x = np.ones(n)
    x /= x.sum()
    y = K * x
    logKgammayphi = np.zeros(2*n + 1)
    logKgammayphi[:n] = np.log(K)
    logKgammayphi[n:-1] = f_gamma(y, T, *gamma_args)
    logKgammayphi[-1] = phi
    try:
        logKgammayphi = flx.aitken(
            pseudo_equilibrium_outer_loop, logKgammayphi,
            args=(z, T, n, f_gamma, gamma_args, inner_loop_options),
            **outer_loop_options,
        )
    except NoEquilibrium:
        return z
    phi = logKgammayphi[-1]
    if 0 < phi < 1:
        K = np.exp(logKgammayphi[:n])
        return z/(1. + phi * (K - 1.)) * (1 - phi)
    else:
        return z

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
            L=[('Water', 2.671), ('Ethanol', 2.284), ('Octane', 39.92), ('Hexane', 0.9885)],
            l=[('Water', 301.3), ('Ethanol', 27.72), ('Octane', 0.07884), ('Hexane', 0.01154)]),
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
                 '_gamma_y',
                 '_phi'
    )
    default_method = 'pseudo equilibrium'
    shgo_options = dict(f_tol=1e-6, minimizer_kwargs=dict(f_tol=1e-6))
    differential_evolution_options = {'seed': 0,
                                      'popsize': 12,
                                      'tol': 1e-6}
    pseudo_equilibrium_outer_loop_options = dict(
        xtol=1e-12, rtol=1e-12, maxiter=100, checkiter=False, 
        checkconvergence=False, convergenceiter=10,
    )
    pseudo_equilibrium_inner_loop_options = dict(
        xtol=1e-9, rtol=1e-9, maxiter=20, checkiter=False,
        checkconvergence=False, convergenceiter=5,
    )
    default_composition_cache_tolerance = 1e-5
    default_temperature_cache_tolerance = 1e-3
    
    def __init__(self, imol=None, thermal_condition=None, thermo=None,
                 composition_cache_tolerance=None,
                 temperature_cache_tolerance=None,
                 method=None):
        super().__init__(imol, thermal_condition, thermo)
        self.composition_cache_tolerance = (
            self.default_composition_cache_tolerance 
            if composition_cache_tolerance  is None else 
            composition_cache_tolerance 
        )
        self.temperature_cache_tolerance = (
            self.default_temperature_cache_tolerance 
            if temperature_cache_tolerance is None else 
            temperature_cache_tolerance
        )
        self.method = self.default_method if method is None else method
        self._lle_chemicals = None
        self._K = None
        self._phi = None
    
    def __call__(self, T, P=None, top_chemical=None, update=True, use_cache=False, single_loop=False):
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
        imol = self._imol
        if update:
            thermal_condition = self._thermal_condition
            thermal_condition.T = T
            if P: thermal_condition.P = P
        else:
            old_data = imol.data.copy()
        mol, index, lle_chemicals = self.get_liquid_mol_data()
        if top_chemical is None:
            top_chemical = self.chemicals.IDs[mol.argmax()]
        F_mol = mol.sum()
        if F_mol and len(lle_chemicals) > 1:
            z_mol = mol = mol / F_mol # Normalize first
            use_cache = use_cache and self._lle_chemicals == lle_chemicals
            if use_cache:
                use_cache = (
                    T - self._T < self.temperature_cache_tolerance 
                    and (self._z_mol - z_mol < self.composition_cache_tolerance).all()
                )
            if use_cache:
                K = self._K 
                self._phi = phi = phase_fraction(z_mol, K, self._phi)
                if phi >= 1.:
                    mol_l = mol
                    mol_L = 0. * mol
                else:
                    y = z_mol * K / (phi * K + (1 - phi))
                    mol_l = y * phi
                    mol_L = mol - mol_l
            else:
                if self._lle_chemicals != lle_chemicals: 
                    self._K = None
                    self._gamma_y = None
                    self._phi = None
                mol_L = self.solve_lle_liquid_mol(mol, T, lle_chemicals, single_loop)
                mol_l = mol - mol_L
                if (mol_L < 0).any(): 
                    self.solve_lle_liquid_mol(mol, T, lle_chemicals, single_loop)
                if (mol_l < 0).any(): 
                    self.solve_lle_liquid_mol(mol, T, lle_chemicals, single_loop)
            MW = self.chemicals.MW[index]
            mass_L = mol_L * MW
            mass_l = mol_l * MW
            IDs = {i.ID: n for n, i in enumerate(lle_chemicals)}
            top_chemical_index = IDs[top_chemical]
            ML = mass_L.sum()
            Ml = mass_l.sum()
            if ML and Ml:
                C_L = mass_L[top_chemical_index] / ML
                C_l = mass_l[top_chemical_index] / Ml
                if C_L < C_l: mol_l, mol_L = mol_L, mol_l
            elif Ml:
                mol_l, mol_L = mol_L, mol_l
            F_mol_l = mol_l.sum()
            F_mol_L = mol_L.sum()
            if not F_mol_L:
                mol_l, mol_L = mol_L, mol_l
                self._K = 1e16 * np.ones_like(mol)
                self._gamma_y = np.ones_like(mol)
                self._phi = 1.
            elif not F_mol_l:
                self._K = 1e16 * np.ones_like(mol)
                self._gamma_y = np.ones_like(mol)
                self._phi = 1.
            else:
                x_mol_l = mol_l / F_mol_l
                x_mol_L = mol_L / F_mol_L
                x_mol_l[x_mol_l < 1e-16] = 1e-16
                K = x_mol_L / x_mol_l
                gamma = self.thermo.Gamma(lle_chemicals)
                self._K = K
                self._gamma_y = gamma(x_mol_L, T)
                self._phi = F_mol_L / (F_mol_L + F_mol_l)
            self._lle_chemicals = lle_chemicals
            self._z_mol = z_mol
            self._T = T
            if not update: 
                imol.data[:] = old_data
                return self._lle_chemicals, self._K, self._gamma_y, self._phi
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
            gamma_y = np.ones_like(mol)
            imol.data[:] = old_data
            return lle_chemicals, K, gamma_y, phi
        
    def solve_lle_liquid_mol(self, mol, T, lle_chemicals, single_loop):
        gamma = self.thermo.Gamma(lle_chemicals)
        indices = np.argsort(mol * np.array([i.MW for i in lle_chemicals]))
        method = self.method
        n = mol.size
        if method == 'pseudo equilibrium':
            if self._K is not None and 0 < self._phi < 1:
                K = self._K
                phi = self._phi
            else:
                x = mol.copy()
                y = mol.copy()
                a = indices[-1]
                b = indices[-2]
                x[a] = 0.99
                y[a] = 1e-3
                x[b] = 1e-3
                y[b] = 0.99
                x /= x.sum()
                y /= y.sum()
                K = gamma(y, T) / gamma(x, T)
                phi = 0.5
            if single_loop:
                z = mol
                f_gamma = gamma.f
                gamma_args = gamma.args
                phi = phase_fraction(z, K, phi)
                try:
                    x = z/(1. + phi * (K - 1.))
                except:
                    x = np.ones(n)
                x /= x.sum()
                y = K * x
                Kgammayphi = np.zeros(2*n + 1)
                Kgammayphi[:n] = K
                Kgammayphi[n:-1] = f_gamma(y, T, *gamma_args)
                Kgammayphi[-1] = phi
                Kgammay = Kgammayphi[:-1]
                phi = Kgammayphi[-1]
                args=(z, T, n, f_gamma, gamma_args)
                Kgammay = flx.fixed_point(
                    psuedo_equilibrium_inner_loop, Kgammay, 
                    args=(*args, phi), **self.pseudo_equilibrium_inner_loop_options,
                )
                K = Kgammay[:n]
                try:
                    phi = phase_fraction(z, K, phi)
                except (ZeroDivisionError, FloatingPointError):
                    return z
                if np.isnan(phi): return z
                if phi > 1: phi = 1 - 1e-16
                if phi < 0: phi = 1e-16
                return z/(1. + phi * (K - 1.)) * (1 - phi)
            else:
                K[K <= 0] = 1e-9
                return pseudo_equilibrium(
                    K, phi, mol, T, n, gamma.f, gamma.args, 
                    self.pseudo_equilibrium_inner_loop_options,
                    self.pseudo_equilibrium_outer_loop_options,
                )
        index = indices[-1]
        args = (mol, T, gamma.f, gamma.args)
        bounds = np.zeros([n, 2])
        bounds[:, 1] = mol
        bounds[index, 1] = 0.5 * mol[index] # Remove symmetry
        if method == 'shgo':
            result = shgo(
                lle_objective_function, bounds, args,
                options=self.shgo_options
            )
            if not result.success or (result.x == 0.).all():
                result = differential_evolution(
                    lle_objective_function, bounds, args,
                    **self.differential_evolution_options
                )
            return result.x
        elif method == 'differential evolution':
            result = differential_evolution(
                lle_objective_function, bounds, args,
                **self.differential_evolution_options
            )
            return result.x
        else:
            raise ValueError(f"invalid method {repr(method)}")
        
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
