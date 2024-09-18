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
import thermosteam as tmo
import numpy as np
from math import exp
from thermosteam import functional as fn
from thermo.interaction_parameters import IPDB
from thermo import eos_mix
from .. import units_of_measure as thermo_units
from ..base import PhaseHandle, MockPhaseTHandle, MockPhaseTPHandle, SparseVector, sparse
from .ideal_mixture_model import (
    SinglePhaseIdealTMixtureModel,
    IdealTMixtureModel, 
    IdealTPMixtureModel, 
    IdealEntropyModel, 
    IdealHvapModel
)
from .._chemicals import Chemical, CompiledChemicals, chemical_data_array

__all__ = ('Mixture', 'IdealMixture')

# %% Functions for building mixture models

def create_mixture_model(chemicals, var, Model):
    getfield = getattr
    isa = isinstance
    handles = []
    for chemical in chemicals:
        obj = getfield(chemical, var)
        if isa(obj, PhaseHandle):
            phase_handle = obj
        elif var == 'Cn':
            phase_handle = MockPhaseTHandle(var, obj)
        else:
            phase_handle = MockPhaseTPHandle(var, obj)
        handles.append(phase_handle)
    return Model(handles, var)
    

# %% Energy balance

def iter_T_at_HP(T, H, H_model, phase, mol, P, Cn_model, Cn_cache):
    # Used to solve for temperature at given ethalpy 
    counter, Cn = Cn_cache
    if not counter % 5: Cn_cache[1] = Cn = Cn_model(phase, mol, T, P)
    Cn_cache[0] += 1
    return T + (H - H_model(phase, mol, T, P)) / Cn

def xiter_T_at_HP(T, H, H_model, phase_mol, P, Cn_model, Cn_cache):
    # Used to solve for temperature at given ethalpy 
    counter, Cn = Cn_cache
    if not counter % 5: Cn_cache[1] = Cn = Cn_model(phase_mol, T, P)
    Cn_cache[0] += 1
    return T + (H - H_model(phase_mol, T, P)) / Cn

def iter_T_at_SP(T, S, S_model, phase, mol, P, Cn_model, Cn_cache):
    # Used to solve for temperature at given entropy 
    counter, Cn = Cn_cache
    if not counter % 5: Cn_cache[1] = Cn = Cn_model(phase, mol, T, P)
    Cn_cache[0] += 1
    return T * exp((S - S_model(phase, mol, T, P)) / Cn)

def xiter_T_at_SP(T, S, S_model, phase_mol, P, Cn_model, Cn_cache):
    # Used to solve for temperature at given entropy 
    counter, Cn = Cn_cache
    if not counter % 5: Cn_cache[1] = Cn = Cn_model(phase_mol, T, P)
    Cn_cache[0] += 1
    return T * exp((S - S_model(phase_mol, T, P)) / Cn)


# %% Abstract mixture

class Mixture:
    """
    Abstract class for estimating mixture properties.
    
    Abstract methods
    ----------------
    Cn(phase, mol, T) : float
        Molar isobaric heat capacity [J/mol/K].
    H(phase, mol, T) : float
        Enthalpy [J/mol].
    S(phase, mol, T, P) : float
        Entropy [J/mol].
    mu(phase, mol, T, P) : float
        Dynamic viscosity [Pa*s].
    V(phase, mol, T, P) : float
        Molar volume [m^3/mol].
    kappa(phase, mol, T, P) : float
        Thermal conductivity [W/m/K].
    Hvap(mol, T) : float
        Heat of vaporization [J/mol]
    sigma(mol, T, P) : float
        Surface tension [N/m].
    epsilon(mol, T, P) : float
        Relative permitivity [-]
    
    Abstract attributes
    MWs : 1d array[float]
        Component molecular weights [g/mol].
    
    """
    maxiter = 20
    T_tol = 1e-6
    __slots__ = ()
    
    def MW(self, mol):
        """Return molecular weight [g/mol] given molar array [mol]."""
        if mol.__class__ is not SparseVector: mol = SparseVector(mol)
        total_mol = mol.sum()
        return (mol * self.MWs).sum() / total_mol if total_mol else 0.
    
    def rho(self, phase, mol, T, P):
        """Mixture density [kg/m^3]"""
        MW = self.MW(mol)
        return fn.V_to_rho(self.V(phase, mol, T, P), MW) if MW else 0.
    
    def Cp(self, phase, mol, T):
        """Mixture isobaric heat capacity [J/g/K]"""
        MW = self.MW(mol)
        return self.Cn(phase, mol, T) / MW if MW else 0.
    
    def alpha(self, phase, mol, T, P):
        """Mixture thermal diffusivity [m^2/s]."""
        Cp = self.Cp(phase, mol, T)
        return fn.alpha(self.kappa(phase, mol, T, P), 
                        self.rho(phase, mol, T, P), 
                        Cp * 1000.) if Cp else 0.
    
    def nu(self, phase, mol, T, P):
        """Mixture kinematic viscosity [m^2/s]."""
        rho = self.rho(phase, mol, T, P)
        return fn.mu_to_nu(self.mu(phase, mol, T, P), 
                           rho) if rho else 0.
    
    def Pr(self, phase, mol, T, P):
        """Mixture Prandtl number [-]."""
        Cp = self.Cp(phase, mol, T)
        return fn.Pr(Cp * 1000.,
                     self.kappa(phase, mol, T, P), 
                     self.mu(phase, mol, T, P)) if Cp else 0.
    
    def xrho(self, phase_mol, T, P):
        """Multi-phase mixture density [kg/m3]."""
        return sum([self.rho(phase, mol, T, P) for phase, mol in phase_mol])
    
    def xCp(self, phase_mol, T):
        """Multi-phase mixture isobaric heat capacity [J/g/K]."""
        return sum([self.Cp(phase, mol, T) for phase, mol in phase_mol])
    
    def xalpha(self, phase_mol, T, P):
        """Multi-phase mixture thermal diffusivity [m^2/s]."""
        return sum([self.alpha(phase, mol, T, P) for phase, mol in phase_mol])
    
    def xnu(self, phase_mol, T, P):
        """Multi-phase mixture kinematic viscosity [m^2/s]."""
        return sum([self.nu(phase, mol, T, P) for phase, mol in phase_mol])
    
    def xPr(self, phase_mol, T, P):
        """Multi-phase mixture Prandtl number [-]."""
        return sum([self.Pr(phase, mol, T, P) for phase, mol in phase_mol])
    
    def get_property(self, name, units, *args, **kwargs):
        """
        Return property in requested units.

        Parameters
        ----------
        name : str
            Name of stream property.
        units : str
            Units of measure.
        *args, **kwargs :
            Phase, material and thermal condition.

        """
        value = getattr(self, name)(*args, **kwargs)
        units_dct = thermo_units.chemical_units_of_measure
        if name in units_dct:
            original_units = units_dct[name]
        else:
            raise ValueError(f"'{name}' is not thermodynamic property")
        return original_units.convert(value, units)
    
    def H(self, phase, mol, T, P):
        """Return enthalpy [J/mol]."""
        H = self._H(phase, mol, T, P)
        if self.include_excess_energies: H += self._H_excess(phase, mol, T, P)
        return H
    
    def S(self, phase, mol, T, P):
        """Return entropy in [J/mol/K]."""
        if mol.__class__ is not SparseVector: mol = SparseVector(mol)
        if not mol.dct: return 0.
        S = self._S(phase, mol, T, P)
        if self.include_excess_energies:
            S += self._S_excess(phase, mol, T, P)
        return S
    
    def _load_free_energy_args(self, *args): pass
    def _load_xfree_energy_args(self, *args): pass
    
    def solve_T_at_HP(self, phase, mol, H, T_guess, P):
        """Solve for temperature in Kelvin."""
        self._load_free_energy_args(phase, mol, T_guess, P)
        try:
            args = (H, self.H, phase, mol, P, self.Cn, [0, None])
            T_guess = flx.aitken(iter_T_at_HP, T_guess, self.T_tol, args, self.maxiter, checkiter=False)
            T = iter_T_at_HP(T_guess, *args)
            return (
                flx.aitken_secant(
                    lambda T: self.H(phase, mol, T, P) - H,
                    x0=T_guess, x1=T, xtol=self.T_tol, ytol=0.
                )
                if abs(T - T_guess) > self.T_tol else T
            )
        finally:
            self._free_energy_args.clear()
        
    def xsolve_T_at_HP(self, phase_mol, H, T_guess, P):
        """Solve for temperature in Kelvin."""
        phase_mol = tuple(phase_mol)
        self._load_xfree_energy_args(phase_mol, T_guess, P)
        try:
            args = (H, self.xH, phase_mol, P, self.xCn, [0, None])
            T_guess = flx.aitken(xiter_T_at_HP, T_guess, self.T_tol, args, self.maxiter, checkiter=False)
            T = xiter_T_at_HP(T_guess, *args)
            return (
                flx.aitken_secant(
                    lambda T: self.xH(phase_mol, T, P) - H,
                    x0=T_guess, x1=T, xtol=self.T_tol, ytol=0.
                )
                if abs(T - T_guess) > self.T_tol else T
            )
        finally:
            self._free_energy_args.clear()
    
    def solve_T_at_SP(self, phase, mol, S, T_guess, P):
        """Solve for temperature in Kelvin."""
        self._load_free_energy_args(phase, mol, T_guess, P)
        try:
            args = (S, self.S, phase, mol, P, self.Cn, [0, None])
            T_guess = flx.aitken(iter_T_at_SP, T_guess, self.T_tol, args, self.maxiter, checkiter=False)
            T = iter_T_at_SP(T_guess, *args)
            return (
                flx.aitken_secant(
                    lambda T: self.S(phase, mol, T, P) - S,
                    x0=T_guess, x1=T, xtol=self.T_tol, ytol=0.
                )
                if abs(T - T_guess) > self.T_tol else T
            )
        finally:
            self._free_energy_args.clear()
        
    def xsolve_T_at_SP(self, phase_mol, S, T_guess, P):
        """Solve for temperature in Kelvin."""
        phase_mol = tuple(phase_mol)
        self._load_xfree_energy_args(phase_mol, T_guess, P)
        try:
            args = (S, self.xS, phase_mol, P, self.xCn, [0, None])
            T_guess = flx.aitken(xiter_T_at_SP, T_guess, self.T_tol, args, self.maxiter, checkiter=False)
            T = xiter_T_at_SP(T_guess, *args)
            return (
                flx.aitken_secant(
                    lambda T: self.xS(phase_mol, T, P) - S,
                    x0=T_guess, x1=T, xtol=self.T_tol, ytol=0.
                )
                if abs(T - T_guess) > self.T_tol else T
            )
        finally:
            self._free_energy_args.clear()
    
    def xCn(self, phase_mol, T, P=None):
        """Multi-phase mixture molar isobaric heat capacity [J/mol/K]."""
        return sum([self.Cn(phase, mol, T, P) for phase, mol in phase_mol])
    
    def xH(self, phase_mol, T, P):
        """Multi-phase mixture enthalpy [J/mol]."""
        H = self.H
        phase_mol = tuple(phase_mol)
        H_total = sum([H(phase, mol, T, P) for phase, mol in phase_mol])
        return H_total
    
    def xS(self, phase_mol, T, P):
        """Multi-phase mixture entropy [J/mol/K]."""
        S = self.S
        phase_mol = tuple(phase_mol)
        S_total = sum([S(phase, mol, T, P) for phase, mol in phase_mol])
        return S_total
    
    def xV(self, phase_mol, T, P):
        """Multi-phase mixture molar volume [mol/m^3]."""
        return sum([self.V(phase, mol, T, P) for phase, mol in phase_mol])
    
    def xmu(self, phase_mol, T, P):
        """Multi-phase mixture hydrolic [Pa*s]."""
        return sum([self.mu(phase, mol, T, P) for phase, mol in phase_mol])
    
    def xkappa(self, phase_mol, T, P):
        """Multi-phase mixture thermal conductivity [W/m/K]."""
        return sum([self.kappa(phase, mol, T, P) for phase, mol in phase_mol])
    
        
    def __repr__(self):
        return f"{type(self).__name__}(...)"
    
    def _info(self):
        return (f"{type(self).__name__}(...)")
    
    def show(self):
        print(self._info())
    _ipython_display_ = show


# %% Ideal mixture

class IdealMixture(Mixture):
    """
    Create an Mixture object for estimating mixture properties.
    
    Parameters
    ----------
    Cn : function(phase, mol, T)
        Molar isobaric heat capacity mixture model [J/mol/K].
    H : function(phase, mol, T)
        Enthalpy mixture model [J/mol].
    S : function(phase, mol, T, P)
        Entropy mixture model [J/mol].
    H_excess : function(phase, mol, T, P)
        Excess enthalpy mixture model [J/mol].
    S_excess : function(phase, mol, T, P)
        Excess entropy mixture model [J/mol].
    mu : function(phase, mol, T, P)
        Dynamic viscosity mixture model [Pa*s].
    V : function(phase, mol, T, P)
        Molar volume mixture model [m^3/mol].
    kappa : function(phase, mol, T, P)
        Thermal conductivity mixture model [W/m/K].
    Hvap : function(mol, T)
        Heat of vaporization mixture model [J/mol]
    sigma : function(mol, T, P)
        Surface tension mixture model [N/m].
    epsilon : function(mol, T, P)
        Relative permitivity mixture model [-]
    MWs : 1d array[float]
        Component molecular weights [g/mol].
    include_excess_energies=False : bool
        Whether to include excess energies
        in enthalpy and entropy calculations.
    
    Notes
    -----
    Although the mixture models are on a molar basis, this is only if the molar
    data is normalized before the calculation (i.e. the `mol` parameter is 
    normalized before being passed to the model).
    
    See also
    --------
    IdealTMixtureModel
    IdealTPMixtureModel
    
    Attributes
    ----------
    include_excess_energies : bool
        Whether to include excess energies
        in enthalpy and entropy calculations.
    Cn(phase, mol, T) : 
        Mixture molar isobaric heat capacity [J/mol/K].
    mu(phase, mol, T, P) : 
        Mixture dynamic viscosity [Pa*s].
    V(phase, mol, T, P) : 
        Mixture molar volume [m^3/mol].
    kappa(phase, mol, T, P) : 
        Mixture thermal conductivity [W/m/K].
    Hvap(mol, T, P) : 
        Mixture heat of vaporization [J/mol]
    sigma(mol, T, P) : 
        Mixture surface tension [N/m].
    epsilon(mol, T, P) : 
        Mixture relative permitivity [-].
    MWs : 1d-array[float]
        Component molecular weights [g/mol].
    
    """
    __slots__ = (
        'include_excess_energies',
        'Cn', 'mu', 'V', 'kappa',
        'Hvap', 'sigma', 'epsilon',
        'MWs', '_H', '_H_excess', '_S', '_S_excess',
        '_free_energy_args',
    )
    
    def __init__(self, Cn, H, S, H_excess, S_excess,
                 mu, V, kappa, Hvap, sigma, epsilon,
                 MWs, include_excess_energies=False):
        self.include_excess_energies = include_excess_energies
        self.Cn = Cn
        self.mu = mu
        self.V = V
        self.kappa = kappa
        self.Hvap = Hvap
        self.sigma = sigma
        self.epsilon = epsilon
        self.MWs = MWs
        self._H = H
        self._S = S
        self._H_excess = H_excess
        self._S_excess = S_excess
        self._free_energy_args = {}
    
    @classmethod
    def from_chemicals(cls, chemicals, 
                       include_excess_energies=False, 
                       cache=True):
        """
        Create a Mixture object from chemical objects.
        
        Parameters
        ----------
        chemicals : Iterable[Chemical]
            For retrieving pure component chemical data.
        include_excess_energies=False : bool
            Whether to include excess energies in enthalpy and entropy calculations.
        rule : str, optional
            Mixing rule. Defaults to 'ideal'.
        cache : optional
            Whether or not to use cached chemicals and cache new chemicals. Defaults to True.
    
        See also
        --------
        :class:`~.mixture.Mixture`
        :class:`~.IdealMixtureModel`
    
        Examples
        --------
        Calculate enthalpy of evaporation for a water and ethanol mixture:
        
        >>> from thermosteam import IdealMixture
        >>> mixture = IdealMixture.from_chemicals(['Water', 'Ethanol'])
        >>> mixture.Hvap([0.2, 0.8], 350)
        39750.62

        Calculate density for a water and ethanol mixture in g/L:

        >>> from thermosteam import Mixture
        >>> mixture = IdealMixture.from_chemicals(['Water', 'Ethanol'])
        >>> mixture.get_property('rho', 'g/L', 'l', [0.2, 0.8], 350, 101325)
        754.23
        
        """
        isa = isinstance
        if isa(chemicals, CompiledChemicals):
            try:
                MWs = chemicals.MW
            except:
                breakpoint()
            chemicals = chemicals.tuple
        else:
            chemicals = [(i if isa(i, Chemical) else Chemical(i, cache=cache)) for i in chemicals]
            MWs = chemical_data_array(chemicals, 'MW')
        getfield = getattr
        Cn =  create_mixture_model(chemicals, 'Cn', IdealTMixtureModel)
        H =  create_mixture_model(chemicals, 'H', IdealTPMixtureModel)
        S = create_mixture_model(chemicals, 'S', IdealEntropyModel)
        H_excess = create_mixture_model(chemicals, 'H_excess', IdealTPMixtureModel)
        S_excess = create_mixture_model(chemicals, 'S_excess', IdealTPMixtureModel)
        mu = create_mixture_model(chemicals, 'mu', IdealTPMixtureModel)
        V = create_mixture_model(chemicals, 'V', IdealTPMixtureModel)
        kappa = create_mixture_model(chemicals, 'kappa', IdealTPMixtureModel)
        Hvap = IdealHvapModel(chemicals)
        sigma = SinglePhaseIdealTMixtureModel([getfield(i, 'sigma') for i in chemicals], 'sigma')
        epsilon = SinglePhaseIdealTMixtureModel([getfield(i, 'epsilon') for i in chemicals], 'epsilon')
        return cls(Cn, H, S, H_excess, S_excess,
                   mu, V, kappa, Hvap, sigma, epsilon, MWs, include_excess_energies)
    
    def __repr__(self):
        return f"{type(self).__name__}(..., include_excess_energies={self.include_excess_energies})"
    
    def _info(self):
        return (f"{type(self).__name__}(...\n"
                f"    include_excess_energies={self.include_excess_energies}\n"
                 ")")
    
    
# %% Thermo mixture

class EOSMixture(Mixture):
    __slots__ = (
        'chemicals', 'eos_chemicals', 'Cn_ideal', 'mu', 'V', 'kappa',
        'Hvap', 'sigma', 'epsilon', 'H_ideal', 'S_ideal', 'MWs', 
        '_free_energy_args',
    )
    chemsep_db = None
    
    def __init__(self, chemicals, eos_chemicals, Cn_ideal, H_ideal, S_ideal, mu, V, kappa, Hvap, sigma, epsilon, MWs):
        self.chemicals = chemicals
        self.eos_chemicals = eos_chemicals
        self.Cn_ideal = Cn_ideal
        self.H_ideal = H_ideal
        self.S_ideal = S_ideal
        self.mu = mu
        self.V = V
        self.kappa = kappa
        self.Hvap = Hvap
        self.sigma = sigma
        self.epsilon = epsilon
        self.MWs = MWs
        self._free_energy_args = {}
    
    def eos_args(self, phase, mol, T, P):
        chemicals = self.chemicals
        dct = mol.dct
        eos_chemicals = self.eos_chemicals
        chemical_subset = []
        mol_subset = []
        eos_mol = 0
        for i, j in dct.items():
            chemical = chemicals[i]
            if chemical in eos_chemicals:
                chemical_subset.append(chemical)
                mol_subset.append(j)
                eos_mol += j
        zs = [i / eos_mol for i in mol_subset]
        eos_chemicals = tuple(chemical_subset)
        key = (phase, eos_chemicals)
        cache = self.cache
        only_g = phase == 'g'
        only_l = phase == 'l'
        if key in cache:
            eos = cache[key].to_TP_zs(
                T=T, P=P, zs=zs, only_g=only_g, only_l=only_l,
                fugacities=False
            )
        else:
            data = tmo.ChemicalData(eos_chemicals)
            if self.chemsep_db is None:
                kijs = None
            else:
                try:
                    kijs = IPDB.get_ip_asymmetric_matrix(self.chemsep_db, data.CASs, 'kij')
                except:
                    kijs = None
            self.cache[key] = eos = self.EOS(
                Tcs=data.Tcs, Pcs=data.Pcs, omegas=data.omegas, kijs=kijs,
                T=T, P=P, zs=zs, only_g=only_g, only_l=only_l,
                fugacities=False
            )
        return eos, eos_mol, dict(
            P=P, zs=zs, only_g=only_g, only_l=only_l,
            fugacities=False
        )

    def _load_free_energy_args(self, phase, mol, T, P):
        self._free_energy_args[phase] = self.eos_args(phase, mol, T, P)
    
    def _load_xfree_energy_args(self, phase_mol, T, P):
        fea = self._free_energy_args
        for phase, mol in phase_mol: 
            if mol.dct: fea[phase] = self.eos_args(phase, mol, T, P)
    
    def Cn(self, phase, mol, T, P):
        if mol.__class__ is not SparseVector: mol = SparseVector(mol)
        if not mol.dct: return 0
        Cn = self.Cn_ideal(phase, mol, T, P)
        if phase != 's':
            if phase in self._free_energy_args:
                eos, eos_mol, eos_kwargs = self._free_energy_args[phase]
            else:
                eos, eos_mol, eos_kwargs = self.eos_args(
                    phase, mol, T, P
                )
            eos = eos.to_TP_zs(
                T=T, **eos_kwargs
            )
            if phase == 'l':
                try: Cn += eos.Cn_dep_l * eos_mol
                except: 
                    try: Cn += eos.Cn_dep_g * eos_mol
                    except: pass
            else:
                try: Cn += eos.Cn_dep_g * eos_mol
                except: pass
        return Cn
    
    def H(self, phase, mol, T, P):
        """Return enthalpy [J/mol]."""
        if mol.__class__ is not SparseVector: mol = SparseVector(mol)
        if not mol.dct: return 0
        H = self.H_ideal(phase, mol, T, P)
        if phase != 's':
            if phase in self._free_energy_args:
                eos, eos_mol, eos_kwargs = self._free_energy_args[phase]
            else:
                eos, eos_mol, eos_kwargs = self.eos_args(
                    phase, mol, T, P
                )
            eos = eos.to_TP_zs(
                T=T, **eos_kwargs
            )
            if phase == 'l':
                try: H += eos.H_dep_l * eos_mol
                except: 
                    try: H += eos.H_dep_g * eos_mol
                    except: pass
            else:
                try: H += eos.H_dep_g * eos_mol
                except: pass
        return H
    
    def S(self, phase, mol, T, P):
        """Return entropy [J/mol/K]."""
        if mol.__class__ is not SparseVector: mol = SparseVector(mol)
        if not mol.dct: return 0
        S = self.S_ideal(phase, mol, T, P)
        if phase != 's':
            if phase in self._free_energy_args:
                eos, eos_mol, eos_kwargs = self._free_energy_args[phase]
            else:
                eos, eos_mol, eos_kwargs = self.eos_args(
                    phase, mol, T, P
                )
            eos = eos.to_TP_zs(
                T=T, **eos_kwargs
            )
            if phase == 'l':
                try: S += eos.S_dep_l * eos_mol
                except: 
                    try: S += eos.S_dep_g * eos_mol
                    except: pass
            else:
                try: S += eos.S_dep_g * eos_mol
                except: pass
        return S
    
    @classmethod
    def from_chemicals(cls, chemicals, eos_chemicals=None, cache=True):
        """
        Create a EOSMixture object from chemical objects.
        
        Parameters
        ----------
        chemicals : Iterable[Chemical]
            For retrieving pure component chemical data.
        eos_chemicals : Iterable[Chemical]
            Chemicals with equations of state.
        cache : optional
            Whether or not to use cached chemicals and cache new chemicals. Defaults to True.
    
        Examples
        --------
        Calculate enthalpy of evaporation for a water and ethanol mixture:
        
        >>> from thermosteam import PRMixture
        >>> mixture = PRMixture.from_chemicals(['Water', 'Ethanol'])
        >>> mixture.Hvap([0.2, 0.8], 350)
        39750.62

        Calculate density for a water and ethanol mixture in g/L:

        >>> from thermosteam import PRMixture
        >>> mixture = PRMixture.from_chemicals(['Water', 'Ethanol'])
        >>> mixture.get_property('rho', 'g/L', 'l', [0.2, 0.8], 350, 101325)
        754.23
        
        """
        isa = isinstance
        if isa(chemicals, CompiledChemicals):
            MWs = chemicals.MW
            chemicals = chemicals.tuple
        else:
            chemicals = [(i if isa(i, Chemical) else Chemical(i, cache=cache)) for i in chemicals]
            MWs = chemical_data_array(chemicals, 'MW')
        if eos_chemicals is None:
            eos_chemicals = tuple(chemicals)
        getfield = getattr
        Cn = create_mixture_model(chemicals, 'Cn', IdealTMixtureModel)
        H = create_mixture_model(chemicals, 'H', IdealTPMixtureModel)
        S = create_mixture_model(chemicals, 'S', IdealEntropyModel)
        mu = create_mixture_model(chemicals, 'mu', IdealTPMixtureModel)
        V = create_mixture_model(chemicals, 'V', IdealTPMixtureModel)
        kappa = create_mixture_model(chemicals, 'kappa', IdealTPMixtureModel)
        Hvap = IdealHvapModel(chemicals)
        sigma = SinglePhaseIdealTMixtureModel([getfield(i, 'sigma') for i in chemicals], 'sigma')
        epsilon = SinglePhaseIdealTMixtureModel([getfield(i, 'epsilon') for i in chemicals], 'epsilon')
        return cls(chemicals, eos_chemicals, Cn, H, S, mu, V, kappa, Hvap, sigma, epsilon, MWs)
    
    @classmethod
    def subclass(cls, EOS, name=None):
        if name is None: name = EOS.__name__.replace('MIX', '') + 'Mixture'
        return type(name, (cls,), dict(EOS=EOS, cache={}))

    def _info(self):
        return (
            f"{type(self).__name__}(\n"
            f"    eos_chemicals=[{', '.join([i.ID for i in self.eos_chemicals])}]\n"
             ")"
        )

dct = globals()    
clsnames = []
for name in ('PRMIX', 'SRKMIX', 'PR78MIX', 'VDWMIX', 'PRSVMIX',
             'PRSV2MIX', 'TWUPRMIX', 'TWUSRKMIX', 'APISRKMIX', 'IGMIX', 'RKMIX',
             'PRMIXTranslatedConsistent', 'PRMIXTranslatedPPJP', 'PRMIXTranslated',
             'SRKMIXTranslatedConsistent', 'PSRK', 'MSRKMIXTranslated',
             'SRKMIXTranslated'):
    cls = EOSMixture.subclass(getattr(eos_mix, name))
    clsname = cls.__name__
    clsnames.append(clsname)
    dct[clsname] = cls

dct['PRMixture'].chemsep_db = 'ChemSep PR'
__all__ = (*__all__, *clsnames)
del dct, clsnames