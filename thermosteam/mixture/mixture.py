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
import numpy as np
from math import exp
from thermosteam import functional as fn
from .. import units_of_measure as thermo_units
from ..base import PhaseHandle, MockPhaseTHandle, MockPhaseTPHandle
from .ideal_mixture_model import (
    SinglePhaseIdealTMixtureModel,
    IdealTMixtureModel, 
    IdealTPMixtureModel, 
    IdealEntropyModel, 
    IdealHvapModel
)
from .._chemicals import Chemical, CompiledChemicals, chemical_data_array

__all__ = ('Mixture',)

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
    if not counter % 5: Cn_cache[1] = Cn = Cn_model(phase, mol, T)
    Cn_cache[0] += 1
    return T + (H - H_model(phase, mol, T, P)) / Cn

def xiter_T_at_HP(T, H, H_model, phase_mol, P, Cn_model, Cn_cache):
    # Used to solve for temperature at given ethalpy 
    counter, Cn = Cn_cache
    if not counter % 5: Cn_cache[1] = Cn = Cn_model(phase_mol, T)
    Cn_cache[0] += 1
    return T + (H - H_model(phase_mol, T, P)) / Cn

def iter_T_at_SP(T, S, S_model, phase, mol, P, Cn_model, Cn_cache):
    # Used to solve for temperature at given entropy 
    counter, Cn = Cn_cache
    if not counter % 5: Cn_cache[1] = Cn = Cn_model(phase, mol, T)
    Cn_cache[0] += 1
    return T * exp((S - S_model(phase, mol, T, P)) / Cn)

def xiter_T_at_SP(T, S, S_model, phase_mol, P, Cn_model, Cn_cache):
    # Used to solve for temperature at given entropy 
    counter, Cn = Cn_cache
    if not counter % 5: Cn_cache[1] = Cn = Cn_model(phase_mol, T)
    Cn_cache[0] += 1
    return T * exp((S - S_model(phase_mol, T, P)) / Cn)


# %% Ideal mixture

class Mixture:
    """
    Create an Mixture object for estimating mixture properties.
    
    Parameters
    ----------
    rule : str
        Description of mixing rules used.
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
    rule : str
        Description of mixing rules used.
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
    maxiter = 20
    T_tol = 1e-6
    __slots__ = ('rule',
                 'rigorous_energy_balance',
                 'include_excess_energies',
                 'Cn', 'mu', 'V', 'kappa',
                 'Hvap', 'sigma', 'epsilon',
                 'MWs', '_H', '_H_excess', '_S', '_S_excess',
    )
    
    def __init__(self, rule, Cn, H, S, H_excess, S_excess,
                 mu, V, kappa, Hvap, sigma, epsilon,
                 MWs, include_excess_energies=False):
        self.rule = rule
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
    
    @classmethod
    def from_chemicals(cls, chemicals, 
                       include_excess_energies=False, 
                       rule='ideal',
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
        
        >>> from thermosteam import Mixture
        >>> mixture = Mixture.from_chemicals(['Water', 'Ethanol'])
        >>> mixture.Hvap([0.2, 0.8], 350)
        39601.089

        Calculate density for a water and ethanol mixture in g/L:

        >>> from thermosteam import Mixture
        >>> mixture = Mixture.from_chemicals(['Water', 'Ethanol'])
        >>> mixture.get_property('rho', 'g/L', 'l', [0.2, 0.8], 350, 101325)
        752.513
        
        """
        if rule == 'ideal':
            isa = isinstance
            if isa(chemicals, CompiledChemicals):
                MWs = chemicals.MW
                chemicals = chemicals.tuple
            else:
                chemicals = [(i if isa(i, Chemical) else Chemical(i)) for i in chemicals]
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
            return cls(rule, Cn, H, S, H_excess, S_excess,
                       mu, V, kappa, Hvap, sigma, epsilon, MWs, include_excess_energies)
        else:
            raise ValueError("rule '{rule}' is not available (yet)")
    
    def MW(self, mol):
        """Return molecular weight [g/mol] given molar array [mol]."""
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
        total_mol = mol.sum()
        if total_mol == 0.: return 0.
        S = self._S(phase, mol, T, P)
        if self.include_excess_energies:
            S += self._S_excess(phase, mol, T, P)
        return S
    
    def solve_T_at_HP(self, phase, mol, H, T_guess, P):
        """Solve for temperature in Kelvin."""
        args = (H, self.H, phase, mol, P, self.Cn, [0, None])
        T_guess = flx.aitken(iter_T_at_HP, T_guess, self.T_tol, args, self.maxiter, checkiter=False)
        T = iter_T_at_HP(T_guess, *args)
        return (
            flx.aitken_secant(
                lambda T: self.H(phase, mol, T, P) - H,
                x0=T_guess, x1=T, xtol=self.T_tol, ytol=0.
            )
            if abs(T - T_guess) < self.T_tol else T
        )
        
    def xsolve_T_at_HP(self, phase_mol, H, T_guess, P):
        """Solve for temperature in Kelvin."""
        phase_mol = tuple(phase_mol)
        args = (H, self.xH, phase_mol, P, self.xCn, [0, None])
        T_guess = flx.aitken(xiter_T_at_HP, T_guess, self.T_tol, args, self.maxiter, checkiter=False)
        T = xiter_T_at_HP(T_guess, *args)
        return (
            flx.aitken_secant(
                lambda T: self.xH(phase_mol, T, P) - H,
                x0=T_guess, x1=T, xtol=self.T_tol, ytol=0.
            )
            if abs(T - T_guess) < self.T_tol else T
        )
    
    def solve_T_at_SP(self, phase, mol, S, T_guess, P):
        """Solve for temperature in Kelvin."""
        args = (S, self.S, phase, mol, P, self.Cn, [0, None])
        T_guess = flx.aitken(iter_T_at_SP, T_guess, self.T_tol, args, self.maxiter, checkiter=False)
        T = iter_T_at_SP(T_guess, *args)
        return (
            flx.aitken_secant(
                lambda T: self.S(phase, mol, T, P) - S,
                x0=T_guess, x1=T, xtol=self.T_tol, ytol=0.
            )
            if abs(T - T_guess) < self.T_tol else T
        )
        
    def xsolve_T_at_SP(self, phase_mol, S, T_guess, P):
        """Solve for temperature in Kelvin."""
        phase_mol = tuple(phase_mol)
        args = (S, self.xS, phase_mol, P, self.xCn, [0, None])
        T_guess = flx.aitken(xiter_T_at_SP, T_guess, self.T_tol, args, self.maxiter, checkiter=False)
        T = xiter_T_at_SP(T_guess, *args)
        return (
            flx.aitken_secant(
                lambda T: self.xS(phase_mol, T, P) - S,
                x0=T_guess, x1=T, xtol=self.T_tol, ytol=0.
            )
            if abs(T - T_guess) < self.T_tol else T
        )
    
    def xCn(self, phase_mol, T, P=None):
        """Multi-phase mixture molar isobaric heat capacity [J/mol/K]."""
        return sum([self.Cn(phase, mol, T) for phase, mol in phase_mol])
    
    def xH(self, phase_mol, T, P):
        """Multi-phase mixture enthalpy [J/mol]."""
        H = self._H
        phase_mol = tuple(phase_mol)
        H_total = sum([H(phase, mol, T, P) for phase, mol in phase_mol])
        if self.include_excess_energies:
            H_excess = self._H_excess
            H_total += sum([H_excess(phase, mol, T, P) for phase, mol in phase_mol])
        return H_total
    
    def xS(self, phase_mol, T, P):
        """Multi-phase mixture entropy [J/mol/K]."""
        S = self._S
        phase_mol = tuple(phase_mol)
        S_total = sum([S(phase, mol, T, P) for phase, mol in phase_mol])
        if self.include_excess_energies:
            S_excess = self._S_excess
            S_total += sum([S_excess(phase, mol, T, P) for phase, mol in phase_mol])
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
        return f"{type(self).__name__}(rule={repr(self.rule)}, ..., include_excess_energies={self.include_excess_energies})"
    
    def _info(self):
        return (f"{type(self).__name__}(\n"
                f"    rule={repr(self.rule)}, ...\n"
                f"    include_excess_energies={self.include_excess_energies}\n"
                 ")")
    
    def show(self):
        print(self._info())
    _ipython_display_ = show
        