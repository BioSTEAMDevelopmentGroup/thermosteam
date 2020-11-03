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

__all__ = ('Mixture',)

def iter_temperature(T, H, H_model, phase, mol, P, Cn):
    # Used to solve for ethalpy at given temperature
    return T + (H - H_model(phase, mol, T, P)) / Cn

def xiter_temperature(T, H, H_model, phase_mol, P, Cn):
    # Used to solve for ethalpy at given temperature
    return T + (H - H_model(phase_mol, T, P)) / Cn

# %% Ideal mixture

class Mixture:
    """
    Create an Mixture object for estimating mixture properties.
    
    Parameters
    ----------
    rule : str
        Description of mixing rules used.
    Cn : function(phase, mol, T)
        Molar heat capacity mixture model [J/mol/K].
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
    include_excess_energies=False : bool
        Whether to include excess energies
        in enthalpy and entropy calculations.
    
    Notes
    -----
    Although the mixture models are on a molar basis, this is only if the molar data is normalized before the calculation (i. e. the `mol` parameter is normalized before being passed to the model).
    
    See also
    --------
    IdealMixtureModel
    :func:`~.mixture_builders.ideal_mixture`
    
    Attributes
    ----------
    rule : str
        Description of mixing rules used.
    include_excess_energies : bool
        Whether to include excess energies
        in enthalpy and entropy calculations.
    Cn(phase, mol, T) : 
        Mixture molar heat capacity [J/mol/K].
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
    
    
    """
    __slots__ = ('rule',
                 'rigorous_energy_balance',
                 'include_excess_energies',
                 'Cn', 'mu', 'V', 'kappa',
                 'Hvap', 'sigma', 'epsilon',
                 '_H', '_H_excess', '_S', '_S_excess',
    )
    
    def __init__(self, rule, Cn, H, S, H_excess, S_excess,
                 mu, V, kappa, Hvap, sigma, epsilon,
                 include_excess_energies=False):
        self.rule = rule
        self.include_excess_energies = include_excess_energies
        self.Cn = Cn
        self.mu = mu
        self.V = V
        self.kappa = kappa
        self.Hvap = Hvap
        self.sigma = sigma
        self.epsilon = epsilon
        self._H = H
        self._S = S
        self._H_excess = H_excess
        self._S_excess = S_excess
    
    def H(self, phase, mol, T, P):
        """Return enthalpy [J/mol]."""
        H = self._H(phase, mol, T)
        if self.include_excess_energies:
            H += self._H_excess(phase, mol, T, P)
        return H
    
    def S(self, phase, mol, T, P):
        """Return entropy in [J/mol]."""
        S = self._S(phase, mol, T, P)
        if self.include_excess_energies:
            S += self._S_excess(phase, mol, T, P)
        return S
    
    def solve_T(self, phase, mol, H, T_guess, P):
        """Solve for temperature in Kelvin."""
        args = (H, self.H, phase, mol, P, self.Cn(phase, mol, T_guess))
        return flx.aitken(iter_temperature, T_guess, 1e-6, args, 50, checkiter=True)
        
    def xsolve_T(self, phase_mol, H, T_guess, P):
        """Solve for temperature in Kelvin."""
        args = (H, self.xH, tuple(phase_mol), P, self.xCn(phase_mol, T_guess))
        return flx.aitken(xiter_temperature, T_guess, 1e-6, args, 50, checkiter=True)
        
    def xCn(self, phase_mol, T):
        """Multi-phase mixture heat capacity [J/mol/K]."""
        Cn = self.Cn
        return sum([Cn(phase, mol, T) for phase, mol in phase_mol])
    
    def xH(self, phase_mol, T, P):
        """Multi-phase mixture enthalpy [J/mol]."""
        H = self._H
        H_total = sum([H(phase, mol, T) for phase, mol in phase_mol])
        if self.include_excess_energies:
            H_excess = self._H_excess
            H_total += sum([H_excess(phase, mol, T, P) for phase, mol in phase_mol])
        return H_total
    
    def xS(self, phase_mol, T, P):
        """Multi-phase mixture entropy [J/mol]."""
        S = self._S
        S_total = sum([S(phase, mol, T, P) for phase, mol in phase_mol])
        if self.include_excess_energies:
            S_excess = self._S_excess
            S_total += sum([S_excess(phase, mol, T, P) for phase, mol in phase_mol])
        return S_total
    
    def xV(self, phase_mol, T, P):
        """Multi-phase mixture molar volume [mol/m^3]."""
        V = self.V
        return sum([V(phase, mol, T, P) for phase, mol in phase_mol])
    
    def xmu(self, phase_mol, T, P):
        """Multi-phase mixture hydrolic [Pa*s]."""
        mu = self.mu
        return sum([mu(phase, mol, T, P) for phase, mol in phase_mol])
    
    def xkappa(self, phase_mol, T, P):
        """Multi-phase mixture thermal conductivity [W/m/K]."""
        kappa = self.kappa
        return sum([kappa(phase, mol, T, P) for phase, mol in phase_mol])
    
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
        