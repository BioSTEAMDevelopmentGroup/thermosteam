# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 07:37:35 2019

@author: yoelr
"""
from .base import PhaseZTProperty, PhaseZTPProperty, display_asfunctor
import numpy as np

__all__ = ('Mixture',
           'IdealZTProperty',
           'IdealZTPProperty')

# %% Mixture methods

mixture_phaseT_methods = ('Cn',)
mixture_hidden_T_methods = ('_H',)
mixture_phaseTP_methods = ('mu', 'V', 'kappa')
mixture_hidden_phaseTP_methods = ('_H_excess', '_S_excess', '_S')
mixture_T_methods  = ('Hvap', 'sigma', 'epsilon')
mixture_methods = (*mixture_phaseT_methods,
                   *mixture_phaseTP_methods,
                   *mixture_hidden_T_methods,
                   *mixture_hidden_phaseTP_methods,
                   *mixture_T_methods)

# %% Cache

class ThermalCache:
    __slots__ = ('condition', 'nonzero', 'data')
    
    def __init__(self, condition, nonzero, data):
        self.condition = condition
        self.nonzero = nonzero
        self.data = data
        
    def __repr__(self):
        return f"{type(self).__name__}(condition={self.condition}, nonzero={self.nonzero}, data={self.data})"


# %% Utilities

def set_properties_at_T(data, properties, nonzero, T):
    iscallable = callable
    data[nonzero] = [i(T) if iscallable(i) else i
                     for i,j in zip(properties, nonzero) if j]

def set_properties_at_TP(data, properties, nonzero, T, P):
    iscallable = callable
    data[nonzero] = [i(T, P) if iscallable(i) else i
                     for i,j in zip(properties, nonzero) if j]

# %% Mixture properties

class IdealZTPProperty:
    __slots__ = ('var', '_properties', '_cache')

    def __init__(self, properties, var):
        self._properties = properties
        self._cache = {}
        self.var = var

    def at_TP(self, z, TP):
        cache = self._cache
        nonzero = z!=0
        T, P = TP = TP.tuple
        properties = self._properties
        if TP in cache:
            thermal_cache = cache[TP]
            data = thermal_cache.data
            if TP == thermal_cache.condition:
                cache_nonzero = thermal_cache.nonzero
                new_nonzero = (cache_nonzero != nonzero) & nonzero
                set_properties_at_TP(data, properties, new_nonzero, T, P)
                cache_nonzero |= nonzero
            else:
                data[:] = 0
                set_properties_at_TP(data, properties, nonzero, T, P)
                thermal_cache.nonzero = nonzero
                thermal_cache.condition = TP
        else:
            data = np.zeros_like(z)
            set_properties_at_TP(data, properties, nonzero, T, P)
            cache[TP] = ThermalCache(TP, nonzero, data)
        return (z * data).sum()

    def __call__(self, z, T, P):
        z = np.asarray(z)
        data = np.zeros_like(z)
        set_properties_at_T(data, self._properties, z!=0, T, P)
        return (z * data).sum()
    
    def __repr__(self):
        return f"<{display_asfunctor(self)}>"


class IdealZTProperty:
    __slots__ = ('var', '_properties', '_cache')
    __repr__ = IdealZTPProperty.__repr__

    def __init__(self, properties, var):
        self._properties = tuple(properties)
        self._cache = {}
        self.var = var

    def at_TP(self, z, TP):
        cache = self._cache
        nonzero = z!=0
        T = TP.T
        properties = self._properties
        if TP in cache:
            thermal_cache = cache[TP]
            data = thermal_cache.data
            if T == thermal_cache.condition:
                cache_nonzero = thermal_cache.nonzero
                new_nonzero = (cache_nonzero != nonzero) & nonzero
                set_properties_at_T(data, properties, new_nonzero, T)
                cache_nonzero |= nonzero
            else:
                data[:] = 0
                set_properties_at_T(data, properties, nonzero, T)
                thermal_cache.nonzero = nonzero
                thermal_cache.condition = T
        else:
            data = np.zeros_like(z)
            cache[TP] = ThermalCache(T, nonzero, data)
            set_properties_at_T(data, properties, nonzero, T)
        return (z * data).sum()

    def __call__(self, z, T):
        z = np.asarray(z)
        data = np.zeros_like(z)
        set_properties_at_T(data, self._properties, z!=0, T)
        return (z * data).sum()


# %% Ideal mixture phase property
        
def group_properties_by_phase(phase_properties):
    hasfield = hasattr
    getfield = getattr
    iscallable = callable
    properties_by_phase = {'s': [],
                           'l': [],
                           'g': []}
    for phase, properties in properties_by_phase.items():
        for phase_property in phase_properties:
            if iscallable(phase_property) and hasfield(phase_property, phase):
                prop = getfield(phase_property, phase)
            else:
                prop = phase_property
            properties.append(prop)
    return properties_by_phase
    
def build_ideal_PhaseZTProperty(phase_properties, var):
    setfield = setattr
    new = PhaseZTProperty.__new__(PhaseZTProperty)
    for phase, properties in group_properties_by_phase(phase_properties).items():
        setfield(new, phase, IdealZTProperty(properties, var))
    new.var = var
    return new


def build_ideal_PhaseZTPProperty(phase_properties, var):
    setfield = setattr
    new = PhaseZTPProperty.__new__(PhaseZTPProperty)
    for phase, properties in group_properties_by_phase(phase_properties).items():
        setfield(new, phase, IdealZTPProperty(properties, var))
    new.var = var
    return new


# %% Ideal mixture


class Mixture:
    """
    Create an Mixture object for estimating mixture properties.
    
    Parameters
    ----------
    description : str
        Description of mixing rules used.
    Cn, H : PhaseZTProperty
    S, H_excess, S_excess, mu, V, kappa : PhaseZTPProperty
    Hvap, sigma, epsilon: ZTProperty
    rigorous_energy_balance=True : bool
        Whether to rigorously solve for temperature in energy balance or simply approximate.
    include_excess_energies=False : bool
        Whether to include excess energies in enthalpy and entropy calculations.
    
    """
    __slots__ = ('description',
                 'rigorous_energy_balance',
                 'include_excess_energies',
                 *mixture_methods)
    
    def __init__(self, description, Cn, H, S, H_excess, S_excess,
                 mu, V, kappa, Hvap, sigma, epsilon,
                 rigorous_energy_balance=True, include_excess_energies=False):
        self.description = description
        self.rigorous_energy_balance = rigorous_energy_balance
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
    
    @classmethod
    def new_ideal_mixture(cls, chemicals,
                          rigorous_energy_balance=True,
                          include_excess_energies=False):
        """
        Create a Mixture object that computes mixture properties using ideal mixing rules.
        
        Parameters
        ----------
        chemicals : Chemicals
            For retrieving pure component chemical data.
        rigorous_energy_balance=True : bool
            Whether to rigorously solve for temperature in energy balance or simply approximate.
        include_excess_energies=False : bool
            Whether to include excess energies in enthalpy and entropy calculations.

        """
        chemicals = tuple(chemicals)
        
        properties = [i.Cn for i in chemicals]
        Cn =  build_ideal_PhaseZTProperty(properties, 'Cn')
        
        properties = [i.H for i in chemicals]
        H =  build_ideal_PhaseZTProperty(properties, 'H')
        
        properties = [i.S for i in chemicals]
        S = build_ideal_PhaseZTPProperty(properties, 'S')
        
        properties = [i.H_excess for i in chemicals]
        H_excess = build_ideal_PhaseZTPProperty(properties, 'H_excess')
        
        properties = [i.S_excess for i in chemicals]
        S_excess = build_ideal_PhaseZTPProperty(properties, 'S_excess')
        
        properties = [i.mu for i in chemicals]
        mu = build_ideal_PhaseZTPProperty(properties, 'mu')
        
        properties = [i.V for i in chemicals]
        V = build_ideal_PhaseZTPProperty(properties, 'V')
        
        properties = [i.kappa for i in chemicals]
        kappa = build_ideal_PhaseZTPProperty(properties, 'kappa')
        
        properties = [i.Hvap for i in chemicals]
        Hvap = IdealZTProperty(properties, 'Hvap')
        
        properties = [i.sigma for i in chemicals]
        sigma = IdealZTProperty(properties, 'sigma')
        
        properties = [i.epsilon for i in chemicals]
        epsilon = IdealZTProperty(properties, 'epsilon')
        
        return cls('ideal mixing rules', Cn, H, S, H_excess, S_excess,
                   mu, V, kappa, Hvap, sigma, epsilon,
                   rigorous_energy_balance, include_excess_energies)
    
    @property
    def Cn_at_TP(self):
        return self.Cn.at_TP
    @property
    def kappa_at_TP(self):
        return self.kappa.at_TP
    @property
    def mu_at_TP(self):
        return self.mu.at_TP
    @property
    def V_at_TP(self):
        return self.V.at_TP
    @property
    def sigma_at_TP(self):
        return self.sigma.at_TP
    @property
    def epsilon_at_TP(self):
        return self.epsilon.at_TP
    @property
    def Hvap_at_TP(self):
        return self.Hvap.at_TP
    
    def H(self, phase, z, T, P):
        if self.include_excess_energies:
            return self._H(phase, z, T) + self._H_excess(phase, z, T, P)
        else:
            return self._H(phase, z, T)
    
    def S(self, phase, z, T, P):
        if self.include_excess_energies:
            return self._S(phase, z, T, P) + self._S_excess(phase, z, T, P)
        else:
            return self._S(phase, z, T, P)
    
    def H_at_TP(self, phase, z, TP):
        H = self._H.at_TP(phase, z, TP)
        if self.include_excess_energies:
            H += self._H_excess.at_TP(phase, z, TP)
        return H
    
    def S_at_TP(self, phase, z, TP):
        S = self._S.at_TP(phase, z, TP)
        if self.include_excess_energies:
            S += self._S_excess.at_TP(phase, z, TP)
        return S
    
    def solve_T(self, phase, z, H, T_guess, P):
        # First approximation
        Cn = self.Cn(phase, z, T_guess)
        T = T_guess + (H - self.H(phase, z, T_guess, P))/Cn
        if self.rigorous_energy_balance:
            # Solve enthalpy by iteration
            it = 0
            it2 = 0
            Cn = self.Cn(phase, z, T_guess)
            while abs(T - T_guess) > 0.05:
                T_guess = T
                if it == 3:
                    it = 0
                    it2 += 1
                    if it2 > 5: break # Its good enough, no need to find exact solution
                    Cn = self.Cn(phase, z, T_guess)
                else:
                    it += 1
                T += (H - self.H(phase, z, T, P))/Cn
        return T
                
    def xsolve_T(self, phase_data, H, T_guess, P):
        # First approximation
        Cn = self.xCn(phase_data, T_guess)
        T = T_guess + (H - self.xH(phase_data, T_guess, P))/Cn
        if self.rigorous_energy_balance:
            # Solve enthalpy by iteration
            it2 = it = 0
            Cn = self.xCn(phase_data, T_guess)
            while abs(T - T_guess) > 0.05:
                T_guess = T
                if it == 3:
                    it = 0
                    it2 += 1
                    if it2 > 5: break # Its good enough, no need to find exact solution
                    Cn = self.xCn(phase_data, T_guess)
                else:
                    it += 1
                T += (H - self.xH(phase_data, T, P))/Cn
        return T
    
    def xCn(self, phase_data, T):
        Cn = self.Cn
        return sum([Cn(phase, z, T) for phase, z in phase_data])
    
    def xH(self, phase_data, T, P):
        H = self._H
        H_total = sum([H(phase, z, T) for phase, z in phase_data])
        if self.include_excess_energies:
            H = self._H_excess
            H_total += sum([H(phase, z, T, P) for phase, z in phase_data])
        return H_total
    
    def xS(self, phase_data, T, P):
        S = self._S
        S_total = sum([S(phase, z, T, P) for phase, z in phase_data])
        if self.include_excess_energies:
            S = self._S_excess
            S_total += sum([S(phase, z, T, P) for phase, z in phase_data])
        return S_total
    
    def xV(self, phase_data, T, P):
        V = self.V
        return sum([V(phase, z, T, P) for phase, z in phase_data])
    
    def xmu(self, phase_data, T, P):
        mu = self.mu
        return sum([mu(phase, z, T, P) for phase, z in phase_data])
    
    def xkappa(self, phase_data, T, P):
        kappa = self.kappa
        return sum([kappa(phase, z, T, P) for phase, z in phase_data])
    
    def xCn_at_TP(self, phase_data, TP):
        Cn = self.Cn.at_TP
        return sum([Cn(phase, z, TP) for phase, z in phase_data])
    
    def xH_at_TP(self, phase_data, TP):
        H = self._H.at_TP
        H_total = sum([H(phase, z, TP) for phase, z in phase_data])
        if self.include_excess_energies:
            H_excess = self._H_excess.at_TP
            H_total += sum([H_excess(phase, z, TP) for phase, z in phase_data])
        return H_total
    
    def xS_at_TP(self, phase_data, TP):
        S = self.S.at_TP
        S_total = sum([S(phase, z, TP) for phase, z in phase_data])
        if self.include_excess_energies:
            S_excess = self._S_excess.at_TP
            S_total += sum([S_excess(phase, z, TP) for phase, z in phase_data])
        return S_total
    
    def xV_at_TP(self, phase_data, TP):
        V = self.V.at_TP
        return sum([V(phase, z, TP) for phase, z in phase_data])
    
    def xmu_at_TP(self, phase_data, TP):
        mu = self.mu.at_TP
        return sum([mu(phase, z, TP) for phase, z in phase_data])
    
    def xkappa_at_TP(self, phase_data, TP):
        k = self.k.at_TP
        return sum([k(phase, z, TP) for phase, z in phase_data])
    
    def __repr__(self):
        return f"{type(self).__name__}({repr(self.description)}, ..., rigorous_energy_balance={self.rigorous_energy_balance}, include_excess_energies={self.include_excess_energies})"
    