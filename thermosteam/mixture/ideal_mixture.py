# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 07:37:35 2019

@author: yoelr
"""
from ..base import MixturePhaseTProperty, MixturePhaseTPProperty, display_asfunctor, PhaseProperty
from .mixture_method_names import mixture_methods, mixture_phaseT_methods, mixture_phaseTP_methods, mixture_T_methods, mixture_hidden_T_methods, mixture_hidden_phaseTP_methods
from flexsolve import SolverError
import numpy as np

__all__ = ('IdealMixture',
           'IdealMixturePhaseTProperty',
           'IdealMixturePhaseTPProperty',
           'IdealMixtureTProperty',
           'IdealMixtureTPProperty')

# %% Cache

class ThermalCache:
    __slots__ = ('condition', 'nonzero', 'data')
    
    def __init__(self, condition, nonzero, data):
        self.condition = condition
        self.nonzero = nonzero
        self.data = data
    
    @property
    def tuple(self):
        return self.condition, self.nonzero, self.data
    
    def __iter__(self):
        yield self.condition
        yield self.nonzero
        yield self.data
        
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

class IdealMixtureTPProperty:
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


class IdealMixtureTProperty:
    __slots__ = ('var', '_properties', '_cache')
    __repr__ = IdealMixtureTPProperty.__repr__

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
    
class IdealMixturePhaseTProperty(MixturePhaseTProperty):
    __slots__ = ()
    
    @classmethod
    def from_phase_properties(cls, phase_properties, var):
        setfield = setattr
        self = cls.__new__(cls)
        for phase, properties in group_properties_by_phase(phase_properties).items():
            setfield(self, phase, IdealMixtureTProperty(properties, var))
        self.var = var
        return self


class IdealMixturePhaseTPProperty(MixturePhaseTPProperty):
    __slots__ = ()
    
    @classmethod
    def from_phase_properties(cls, phase_properties, var):
        setfield = setattr
        self = cls.__new__(cls)
        for phase, properties in group_properties_by_phase(phase_properties).items():
            setfield(self, phase, IdealMixtureTPProperty(properties, var))
        self.var = var
        return self


# %% Ideal mixture



class IdealMixture:
    __slots__ = ('chemicals', 'rigorous_energy_balance', 
                 'include_excess_energies', *mixture_methods)
    units = {}
    
    def __init__(self, chemicals=(), rigorous_energy_balance=False, include_excess_energies=False):
        self.rigorous_energy_balance = rigorous_energy_balance
        self.include_excess_energies = include_excess_energies
        getfield = getattr
        setfield = setattr
        any_ = any
        self.chemicals = chemicals
        # TODO: Divide this up to functions
        for attr in mixture_hidden_T_methods:
            var = attr[1:]
            phase_properties = [getfield(i, var) for i in chemicals]
            if any_(phase_properties): 
                phase_property = IdealMixturePhaseTProperty.from_phase_properties(phase_properties, var)
                setfield(self, attr, phase_property)
        for attr in mixture_hidden_phaseTP_methods:
            var = attr[1:]
            phase_properties = [getfield(i, var) for i in chemicals]
            if any_(phase_properties): 
                phase_property = IdealMixturePhaseTPProperty.from_phase_properties(phase_properties, var)
                setfield(self, attr, phase_property)
        for var in mixture_phaseT_methods:
            phase_properties = [getfield(i, var) for i in chemicals]
            if any_(phase_properties): 
                phase_property = IdealMixturePhaseTProperty.from_phase_properties(phase_properties, var)
                setfield(self, var, phase_property)
        for var in mixture_phaseTP_methods:
            phase_properties = [getfield(i, var) for i in chemicals]
            if any_(phase_properties): 
                phase_property = IdealMixturePhaseTPProperty.from_phase_properties(phase_properties, var)
                setfield(self, var, phase_property)
        for var in mixture_T_methods:
            properties = [getfield(i, var) for i in chemicals]
            if any_(properties): setfield(self, var, IdealMixtureTProperty(properties, var))
    
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
            while abs(T - T_guess) > 0.01:
                T_guess = T
                if it == 3:
                    it = 0
                    it2 += 1
                    Cn = self.Cn(phase, z, T_guess)
                    if it2 > 5: break # Its good enough, no need to find exact solution
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
            while abs(T - T_guess) > 0.01:
                T_guess = T
                if it == 3:
                    it = 0
                    it2 += 1
                    Cn = self.xCn(phase_data, T_guess)
                    if it2 > 5: break # Its good enough, no need to find exact solution
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
        IDs = [str(i) for i in self.chemicals]
        return f"{type(self).__name__}([{', '.join(IDs)}])"
    