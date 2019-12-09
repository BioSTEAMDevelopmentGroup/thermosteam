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
    __slots__ = ('var', 'units', '_properties', '_cache')

    def __init__(self, properties, var):
        self._properties = properties
        self._cache = {}
        self._set_var(var)

    _set_var = PhaseProperty._set_var

    def at_thermal_condition(self, z, thermal_condition):
        cache = self._cache
        nonzero = z!=0
        T, P = TP = thermal_condition.tuple
        properties = self._properties
        if thermal_condition in cache:
            thermal_cache = cache[thermal_condition]
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
            cache[thermal_condition] = ThermalCache(TP, nonzero, data)
        return (z * data).sum()

    def __call__(self, z, T, P):
        z = np.asarray(z)
        data = np.zeros_like(z)
        set_properties_at_T(data, self._properties, z!=0, T, P)
        return (z * data).sum()
    
    def __repr__(self):
        return f"<{display_asfunctor(self)}>"


class IdealMixtureTProperty:
    __slots__ = ('var', 'units', '_properties', '_cache')
    __repr__ = IdealMixtureTPProperty.__repr__

    def __init__(self, properties, var):
        self._properties = tuple(properties)
        self._cache = {}
        self._set_var(var)

    _set_var = PhaseProperty._set_var

    def at_thermal_condition(self, z, thermal_condition):
        cache = self._cache
        nonzero = z!=0
        T = thermal_condition.T
        properties = self._properties
        if thermal_condition in cache:
            thermal_cache = cache[thermal_condition]
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
            cache[thermal_condition] = ThermalCache(T, nonzero, data)
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
        self._set_var(var)
        return self


class IdealMixturePhaseTPProperty(MixturePhaseTPProperty):
    __slots__ = ()
    
    @classmethod
    def from_phase_properties(cls, phase_properties, var):
        setfield = setattr
        self = cls.__new__(cls)
        for phase, properties in group_properties_by_phase(phase_properties).items():
            setfield(self, phase, IdealMixtureTPProperty(properties, var))
        self._set_var(var)
        return self


# %% Ideal mixture



class IdealMixture:
    __slots__ = ('chemicals', 'rigorous_energy_balance', 
                 'include_excess_energies', *mixture_methods)
    
    def __init__(self, chemicals=(), rigorous_energy_balance=False, include_excess_energies=False):
        self.rigorous_energy_balance = rigorous_energy_balance
        self.include_excess_energies = include_excess_energies
        getfield = getattr
        setfield = setattr
        any_ = any
        self.chemicals = chemicals
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
    
    def H(self, phase, T, P):
        if self.include_excess_energies:
            return self._H(phase, T) + self._H_excess(phase, T, P)
        else:
            return self._H(phase, T)
            
    def solve_T(self, phase, z, H, T_guess, P):
        if self.rigorous_energy_balance:
            # First approximation
            Cp = self.Cp(phase, z, T_guess)
            T = T_guess + (H - self.H(phase, z, T_guess, P))/Cp
        
            # Solve enthalpy by iteration
            it = 0
            it2 = 0
            Cp = self.Cp(phase, z, T_guess)
            while abs(T - T_guess) > 0.01:
                T_guess = T
                T += (H - self.H(phase, z, T, P))/Cp
                if it == 5:
                    it = 0
                    it2 += 1
                    Cp = self.Cp(phase, z, T_guess)
                    if it2 > 10:
                        raise SolverError("could not solve temperature "
                                          "given enthalpy")
                else: it += 1
        else:
            return T_guess + (H - self.H(phase, z, T_guess, P))/self.Cp(phase, z, T_guess)    
                
    def xsolve_T(self, phase_data, H, T_guess, P):
        T = T_guess
        if self.rigorous_energy_balance:
            # First approximation
            Cp = self.xCp(phase_data, T_guess)
            T = T_guess + (H - self.xH(phase_data, T_guess, P))/Cp
        
            # Solve enthalpy by iteration
            it = 0
            it2 = 0
            Cp = self.xCp(phase_data, T_guess)
            while abs(T - T_guess) > 0.01:
                T_guess = T
                T += (H - self.xH(phase_data, T, P))/Cp
                if it == 5:
                    it = 0
                    it2 += 1
                    Cp = self.xCp(phase_data, T_guess)
                    if it2 > 10:
                        raise SolverError("could not solve temperature "
                                          "given enthalpy")
                else: it += 1
        else:
            return T_guess + (H - self.xH(phase_data, T_guess, P))/self.xCp(phase_data, T_guess)   
    
    def xCp(self, phase_data, T):
        Cp = self.Cp
        return sum([Cp(phase, z, T) for phase, z in phase_data])
    
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
    
    def xk(self, phase_data, T, P):
        k = self.k
        return sum([k(phase, z, T, P) for phase, z in phase_data])
    
    def xCp_at_thermal_condition(self, phase_data, thermal_condition):
        Cp = self.Cp.at_thermal_condition
        return sum([Cp(phase, z, thermal_condition) for phase, z in phase_data])
    
    def xH_at_thermal_condition(self, phase_data, thermal_condition):
        H = self._H.at_thermal_condition
        H_total = sum([H(phase, z, thermal_condition) for phase, z in phase_data])
        if self.include_excess_energies:
            H_excess = self._H_excess.at_thermal_condition
            H_total += sum([H_excess(phase, z, thermal_condition) for phase, z in phase_data])
        return H_total
    
    def xS_at_thermal_condition(self, phase_data, thermal_condition):
        S = self.S.at_thermal_condition
        S_total = sum([S(phase, z, thermal_condition) for phase, z in phase_data])
        if self.include_excess_energies:
            S_excess = self._S_excess.at_thermal_condition
            S_total += sum([S_excess(phase, z, thermal_condition) for phase, z in phase_data])
        return S_total
    
    def xV_at_thermal_condition(self, phase_data, thermal_condition):
        V = self.V.at_thermal_condition
        return sum([V(phase, z, thermal_condition) for phase, z in phase_data])
    
    def xmu_at_thermal_condition(self, phase_data, thermal_condition):
        mu = self.mu.at_thermal_condition
        return sum([mu(phase, z, thermal_condition) for phase, z in phase_data])
    
    def xk_at_thermal_condition(self, phase_data, thermal_condition):
        k = self.k.at_thermal_condition
        return sum([k(phase, z, thermal_condition) for phase, z in phase_data])
    
    def __repr__(self):
        IDs = [str(i) for i in self.chemicals]
        return f"{type(self).__name__}([{', '.join(IDs)}])"
    