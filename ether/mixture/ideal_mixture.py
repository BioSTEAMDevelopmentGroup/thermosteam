# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 07:37:35 2019

@author: yoelr
"""
from ..base import MixturePhaseTProperty, MixturePhaseTPProperty, display_asfunctor, Units # TODO: Incorporate units in every instance
from ..settings import settings
from numpy import asarray, logical_and, logical_or, zeros
from flexsolve import SolverError

__all__ = ('IdealMixture',
           'IdealMixturePhaseTProperty',
           'IdealMixturePhaseTPProperty',
           'IdealMixtureTProperty',
           'IdealMixtureTPProperty')

# %% Mixture properties

class IdealMixtureTPProperty:
    __slots__ = ('_properties', '_TP', '_nonzero', '_data')

    def __init__(self, properties):
        self._TP = (0., 0.)
        self._nonzero = None
        self._data = zeros(len(properties))
        self._properties = properties

    def copy(self):
        new = self.__new__(self.__class__)
        new._TP = self._TP
        new._data = self._data.copy()
        new._properties = self._properties
        return new

    @property
    def var(self):
        for i in self._properties:
            try:
                var = i.var
                if var: return var
            except: pass

    def __call__(self, z, T, P):
        z = asarray(z)
        self._nonzero = nonzero = z!=0
        if (T, P) != self._TP:
            self._data[:] = 0.
            iscallable = callable
            self._data[nonzero] = [i(T, P) if iscallable(i) else i
                                   for i,j in zip(self._properties, nonzero) if j]
            self._TP = (T, P)
        else:
            nomatch = self._nonzero != nonzero
            new_nonzero = logical_and(nonzero, nomatch)
            iscallable = callable
            self._data[new_nonzero] = [i(T, P) if iscallable(i) else i
                                       for i,j in zip(self._properties, new_nonzero) if j]
            self._nonzero = logical_or(self._nonzero, nonzero)
        return (z * self._data).sum()
    
    def __repr__(self):
        return f"<{display_asfunctor(self)}>"


class IdealMixtureTProperty:
    __slots__ = ('_properties', '_T', '_nonzero', '_data')
    var = IdealMixtureTPProperty.var
    __repr__ = IdealMixtureTPProperty.__repr__

    def __init__(self, properties):
        self._T = 0.0
        self._nonzero = None
        self._data = zeros(len(properties))
        self._properties = tuple(properties)

    def copy(self):
        new = self.__new__(self.__class__)
        new._T = self._T
        new._data = self._data.copy()
        new._properties = self._properties
        return new

    def from_stream(stream, units):
        # TODO: Left off here
        thermal_condition = stream._thermal_condition
        

    def __call__(self, z, T):
        z = asarray(z)
        self._nonzero = nonzero = z!=0
        if T != self._T:
            self._data[:] = 0.
            iscallable = callable
            self._data[nonzero] = [i(T) if iscallable(i) else i
                                   for i,j in zip(self._properties, nonzero) if j]
            self._T = T
        else:
            nomatch = self._nonzero != nonzero
            new_nonzero = logical_and(nonzero, nomatch)
            self._data[new_nonzero] = [i(T) if iscallable(i) else i
                                       for i,j in zip(self._properties, new_nonzero) if j]
            self._nonzero = logical_or(self._nonzero, nonzero)
        return (z * self._data).sum()


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
    __slots__ = ('s', 'l', 'g')
    
    @classmethod
    def from_phase_properties(cls, phase_properties):
        setfield = setattr
        self = cls.__new__(cls)
        for phase, properties in group_properties_by_phase(phase_properties).items():
            setfield(self, phase, IdealMixtureTProperty(properties))
        return self


class IdealMixturePhaseTPProperty(MixturePhaseTPProperty):
    __slots__ = ('s', 'l', 'g')
    
    @classmethod
    def from_phase_properties(cls, phase_properties):
        setfield = setattr
        self = cls.__new__(cls)
        for phase, properties in group_properties_by_phase(phase_properties).items():
            setfield(self, phase, IdealMixtureTPProperty(properties))
        return self


# %% Ideal mixture

mixture_phaseT_methods = ('H', 'Cp')
mixture_phaseTP_methods = ('H_excess', 'S_excess', 'mu', 'V', 'k', 'S')
mixture_T_methods  = ('Hvap', 'sigma', 'epsilon')
mixture_methods = (*mixture_phaseT_methods,
                   *mixture_phaseTP_methods,
                   *mixture_T_methods)

class IdealMixture:
    __slots__ = ('chemicals', 'rigorous_energy_balance', *mixture_methods)
    
    def __init__(self, chemicals=(), rigorous_energy_balance=False):
        self.rigorous_energy_balance = rigorous_energy_balance
        getfield = getattr
        setfield = setattr
        any_ = any
        self.chemicals = chemicals
        for var in mixture_phaseT_methods:
            phase_properties = [getfield(i, var) for i in chemicals]
            if any_(phase_properties): 
                phase_property = IdealMixturePhaseTProperty.from_phase_properties(phase_properties)
                setfield(self, var, phase_property)
        for var in mixture_phaseTP_methods:
            phase_properties = [getfield(i, var) for i in chemicals]
            if any_(phase_properties): 
                phase_property = IdealMixturePhaseTPProperty.from_phase_properties(phase_properties)
                setfield(self, var, phase_property)
        for var in mixture_T_methods:
            properties = [getfield(i, var) for i in chemicals]
            if any_(properties): setfield(self, var, IdealMixtureTProperty(properties))
    
    def copy(self):
        getfield = getattr
        setfield = setattr
        new = self.__new__(self.__class__)
        new.chemicals = self.chemicals
        for i in mixture_methods:
            setfield(new, i, getfield(self, i).copy())
        return new
    
    def solve_T(self, phase, z, H, T_guess):
        if self.rigorous_energy_balance:
            # First approximation
            Cp = self.Cp(phase, z, T_guess)
            T = T_guess + (H - self.H(phase, z, T_guess))/Cp
        
            # Solve enthalpy by iteration
            it = 0
            it2 = 0
            Cp = self.Cp(phase, z, T_guess)
            while abs(T - T_guess) > 0.01:
                T_guess = T
                T += (H - self.H(phase, z, T))/Cp
                if it == 5:
                    it = 0
                    it2 += 1
                    Cp = self.Cp(phase, z, T_guess)
                    if it2 > 10:
                        raise SolverError("could not solve temperature "
                                          "given enthalpy")
                else: it += 1
        else:
            return T_guess + (H - self.H(phase, z, T_guess))/self.Cp(phase, z, T_guess)    
                
    def xsolve_T(self, phase_data, H, T_guess):
        T = T_guess
        if self.rigorous_energy_balance:
            # First approximation
            Cp = self.xCp(phase_data, T_guess)
            T = T_guess + (H - self.xH(phase_data, T_guess))/Cp
        
            # Solve enthalpy by iteration
            it = 0
            it2 = 0
            Cp = self.xCp(phase_data, T_guess)
            while abs(T - T_guess) > 0.01:
                T_guess = T
                T += (H - self.xH(phase_data, T))/Cp
                if it == 5:
                    it = 0
                    it2 += 1
                    Cp = self.xCp(phase_data, T_guess)
                    if it2 > 10:
                        raise SolverError("could not solve temperature "
                                          "given enthalpy")
                else: it += 1
        else:
            return T_guess + (H - self.xH(phase_data, T_guess))/self.xCp(phase_data, T_guess)   
        
    
    def xCp(self, phase_data, T):
        Cp = self.Cp
        return sum([Cp(phase, z, T) for phase, z in phase_data])
    
    def xH(self, phase_data, T):
        H = self.H
        return sum([H(phase, z, T) for phase, z in phase_data])
    
    def xS(self, phase_data, T, P):
        S = self.S
        return sum([S(phase, z, T, P) for phase, z in phase_data])
    
    def xH_excess(self, phase_data, T, P):
        H_excess = self.H_excess
        return sum([H_excess(phase, z, T, P) for phase, z in phase_data])
    
    def xS_excess(self, phase_data, T, P):
        S_excess = self.S_excess
        return sum([S_excess(phase, z, T, P) for phase, z in phase_data])
    
    def xV(self, phase_data, T, P):
        V = self.V
        return sum([V(phase, z, T, P) for phase, z in phase_data])
    
    def xmu(self, phase_data, T, P):
        mu = self.mu
        return sum([mu(phase, z, T, P) for phase, z in phase_data])
    
    def xk(self, phase_data, T, P):
        k = self.k
        return sum([k(phase, z, T, P) for phase, z in phase_data])
    
    def __repr__(self):
        IDs = [str(i) for i in self.chemicals]
        return f"{type(self).__name__}([{', '.join(IDs)}])"
    
    def show(self):
        IDs = [str(i) for i in self.chemicals]
        info = f"{type(self).__name__}: {', '.join(IDs)}"
        getfield = getattr
        for i in self.__slots__[1:]:
            f = getfield(self, i)
            if callable(f):
                info += f"\n {display_asfunctor(f, name=i, var=i, show_var=False)}"
            else:
                info += f"\n {i}: {f}"
            
        print(info)
    
    _ipython_display_ = show