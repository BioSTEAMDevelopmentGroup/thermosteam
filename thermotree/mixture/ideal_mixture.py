# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 07:37:35 2019

@author: yoelr
"""
from ..base import MixturePhaseTProperty, MixturePhaseTPProperty, display_asfunctor
from numpy import asarray, logical_and, logical_or, zeros

__all__ = ('IdealMixture',
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
        iscallable = callable
        if (T, P) != self._TP:
            self._data[:] = 0.
            self._data[nonzero] = [(i(T, P) if iscallable(i) else i)
                                   for i,j in zip(self._properties, nonzero) if j]
            self._TP = (T, P)
        else:
            nomatch = self._nonzero != nonzero
            new_nonzero = logical_and(nonzero, nomatch)
            self._data[new_nonzero] = [(i(T, P) if iscallable(i) else i)
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

    def __call__(self, z, T):
        z = asarray(z)
        self._nonzero = nonzero = z!=0
        iscallable = callable
        if T != self._T:
            self._data[:] = 0.
            self._data[nonzero] = [(i(T) if iscallable(i) else i)
                                   for i,j in zip(self._properties, nonzero) if j]
            self._T = T
        else:
            nomatch = self._nonzero != nonzero
            new_nonzero = logical_and(nonzero, nomatch)
            self._data[new_nonzero] = [(i(T) if iscallable(i) else i)
                                       for i,j in zip(self._properties, new_nonzero) if j]
            self._nonzero = logical_or(self._nonzero, nonzero)
        return (z * self._data).sum()


# %% Ideal mixture phase property
        
def gather_properties_by_phase(phase_properties):
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
    
    def __init__(self, phase_properties):
        setfield = setattr
        for phase, properties in gather_properties_by_phase(phase_properties).items():
            setfield(self, phase, IdealMixtureTProperty(properties))


class IdealMixturePhaseTPProperty(MixturePhaseTPProperty):
    __slots__ = ('s', 'l', 'g')
    
    def __init__(self, phase_properties):
        setfield = setattr
        for phase, properties in gather_properties_by_phase(phase_properties).items():
            setfield(self, phase, IdealMixtureTPProperty(properties))


# %% Ideal mixture

chemical_phaseT_methods = ('Cp', 'H')
chemical_phaseTP_methods = ('H_excess', 'S_excess', 'mu', 'S', 'V', 'k')
chemical_T_methods  = ('Hvap', 'sigma', 'epsilon')

class IdealMixture:
    __slots__ = chemical_phaseTP_methods + chemical_phaseT_methods + chemical_T_methods
    
    def __init__(self, chemicals=()):
        getfield = getattr
        setfield = setattr
        for var in chemical_phaseT_methods:
            properties = [getfield(i, var) for i in chemicals]
            if any(properties): setfield(self, var, IdealMixturePhaseTProperty(properties))
        for var in chemical_phaseTP_methods:
            properties = [getfield(i, var) for i in chemicals]
            if any(properties): setfield(self, var, IdealMixturePhaseTPProperty(properties))
        for var in chemical_T_methods:
            properties = [getfield(i, var) for i in chemicals]
            if any(properties): setfield(self, var, IdealMixtureTProperty(properties))
    
    def __repr__(self):
        return f"<{type(self).__name__}>"
    
    def show(self):
        info = f"{type(self).__name__}:"
        getfield = getattr
        for i in self.__slots__[1:]:
            f = getfield(self, i)
            info += f"\n {display_asfunctor(f, name=i, var=i, show_var=False)}"
        print(info)
    
    _ipython_display_ = show