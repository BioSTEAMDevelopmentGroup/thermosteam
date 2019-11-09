# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 07:37:35 2019

@author: yoelr
"""
from ..base import MixturePhaseTProperty, MixturePhaseTPProperty, display_asfunctor
from numpy import asarray, array, logical_and, logical_or

__all__ = ('IdealMixture', 'IdealMixtureTProperty', 'IdealMixtureTPProperty')

# %% Mixture properties

class IdealMixtureTPProperty:
    __slots__ = ('_sources', '_TP', '_nonzero', '_data')

    def __init__(self, sources):
        self._TP = (0., 0.)
        self._nonzero = None
        self._data = None
        self._sources = sources

    @property
    def var(self):
        for i in self._sources:
            try:
                var = i.var
                if var: return var
            except: pass

    def __call__(self, z, T, P):
        z = asarray(z)
        self._nonzero = nonzero = z!=0
        if (T, P) != self._TP:
            self._data = array([(i(T, P) if j else 0.) for i,j in zip(self._sources, nonzero)], dtype=float)
            self._TP = (T, P)
        else:
            nomatch = self._nonzero != nonzero
            new_nonzero = logical_and(nonzero, nomatch)
            self._data[new_nonzero] = [i(T, P) for i,j in zip(self._sources, new_nonzero) if j]
            self._nonzero = logical_or(self._nonzero, nonzero)
        return (z * self._data).sum()
    
    def __repr__(self):
        return f"<{display_asfunctor(self)}>"


class IdealMixtureTProperty:
    __slots__ = ('_sources', '_T', '_nonzero', '_data')
    var = IdealMixtureTPProperty.var
    __repr__ = IdealMixtureTPProperty.__repr__

    def __init__(self, sources):
        self._T = 0.0
        self._nonzero = None
        self._data = None
        self._sources = tuple(sources)

    def __call__(self, z, T):
        z = asarray(z)
        self._nonzero = nonzero = z!=0
        if T != self._T:
            self._data = array([(i(T) if j else 0.) for i,j in zip(self._sources, nonzero)], dtype=float)
            self._T = T
        else:
            nomatch = self._nonzero != nonzero
            new_nonzero = logical_and(nonzero, nomatch)
            self._data[new_nonzero] = [i(T) for i,j in zip(self._sources, new_nonzero) if j]
            self._nonzero = logical_or(self._nonzero, nonzero)
        return (z * self._data).sum()


# %% Ideal mixture phase property
        
class IdealMixturePhaseTProperty(MixturePhaseTProperty):
    __slots__ = ('s', 'l', 'g')
    
    def __init__(self, sources):
        getfield = getattr
        setfield = setattr
        for phase in ('s', 'l', 'g'):
            setfield(self, phase, IdealMixtureTProperty([getfield(i, phase) for i in sources]))


class IdealMixturePhaseTPProperty(MixturePhaseTPProperty):
    __slots__ = ('s', 'l', 'g')
    
    def __init__(self, sources):
        getfield = getattr
        setfield = setattr
        for phase in ('s', 'l', 'g'):
            setfield(self, phase, IdealMixtureTPProperty([getfield(i, phase) for i in sources]))


# %% Ideal mixture

chemical_phaseT_methods = ('Cp', 'H')
chemical_phaseTP_methods = ('H_excess', 'S_excess', 'mu', 'S', 'V', 'k')
chemical_T_methods  = ('Hvap', 'sigma', 'epsilon')

class IdealMixture:
    __slots__ = chemical_phaseTP_methods + chemical_phaseT_methods + chemical_T_methods
    
    def __init__(self, chemicals=()):
        chemical_methods = [i.methods for i in chemicals]
        getfield = getattr
        setfield = setattr
        for var in chemical_phaseT_methods:
            sources = [getfield(i, var) for i in chemical_methods]
            if any(sources): setfield(self, var, IdealMixturePhaseTProperty(sources))
        for var in chemical_phaseTP_methods:
            sources = [getfield(i, var) for i in chemical_methods]
            if any(sources): setfield(self, var, IdealMixturePhaseTPProperty(sources))
        for var in chemical_T_methods:
            sources = [getfield(i, var) for i in chemical_methods]
            if any(sources): setfield(self, var, IdealMixtureTProperty(sources))
    
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