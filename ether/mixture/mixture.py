# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 22:33:31 2019

@author: yoelr
"""
from .base import display_asfunctor

chemical_phaseT_methods = ('H', 'Cp')
chemical_phaseTP_methods = ('H_excess', 'S_excess', 'mu', 'V', 'k', 'S')
chemical_T_methods  = ('Hvap', 'sigma', 'epsilon')

class Mixture:
    __slots__ = (('chemicals',)
                 + chemical_phaseTP_methods
                 + chemical_phaseT_methods
                 + chemical_T_methods)
    
    def __init__(self, chemicals=()):
        getfield = getattr
        setfield = setattr
        any_ = any
        self.chemicals = chemicals
        MixturePhaseTProperty = self.MixturePhaseTProperty
        MixturePhaseTPProperty = self.MixturePhaseTPProperty
        MixtureTProperty = self.MixtureTProperty
        for var in chemical_phaseT_methods:
            properties = [getfield(i, var) for i in chemicals]
            if any_(properties): setfield(self, var, MixturePhaseTProperty(properties, var))
        for var in chemical_phaseTP_methods:
            properties = [getfield(i, var) for i in chemicals]
            if any_(properties): setfield(self, var, MixturePhaseTPProperty(properties, var))
        for var in chemical_T_methods:
            properties = [getfield(i, var) for i in chemicals]
            if any_(properties): setfield(self, var, MixtureTProperty(properties, var))
    
    def __repr__(self):
        IDs = [str(i) for i in self.chemicals]
        return f"{type(self).__name__}([{', '.join(IDs)}])"
    
    def show(self):
        IDs = [str(i) for i in self.chemicals]
        info = f"{type(self).__name__}: {', '.join(IDs)}"
        getfield = getattr
        for i in self.__slots__[1:]:
            f = getfield(self, i)
            info += f"\n {display_asfunctor(f, name=i, var=i, show_var=False)}"
        print(info)
    
    _ipython_display_ = show