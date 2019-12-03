# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:18:27 2019

@author: yoelr
"""
from . import equilibrium as eq
from .chemicals import Chemicals
from .mixture import IdealMixture
from .utils import read_only
from .exceptions import assert_value
from .settings import settings

__all__ = ('Thermo',)

@read_only
class Thermo:
    __slots__ = ('chemicals', 'mixture', 'Gamma', 'Phi', 'PCF') 
    _cached = {}
    def __new__(cls, chemicals, mixture=None,
                Gamma=eq.DortmundActivityCoefficients,
                Phi=eq.IdealFugacityCoefficients,
                PCF=eq.IdealPoyintingCorrectionFactor):
        args = (chemicals, mixture, Gamma, Phi, PCF)
        cached = cls._cached
        if args in cached:
            self = cached[args]
        else:
            self = super().__new__(cls)
            mixture = mixture or IdealMixture(chemicals)
            if not isinstance(chemicals, Chemicals):
                chemicals = Chemicals(chemicals)
            chemicals.compile()
            if settings._debug:
                assert_value(issubclass, Gamma, 'Gamma', eq.ActivityCoefficients)
                assert_value(issubclass, Phi, 'Phi', eq.FugacityCoefficients)
                assert_value(issubclass, PCF, 'PCF', eq.PoyintingCorrectionFactor)
            setattr = object.__setattr__
            setattr(self, 'chemicals', chemicals)
            setattr(self, 'mixture', mixture)
            setattr(self, 'Gamma', Gamma)
            setattr(self, 'Phi', Phi)
            setattr(self, 'PCF', PCF)    
            cached[args] = self
        settings._thermo = self
        return self
    
    def __repr__(self):
        IDs = [i for i in self.chemicals.IDs]
        return f"{type(self).__name__}([{', '.join(IDs)}])"