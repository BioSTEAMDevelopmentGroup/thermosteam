# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 02:38:40 2019

@author: yoelr
"""
from .handle_builder import HandleBuilder
from .thermo_model_handle import TDependentModelHandle, TPDependentModelHandle
from .functor import functor_lookalike
from .utils import shallow_copy
from ..settings import settings
from ..exceptions import UndefinedPhase

__all__ = ('PhaseProperty', #'PhasePropertyBuilder', 
           'ChemicalPhaseTProperty', 'ChemicalPhaseTPProperty',
           'ChemicalPhaseTPropertyBuilder', 'ChemicalPhaseTPPropertyBuilder',
           'MixturePhaseTPProperty', 'MixturePhaseTProperty')

# %% Utilities

getattr = getattr
phase_equivalents = settings.phase_equivalents

def set_phase_property(phase_property, phase, builder, data):
    if not builder: return
    setattr(phase_property, phase, builder(data))
    

# %% Abstract class    

@functor_lookalike
class PhaseProperty:
    __slots__ = ('s', 'l', 'g', 'var')
    
    def __init__(self, s=None, l=None, g=None, var=None):
        self.s = s
        self.l = l
        self.g = g
        self.var = var

    def __getattr__(self, phase):
        try:
            phase = phase_equivalents[phase]
        except KeyError:
            raise UndefinedPhase(phase)
        return getattr(self, phase)

    def __bool__(self):
        return any((self.s, self.l, self.g)) 
    
    def copy(self):
        return self.__class__(shallow_copy(self.s),
                              shallow_copy(self.l),
                              shallow_copy(self.g))


# %% Pure component

class ChemicalPhaseTProperty(PhaseProperty):
    __slots__ = ()
    
    def __init__(self, s=None, l=None, g=None, var=None):
        self.s = TDependentModelHandle() if s is None else s
        self.l = TDependentModelHandle() if l is None else l
        self.g = TDependentModelHandle() if g is None else g
        self.var = var
    
    def __call__(self, phase, T):
        return getattr(self, phase)(T)
    
    
class ChemicalPhaseTPProperty(PhaseProperty):
    __slots__ = ()
    
    def __init__(self, s=None, l=None, g=None, var=None):
        self.s = TPDependentModelHandle() if s is None else s
        self.l = TPDependentModelHandle() if l is None else l
        self.g = TPDependentModelHandle() if g is None else g
        self.var = var
    
    def __call__(self, phase, T, P):
        return getattr(self, phase)(T, P)
    

# %% Mixture
    
class MixturePhaseTProperty(PhaseProperty):
    __slots__ = ()
    
    def at_TP(self, phase, z, TP):
        return getattr(self, phase).at_TP(z, TP)
    
    def __call__(self, phase, z, T):
        return getattr(self, phase)(z, T)

        
class MixturePhaseTPProperty(PhaseProperty):
    __slots__ = ()
    
    def at_TP(self, phase, z, TP):
        return getattr(self, phase).at_TP(z, TP)
    
    def __call__(self, phase, z, T, P):
        return getattr(self, phase).at_TP(z, T, P)
    

# %% Builders

class PhasePropertyBuilder:
    __slots__ = ('s', 'l', 'g', 'var')
    
    def __init__(self, s, l, g, var):
        self.s = s
        self.l = l
        self.g = g
        self.var = var
        
    def __call__(self, sdata, ldata, gdata, phase_property=None):
        if phase_property is None: phase_property = self.PhaseProperty(var=self.var) 
        phases = ('s', 'g', 'l')
        builders = (self.s, self.g, self.l)
        phases_data = (sdata, gdata, ldata)
        for phase, builder, data in zip(phases, builders, phases_data):
            set_phase_property(phase_property, phase, builder, data)
        return phase_property

class ChemicalPhaseTPropertyBuilder(PhasePropertyBuilder):
    __slots__ = ()
    PhaseProperty = ChemicalPhaseTProperty
    
        
class ChemicalPhaseTPPropertyBuilder(PhasePropertyBuilder):
    __slots__ = ()
    PhaseProperty = ChemicalPhaseTPProperty
        
        
        
        