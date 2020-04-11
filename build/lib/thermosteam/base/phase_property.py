# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 02:38:40 2019

@author: yoelr
"""
from .thermo_model_handle import TDependentModelHandle, TPDependentModelHandle
from .functor import functor_lookalike
from ..utils import copy_maybe

__all__ = ('PhaseProperty', #'PhasePropertyBuilder', 
           'PhaseTProperty', 'PhaseTPProperty',
           'PhaseTPropertyBuilder', 'PhaseTPPropertyBuilder',
           'PhaseZTPProperty', 'PhaseZTProperty')

# %% Utilitie

def set_phase_property(phase_property, phase, builder, data):
    if not builder: return
    if hasattr(builder, 'from_args'):
        model_handle = builder.from_args(data)
        setattr(phase_property, phase, model_handle)
    else:
        model_handle = getattr(phase_property, phase)
        builder.build(model_handle, *data)
    

# %% Abstract class    

@functor_lookalike
class PhaseProperty:
    __slots__ = ('var', 's', 'l', 'g')
    
    def __init__(self, var, s=None, l=None, g=None):
        self.var = var
        self.s = s
        self.l = l
        self.g = g

    @property
    def S(self): return self.s
    @property
    def L(self): return self.l
    @property
    def G(self): return self.g
    
    def __bool__(self):
        return any((self.s, self.l, self.g)) 
    
    def copy(self):
        return self.__class__(self.var,
                              copy_maybe(self.s),
                              copy_maybe(self.l),
                              copy_maybe(self.g))
    __copy__ = copy


# %% Pure component

class PhaseTProperty(PhaseProperty):
    __slots__ = ()
    
    def __init__(self, var, s=None, l=None, g=None):
        self.var = var
        self.s = TDependentModelHandle(var + '.s') if s is None else s
        self.l = TDependentModelHandle(var + '.l') if l is None else l
        self.g = TDependentModelHandle(var + '.g') if g is None else g
    
    def __call__(self, phase, T):
        return getattr(self, phase)(T)
    
    
class PhaseTPProperty(PhaseProperty):
    __slots__ = ()
    
    def __init__(self, var, s=None, l=None, g=None):
        self.var = var
        self.s = TPDependentModelHandle(var + '.s') if s is None else s
        self.l = TPDependentModelHandle(var + '.l') if l is None else l
        self.g = TPDependentModelHandle(var + '.g') if g is None else g
    
    def __call__(self, phase, T, P):
        return getattr(self, phase)(T, P)
    

# %% Mixture
    
class PhaseZTProperty(PhaseProperty):
    __slots__ = ()
    
    def at_TP(self, phase, z, TP):
        return getattr(self, phase).at_TP(z, TP)
    
    def __call__(self, phase, z, T):
        return getattr(self, phase)(z, T)

        
class PhaseZTPProperty(PhaseProperty):
    __slots__ = ()
    
    def at_TP(self, phase, z, TP):
        return getattr(self, phase).at_TP(z, TP)
    
    def __call__(self, phase, z, T, P):
        return getattr(self, phase)(z, T, P)
    

# %% Builders

class PhasePropertyBuilder:
    __slots__ = ('var', 's', 'l', 'g')
    
    def __init__(self, var, s, l, g):
        self.var = var
        self.s = s
        self.l = l
        self.g = g
        
    def __call__(self, sdata, ldata, gdata, phase_property=None):
        if phase_property is None: phase_property = self.PhaseProperty(self.var) 
        phases = ('s', 'g', 'l')
        builders = (self.s, self.g, self.l)
        phases_data = (sdata, gdata, ldata)
        for phase, builder, data in zip(phases, builders, phases_data):
            set_phase_property(phase_property, phase, builder, data)
        return phase_property

class PhaseTPropertyBuilder(PhasePropertyBuilder):
    __slots__ = ()
    PhaseProperty = PhaseTProperty
    
        
class PhaseTPPropertyBuilder(PhasePropertyBuilder):
    __slots__ = ()
    PhaseProperty = PhaseTPProperty
        
        
        
        