# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 02:38:40 2019

@author: yoelr
"""
from .handle_builder import HandleBuilder
from .functor import display_asfunctor

__all__ = ('PhaseProperty', #'PhasePropertyBuilder', 
           'ChemicalPhaseTProperty', 'ChemicalPhaseTPProperty',
           'ChemicalPhaseTPropertyBuilder', 'ChemicalPhaseTPPropertyBuilder',
           'MixturePhaseTPProperty', 'MixturePhaseTProperty')

# %% Utilities

def set_phase_property(phase_property, phase, builder, data):
    if not builder: return
    if isinstance(builder, HandleBuilder):
        prop = builder(data)
    else:
        prop = builder(data)
    setattr(phase_property, phase, prop)


# %% Abstract class

class PhaseProperty:
    __slots__ = ('s', 'l', 'g')
    
    def __init__(self, s=None, l=None, g=None):
        self.s = s
        self.l = l
        self.g = g

    def __bool__(self):
        return any((self.s, self.l, self.g)) 
        
    @property
    def var(self):
        for phase in self.__slots__:
            try:
                var = getattr(self, phase).var
                if var: return var.split('.')[0]
            except: pass
        
    def __repr__(self):
        return f"<{display_asfunctor(self)}>"


# %% Pure component

class ChemicalPhaseTProperty(PhaseProperty):
    __slots__ = ()
    
    def __call__(self, phase, T):
        return getattr(self, phase)(T)
    
    
class ChemicalPhaseTPProperty(PhaseProperty):
    __slots__ = ()
    
    def __call__(self, phase, T, P):
        return getattr(self, phase)(T, P)
    

# %% Mixture
    
class MixturePhaseTProperty(PhaseProperty):
    __slots__ = ()
    
    def __call__(self, phase, z, T):
        return getattr(self, phase)(z, T)

        
class MixturePhaseTPProperty(PhaseProperty):
    __slots__ = ()
    
    def __call__(self, phase, z, T, P):
        return getattr(self, phase)(z, T, P)
    

# %% Builders

class PhasePropertyBuilder:
    __slots__ = ('s', 'l', 'g')
    
    def __init__(self, s, l, g):
        self.s = s
        self.l = l
        self.g = g
        
    def __call__(self, sdata, ldata, gdata, phase_property=None):
        pp = phase_property or self.PhaseProperty()
        phases = ('s', 'g', 'l')
        builders = (self.s, self.g, self.l)
        phases_data = (sdata, gdata, ldata)
        for phase, builder, data in zip(phases, builders, phases_data):
            set_phase_property(pp, phase, builder, data)
        return pp

class ChemicalPhaseTPropertyBuilder(PhasePropertyBuilder):
    __slots__ = ()
    PhaseProperty = ChemicalPhaseTProperty
    
        
class ChemicalPhaseTPPropertyBuilder(PhasePropertyBuilder):
    __slots__ = ()
    PhaseProperty = ChemicalPhaseTPProperty
        
        
        
        