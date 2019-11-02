# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 02:38:40 2019

@author: yoelr
"""
from .utils import var_with_units
from .handle_builder import HandleBuilder

__all__ = ('PhaseProperty', 'PhasePropertyBuilder')

class PhaseProperty:
    __slots__ = ('var', 's', 'l', 'g')
    
    def __init__(self, var, s=None, l=None, g=None):
        self.var = var
        self.s = s
        self.l = l
        self.g = g
    
    def __call__(self, phase, T, P):
        return getattr(self, phase)(T, P)
    
    def __repr__(self):
        return f"<{type(self).__name__}(phase, T, P) -> {var_with_units(self.var)}>"


class PhasePropertyBuilder:
    __slots__ = ('s', 'l', 'g')
    
    def __init__(self, s, l, g):
        self.s = s
        self.l = l
        self.g = g
        
    def __call__(self, var, sdata, ldata, gdata, phase_property=None):
        isa = isinstance
        attr = setattr
        pp = phase_property or PhaseProperty(var)
        s = self.s
        g = self.g
        l = self.l
        attr(pp, 's', s(var + ".s", sdata) if isa(s, HandleBuilder) else s(sdata))
        attr(pp, 'g', g(var + ".g", gdata) if isa(g, HandleBuilder) else g(gdata))
        attr(pp, 'l', l(var + ".l", ldata) if isa(l, HandleBuilder) else l(ldata))
        return pp
        
        
        
        
        
        
        