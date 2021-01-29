# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from .thermo_model_handle import TDependentModelHandle, TPDependentModelHandle
from .functor import functor_lookalike
from ..utils import read_only, cucumber

__all__ = ('PhaseHandle',
           'PhaseTHandle', 'PhaseTPHandle',
           'PhaseTHandleBuilder', 'PhaseTPHandleBuilder',
           'PhaseMixtureHandle',
           'PhaseFunctorBuilder',
           'PhaseTFunctorBuilder',
           'PhaseTPFunctorBuilder')


# %% Abstract class    

@cucumber
@read_only
@functor_lookalike
class PhaseHandle:
    __slots__ = ('var', 's', 'l', 'g')
    
    def __init__(self, var, s=None, l=None, g=None):
        setattr = object.__setattr__
        setattr(self, 'var', var)
        setattr(self, 's', s)
        setattr(self, 'l', l)
        setattr(self, 'g', g)
    
    @classmethod
    def blank(cls, var):
        new = cls.__new__(cls)
        setattr = object.__setattr__
        setattr(new, 'var', var)
        setattr(new, 's', None)
        setattr(new, 'l', None)
        setattr(new, 'g', None)
        return new
    
    @property
    def S(self): return self.s
    @property
    def L(self): return self.l
    @property
    def G(self): return self.g
    
    def __iter__(self):
        if self.s is not None: yield 's', self.s
        if self.l is not None: yield 'l', self.l
        if self.g is not None: yield 'g', self.g 
    
    def __bool__(self):
        return any((self.s, self.l, self.g)) 
    
    def copy(self):
        return self.__class__(self.var,
                              self.s.copy(),
                              self.l.copy(),
                              self.g.copy())
    __copy__ = copy
    
    def show(self):
        print(self)
    _ipython_display_ = show


# %% Pure component

class PhaseTHandle(PhaseHandle):
    __slots__ = ()
    
    def __init__(self, var, s=None, l=None, g=None):
        super().__init__(
            var,
            TDependentModelHandle(var + '.s') if s is None else s,
            TDependentModelHandle(var + '.l') if l is None else l,
            TDependentModelHandle(var + '.g') if g is None else g,
        )
    
    def set_value(self, var, value):
        for i in (self.s, self.l, self.g):
            if i: i.set_value(var, value)
    
    def __call__(self, phase, T, P=None):
        return getattr(self, phase)(T)
    
    
class PhaseTPHandle(PhaseHandle):
    __slots__ = ()
    
    def __init__(self, var, s=None, l=None, g=None):
        super().__init__(
            var,
            TPDependentModelHandle(var + '.s') if s is None else s,
            TPDependentModelHandle(var + '.l') if l is None else l,
            TPDependentModelHandle(var + '.g') if g is None else g,
        )
    
    set_value = PhaseTHandle.set_value
    
    def __call__(self, phase, T, P):
        return getattr(self, phase)(T, P)


# %% Mixture
    
class PhaseMixtureHandle(PhaseHandle):
    __slots__ = ()
    
    def __call__(self, phase, z, T, P=None):
        return getattr(self, phase)(z, T, P)


# %% Builders

class PhaseHandleBuilder:
    __slots__ = ('var', 's', 'l', 'g', 'build_functors')
    
    def __init__(self, var, s, l, g, build_functors=False):
        self.var = var
        self.s = s
        self.l = l
        self.g = g
        self.build_functors = build_functors
        
    def __call__(self, sdata, ldata, gdata, phase_handle=None):
        if phase_handle is None: 
            phase_handle = self.PhaseHandle(self.var)
        phases = ('s', 'g', 'l')
        builders = (self.s, self.g, self.l)
        phases_data = (sdata, gdata, ldata)
        for phase, builder, data in zip(phases, builders, phases_data):
            if builder:
                handle = getattr(phase_handle, phase)
                builder.build(handle, *data)
        return phase_handle


class PhaseTHandleBuilder(PhaseHandleBuilder):
    __slots__ = ()
    PhaseHandle = PhaseTHandle
    
        
class PhaseTPHandleBuilder(PhaseHandleBuilder):
    __slots__ = ()
    PhaseHandle = PhaseTPHandle
        
    
class PhaseFunctorBuilder:
    __slots__ = ('var', 's', 'l', 'g', 'build_functors')
    
    def __init__(self, var, s, l, g):
        self.var = var
        self.s = s
        self.l = l
        self.g = g
        
    def __call__(self, sdata, ldata, gdata):
        setfield = object.__setattr__
        phase_handle = self.PhaseHandle.blank(self.var)
        phases = ('s', 'g', 'l')
        builders = (self.s, self.g, self.l)
        phases_data = (sdata, gdata, ldata)
        for phase, builder, data in zip(phases, builders, phases_data):
            if builder:
                functor = builder.from_args(data)
                setfield(phase_handle, phase, functor)
        return phase_handle


class PhaseTFunctorBuilder(PhaseFunctorBuilder):
    __slots__ = ()
    PhaseHandle = PhaseTHandle
    
        
class PhaseTPFunctorBuilder(PhaseFunctorBuilder):
    __slots__ = ()
    PhaseHandle = PhaseTPHandle
        
        