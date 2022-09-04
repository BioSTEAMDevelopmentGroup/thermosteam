# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from .functor import functor_lookalike
from ..utils import read_only, cucumber

__all__ = ('PhaseHandle',
           'MockPhaseHandle',
           'MockPhaseTHandle',
           'MockPhaseTPHandle',
           'PhaseTHandle', 
           'PhaseTPHandle',
           'PhaseFunctorBuilder',
           'PhaseTFunctorBuilder',
           'PhaseTPFunctorBuilder')


# %% Abstract class    

@cucumber
@read_only
@functor_lookalike
class PhaseHandle:
    __slots__ = ('Tc', 'var', 's', 'l', 'g')
    
    def __init__(self, var, s, l, g, Tc=None):
        setattr = object.__setattr__
        setattr(self, 'var', var)
        setattr(self, 's', s)
        setattr(self, 'l', l)
        setattr(self, 'g', g)
        setattr(self, 'Tc', Tc)
    
    @property
    def S(self): return self.s
    @property
    def L(self): return self.l
    
    def __iter__(self):
        return iter((('s', self.s), ('l', self.l), ('g', self.g)))
    
    def __bool__(self):
        return any((self.s, self.l, self.g)) 
    
    def copy(self):
        return self.__class__(
            self.var, self.s.copy(), self.l.copy(), self.g.copy(), self.Tc,
        )
    __copy__ = copy
    
    def show(self):
        print(self)
    _ipython_display_ = show

@cucumber
@read_only
@functor_lookalike
class MockPhaseHandle:
    __slots__ = ('var', 'model')
    
    def __init__(self, var, model):
        setattr = object.__setattr__
        setattr(self, 'var', var)
        setattr(self, 'model', model)
    
    @property
    def S(self): return self.model
    @property
    def L(self): return self.model
    @property
    def s(self): return self.model
    @property
    def l(self): return self.model
    @property
    def g(self): return self.model
    
    def __iter__(self):
        return iter((('s', self.model), ('l', self.model), ('g', self.model)))
    
    def __bool__(self):
        return bool(self.model)
    
    def copy(self):
        return self.__class__(
            self.var, self.model.copy(), 
        )
    __copy__ = copy
    
    def show(self):
        print(self)
    _ipython_display_ = show
    
    
# %% Pure component

class PhaseTHandle(PhaseHandle):
    __slots__ = ()
    force_gas_critical_phase = False
    
    def __call__(self, phase, T, P=None):
        if self.force_gas_critical_phase and T > self.Tc: phase = 'g'
        return getattr(self, phase)(T)
    
    
class PhaseTPHandle(PhaseHandle):
    __slots__ = ()
    force_gas_critical_phase = False
    
    def __call__(self, phase, T, P=None):
        if self.force_gas_critical_phase and T > self.Tc: phase = 'g'
        return getattr(self, phase)(T, P)


class MockPhaseTHandle(MockPhaseHandle):
    __slots__ = ()
    
    def __call__(self, phase, T, P=None):
        return self.model(T)
    
    
class MockPhaseTPHandle(MockPhaseHandle):
    __slots__ = ()
    
    def __call__(self, phase, T, P):
        return self.model(T, P)

# %% Builders

class PhaseFunctorBuilder:
    __slots__ = ('var', 's', 'l', 'g', 'build_functors')
    
    def __init__(self, var, s, l, g):
        self.var = var
        self.s = s
        self.l = l
        self.g = g
        
    def __call__(self, sdata, ldata, gdata, Tc):
        phases = ('s', 'g', 'l')
        builders = (self.s, self.g, self.l)
        phases_data = (sdata, gdata, ldata)
        slg = {}
        for phase, builder, data in zip(phases, builders, phases_data):
            if builder:
                functor = builder.from_args(data)
                slg[phase] = functor
        return self.PhaseHandle(self.var, **slg, Tc=Tc)


class PhaseTFunctorBuilder(PhaseFunctorBuilder):
    __slots__ = ()
    PhaseHandle = PhaseTHandle
    
        
class PhaseTPFunctorBuilder(PhaseFunctorBuilder):
    __slots__ = ()
    PhaseHandle = PhaseTPHandle
    
    