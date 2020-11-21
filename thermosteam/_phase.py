# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from .exceptions import UndefinedPhase

__all__ = ('Phase', 'LockedPhase', 'NoPhase', 'PhaseIndexer',
           'phase_tuple')

isa = isinstance
new = object.__new__
setfield = object.__setattr__
valid_phases = {'s', 'l', 'g', 'S', 'L', 'G'}

def check_phase(phase):
    if phase not in valid_phases:
        raise RuntimeError(
            f"invalid phase {repr(phase)} encountered; valid phases are "
            "'s', 'l', 'g', 'S', 'L', and 'G'"
        )  

def phase_tuple(phases):
    phases = set(phases)
    for i in phases: check_phase(i)
    return tuple(sorted(phases))

class PhaseIndexer:
    __slots__ = ('_index',)
    _index_cache = {}
    
    def __new__(cls, phases):
        self = new(cls)
        phases = frozenset(phases)
        cache = self._index_cache
        if phases in cache:
            self._index = cache[phases]
        else:
            self._index = cache[phases] = index = {j:i for i,j in enumerate(sorted(phases))}
            index[...] = slice(None) 
        return self
    
    def __call__(self, phase):
        try:
            return self._index[phase]
        except:
            raise UndefinedPhase(phase)        
    
    def __repr__(self):
        phases = list(self._index)
        return f"{type(self).__name__}({phases})"

class Phase:
    __slots__ = ('_phase',)
    
    @classmethod
    def convert(cls, phase):
        return phase if isa(phase, cls) else cls(phase)
    
    def __new__(cls, phase):
        self = new(cls)
        self._phase = phase
        return self
    
    def __reduce__(self):
        return Phase, (self.phase,)
    
    @property
    def phase(self):
        return self._phase
    @phase.setter
    def phase(self, phase):
        
        self._phase = phase
    
    def copy(self):
        return self.__class__(self.phase)
    __copy__ = copy
    
    def __repr__(self):
        return f"{type(self).__name__}({repr(self.phase)})"


class LockedPhase(Phase):
    __slots__ = ()
    _cache = {}
    
    def __new__(cls, phase):
        cache = cls._cache
        if phase in cache:
            self = cache[phase]
        else:
            cache[phase] = self = new(cls)
            setfield(self, '_phase', phase)
        return self
    
    def __reduce__(self):
        return Phase, (self.phase,)
    
    def __setattr__(self, name, value):
        if value != self.phase:
            raise AttributeError('phase is locked')
        
NoPhase = LockedPhase(None)
