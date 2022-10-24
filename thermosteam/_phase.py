# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from .exceptions import UndefinedPhase

__all__ = ('Phase', 'LockedPhase', 'NoPhase', 'PhaseIndexer',
           'phase_tuple', 'valid_phases')

new = object.__new__
valid_phases = {
    's': 'solid',
    'l': 'liquid',
    'g': 'gas',
    'S': 'SOLID',
    'L': 'LIQUID',
}

def check_phase(phase):
    """
    Raise a RuntimeError if phase is invalid.
    
    Examples
    --------
    >>> check_phase('q')
    Traceback (most recent call last):
    RuntimeError: invalid phase 'q' encountered; valid phases are 
    's' (solid), 'l' (liquid), 'g' (gas), 'S' (SOLID), and 'L' (LIQUID)
    
    """
    if phase not in valid_phases:
        raise RuntimeError(
            f"invalid phase {repr(phase)} encountered; valid phases are "
            "'s' (solid), 'l' (liquid), 'g' (gas), 'S' (SOLID), and 'L' (LIQUID)"
        )  

def phase_tuple(phases):
    """
    Return a sorted set of phases.

    Parameters
    ----------
    phases : Iterable['s', 'l', 'g', 'S', or 'L']

    Examples
    --------
    >>> phase_tuple(['g', 's', 'l', 's'])
    ('g', 'l', 's')
    
    """
    phases = set(phases)
    for i in phases: check_phase(i)
    return tuple(sorted(phases))

class PhaseIndexer:
    """
    Create a PhaseIndexer object that can be used to find phase index of a 
    material array.
    
    Parameters
    ----------
    phases : Iterable[str]
    
    Examples
    --------
    Create a phase indexer for liquid and gas:
        
    >>> phase_indexer = PhaseIndexer(['l', 'g'])
    >>> phase_indexer # Note that phases are sorted
    PhaseIndexer(['g', 'l'])
    
    Find phase index:
    
    >>> phase_indexer('l')
    1
    
    An exception is raised when no index is available for a given phase:
    
    >>> phase_indexer('s')
    Traceback (most recent call last):
    UndefinedPhase: 'g'
    
    Phase indexers are unique for a given set of phases, regardless of order:
        
    >>> other = PhaseIndexer(['g', 'l'])
    >>> phase_indexer is other
    True
    
    """
    __slots__ = ('_index',)
    _index_cache = {}
    
    def __new__(cls, phases):
        phases = frozenset(phases)
        cache = cls._index_cache
        if phases in cache:
            self = cache[phases]
        else:
            cache[phases] = self = new(cls)
            self._index = index = {j:i for i,j in enumerate(sorted(phases))}
            index[...] = slice(None) 
        return self
    
    def __call__(self, phase):
        try:
            return self._index[phase]
        except:
            raise UndefinedPhase(phase)        
    
    @property
    def phases(self):
        return tuple(self._index)[:-1]
    
    def __reduce__(self):
        return PhaseIndexer, (self.phases,)
    
    def __repr__(self):
        return f"{type(self).__name__}({list(self.phases)})"

class Phase:
    __slots__ = ('_phase',)
    
    @classmethod
    def convert(cls, phase):
        return phase if isinstance(phase, cls) else cls(phase)
    
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
        check_phase(phase)
        self._phase = phase
    
    def copy(self):
        return Phase(self.phase)
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
            object.__setattr__(self, '_phase', phase)
        return self
    
    def __reduce__(self):
        return Phase, (self.phase,)
    
    def __setattr__(self, name, value):
        if value != self.phase:
            raise AttributeError('phase is locked')
        
NoPhase = LockedPhase(None)
