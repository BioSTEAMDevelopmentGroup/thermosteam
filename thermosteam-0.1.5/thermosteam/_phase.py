# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:36:37 2019

@author: yoelr
"""
__all__ = ('Phase', 'LockedPhase', 'NoPhase')

isa = isinstance
new = object.__new__
setfield = object.__setattr__

class Phase:
    __slots__ = ('phase',)
    
    @classmethod
    def convert(cls, phase):
        return phase if isa(phase, cls) else cls(phase)
    
    def __new__(cls, phase):
        self = new(cls)
        self.phase = phase
        return self
        
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
            setfield(self, 'phase', phase)
        return self
        
    def __setattr__(self, name, value):
        raise AttributeError('phase is locked')
        
NoPhase = LockedPhase(None)