# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:36:37 2019

@author: yoelr
"""
__all__ = ('Phase', 'LockedPhase', 'NoPhase')

isa = isinstance

class Phase:
    __slots__ = ('phase',)
    
    @classmethod
    def convert(cls, phase):
        return phase if isa(phase, cls) else cls(phase)
    
    def __init__(self, phase):
        self.phase = phase
        
    def copy(self):
        return self.__class__(self.phase)
    __copy__ = copy
    
    def __repr__(self):
        return f"{type(self).__name__}({repr(self.phase)})"


class LockedPhase(Phase):
    __slots__ = ()
    
    def __init__(self, phase):
        super().__setattr__('phase', phase)
        
    def __setattr__(self, name, value):
        raise AttributeError('phase is locked')
        
NoPhase = LockedPhase(None)