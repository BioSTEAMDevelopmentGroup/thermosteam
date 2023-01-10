# -*- coding: utf-8 -*-
"""
"""
from .phase_handle import PhaseHandle
from .._thermal_condition import mock_thermal_condition

__all__ = (
    'WrappedDictionary',
    'MassFlowDict',
    'VolumetricFlowDict',
)

TP_V = (mock_thermal_condition, None) # Initial cache for molar volume

class WrappedDictionary: # Abstract class
    __slots__ = ('dct',)
    
    def __iter__(self):
        return self.dct.__iter__()
    
    def __len__(self):
        return self.dct.__len__()
    
    def __bool__(self):
        return bool(self.dct)
    
    def __delitem__(self, key):
        del self.dct[key]
    
    def __getitem__(self, key):
        return self.getter(key, self.dct[key])
    
    def __setitem__(self, key, value):
        self.dct[key] = self.setter(key, value)
        
    def keys(self):
        return self.dct.keys()
    
    def items(self):
        for i, j in self.dct.items():
            yield (i, self.getter(i, j))
            
    def values(self):
        for i, j in self.dct.items():
            yield self.getter(i, j)
    
    def clear(self):
        self.dct.clear()
        
    def copy(self):
        return {i: self.getter(i, j) for i, j in self.dct.items()}
    
    def get(self, key, default=None):
        dct = self.dct
        if key in dct:
            return self.getter(key, dct[key])
        else:
            return default
        
    def pop(self, key):
        return self.getter(key, self.dct.pop(key))
        
    def popitem(self):
        key, value = self.dct.popitem()
        return self.getter(key, value)
    
    def setdefault(self, key, default=None):
        if key not in self.dct: self[key] = default
        
    def from_keys(self, keys, value=None):
        return self.dct.from_keys(keys, value)
    
    def update(self, dct):
        for i, j in dct.items(): self[i] = j


class MassFlowDict(WrappedDictionary): # Wraps a dict of molar flows
    __slots__ = ('MW',)
    
    def __init__(self, dct, MW):
        self.dct = dct
        self.MW = MW
    
    def getter(self, index, value):
        return value * self.MW[index] # From mol to kg

    def setter(self, index, value):
        return value / self.MW[index] # From kg to mol


class VolumetricFlowDict(WrappedDictionary): # Wraps a dict of molar flows
    __slots__ = ('TP', 'V', 'phase', 'phase_container', 'cache')
    
    def __init__(self, dct, TP, V, phase, phase_container, cache):
        self.dct = dct
        self.TP = TP
        self.V = V
        self.phase = phase
        self.phase_container = phase_container
        self.cache = cache
    
    def getter(self, index, value):
        TP, V = self.cache.get(index, TP_V)
        if not TP.in_equilibrium(self.TP):
            phase = self.phase or self.phase_container.phase
            V = self.V[index]
            V = 1000. * (getattr(V, phase) if isinstance(V, PhaseHandle) else V)(*self.TP)
            self.cache[index] = (self.TP.copy(), V)
        return value * V # From mol to m3

    def setter(self, index, value):
        TP, V = self.cache.get(index, TP_V)
        if not TP.in_equilibrium(self.TP):
            phase = self.phase or self.phase_container.phase
            V = self.V[index]
            V = 1000. * (getattr(V, phase) if isinstance(V, PhaseHandle) else V)(*self.TP)
            self.cache[index] = (self.TP.copy(), V)
        return value / V # From m3 to mol

