# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:59:17 2019

@author: yoelr
"""
import pickle

__all__ = ('save', 'load', 'read_only', 'fill_like', 'CachedValue')

def fill_like(A, B, fields):
    setfield = setattr
    getfield = getattr
    for i in fields: setfield(A, i, getfield(B, i))

def deny_deletion(self, ID):
    raise AttributeError(f"'{type(self).__name__} object is read-only")
    
def deny_mutation(self, ID, name):
    raise AttributeError(f"'{type(self).__name__} object is read-only")

def read_only(cls):
    cls.__delattr__ = deny_deletion
    cls.__setattr__ = deny_mutation
    return cls
    
def save(object, file):
    with open(file, 'wb') as f: pickle.dump(object, f)
    
def load(file):
    with open(file, "rb") as f: return pickle.load(f)
    
class CachedValue:
    __slots__ = ('key', 'value')
    
    def __init__(self, key, value):
        self.key = key
        self.value = value
    
    def copy(self):
        return self.__class__(self.key, self.value)
    
    def __repr__(self):
        return f"{type(self).__name__}(key={self.key}, value={self.value})"