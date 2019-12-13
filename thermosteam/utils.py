# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:59:17 2019

@author: yoelr
"""
import pickle

__all__ = ('save', 'load', 'read_only', 'fill_like',
           'Cache', 'repr_kwargs')

def fill_like(A, B, fields):
    setfield = setattr
    getfield = getattr
    for i in fields: setfield(A, i, getfield(B, i))

def deny(self, *args, **kwargs):
    raise TypeError(f"'{type(self).__name__}' object is read-only")

def read_only(cls=None, methods=()):
    if not cls and methods:
        return lambda cls: read_only(cls, methods)
    else:
        for i in methods: setattr(cls, i, deny)
        cls.__delattr__ = deny
        cls.__setattr__ = deny
        return cls
    
def save(object, file):
    with open(file, 'wb') as f: pickle.dump(object, f)
    
def load(file):
    with open(file, "rb") as f: return pickle.load(f)
    
class Cache:
    __slots__ = ('loader', 'value', 'args')
    def __init__(self, loader, args=()):
        self.loader = loader
        self.value = None
        self.args = args
    
    def __call__(self):
        value = self.value
        if value is None:
            self.value = value = self.loader(*self.args)
        return value
            
def repr_kwargs(IDs, data, dlim=", ", start=None):
    return (start or dlim) + dlim.join([f"{ID}={i:.4g}" for ID, i in zip(IDs, data) if i])

def repr_kwarg(name, value, spec=None, dlim=", "):
    if value:
        if spec: value = format(value, spec)
        else: value = repr(value)
        return dlim + f"{name}={value}"
    else:
        return ""

def repr_couples(dlim, IDs, data):
    return dlim.join([f"('{ID}', {i:.4g})" for ID, i in zip(IDs, data) if i])