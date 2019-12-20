# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:59:17 2019

@author: yoelr
"""
import pickle
from . other_utils import getfields, setfields

__all__ = ('save', 'load', 'cucumber',)

def save(object, file):
    with open(file, 'wb') as f: pickle.dump(object, f)
    
def load(file):
    with open(file, "rb") as f: return pickle.load(f)
    
def get_state(obj):
    slots = obj._pickle_recipe if hasattr(obj, '_recipe_') else obj.__slots__
    return (obj.__class__, slots, getfields(obj, slots))
    
def new_from_state(cls, slots, values):
    obj = object.__new__(cls)
    setfields(obj, slots, values, object.__setattr__)
    return obj

def __reduce__(self):
    return new_from_state, get_state(self)

def cucumber(cls):
    cls.__reduce__ = __reduce__
    return cls
    