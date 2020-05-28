# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import pickle
from . other_utils import getfields, setfields

__all__ = ('save', 'load', 'cucumber',)

def save(object, file):
    with open(file, 'wb') as f: pickle.dump(object, f)
    
def load(file):
    with open(file, "rb") as f: return pickle.load(f)
    
def get_state(obj):
    slots = obj._pickle_recipe if hasattr(obj, '_pickle_recipe') else obj.__slots__
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
    