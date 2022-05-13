# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from collections.abc import Iterable
from math import floor, log10
import numpy as np

__all__ = (
    'flattened',
    'fill_like', 
    'getfields', 
    'setfields', 
    'copy_maybe', 
    'get_instance',
    'roundsigfigs',
    'array_roundsigfigs',
)

def flattened(lst):
    flat = []
    isa = isinstance
    for i in lst:
        if isa(i, Iterable):
            flat.extend(flattened(i))
        else:
            flat.append(i)
    return flat

def fill_like(A, B, fields):
    setfield = setattr
    getfield = getattr
    for i in fields: setfield(A, i, getfield(B, i))
    
def getfields(obj, fields, getfield=getattr):
    return [getfield(obj, i) for i in fields]

def setfields(obj, names, fields, setfield=setattr):
    for i,j in zip(names, fields): setfield(obj, i, j)

def copy_maybe(obj):
    return obj.copy() if hasattr(obj, 'copy') else obj

def get_instance(iterable, cls):
    """Return object that is an instance of given class."""
    isa = isinstance
    objs = [i for i in iterable if isa(i, cls)]
    N = len(objs)
    if N == 1:
        return objs[0]
    elif N == 0: # pragma: no cover
        raise ValueError('instance not found')
    else: # pragma: no cover
        raise ValueError('multiple instances found')
                
def roundsigfigs(x, sigfigs=2, index=1):
    if isinstance(x, Iterable):
        if isinstance(x, str): return x
        n = sigfigs - int(floor(log10(abs(x[index])))) - 1 if abs(x[index]) > 1e-12 else 0.
        try:
            value = np.round(x, n)
        except:
            return np.array(x, dtype=int)
        if (np.array(value, int) == value).all():
            return np.array(value, int)
        else:
            return value
    else:
        n = sigfigs - int(floor(log10(abs(x)))) - 1 if abs(x) > 1e-12 else 0.
        try:
            value = round(x, n)
        except:
            return int(x)
        if int(value) == value:
            return int(value)
        else:
            return value

def array_roundsigfigs(arr, sigfigs=2, index=1, inplace=False):
    if not inplace: arr = arr.copy()
    for idx, x in np.ndenumerate(arr):
        if isinstance(x, str): 
            try:
                x = float(x)
            except:
                arr[idx] = x
                continue
        n = sigfigs - int(floor(log10(abs(x)))) - 1 if abs(x) > 1e-12 else 0.
        try:
            value = round(x, n)
        except:
            arr[idx] = int(x)
            continue
        if int(value) == value:
            arr[idx] = int(value)
        else:
            arr[idx] = value
    return arr