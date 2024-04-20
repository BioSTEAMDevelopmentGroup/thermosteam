# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
This module includes arbitrary classes and functions.

"""
from __future__ import annotations
import thermosteam as tmo
from math import ceil
import numpy as np
from collections.abc import Iterable
from math import floor, log10
from inspect import signature
from types import FunctionType

__all__ = (
    'extended_signature',
    'factor', 'checkbounds', 'strtuple',
    'format_title', 'format_unit_name',
    'remove_undefined_chemicals',
    'default_chemical_dict', 'subgroup',
    'repr_subgroups', 'repr_items',
    'list_available_names',
    'dictionaries2array',
    'flattened',
    'fill_like', 
    'getfields', 
    'setfields', 
    'copy_maybe', 
    'get_instance',
    'roundsigfigs',
    'array_roundsigfigs',
    'docround',
)

# %% Function signature

def extended_signature(f, g):
    if hasattr(f, '__wrapped__'): f = f.__wrapped__
    sigf = signature(f)
    paramsf = [*sigf.parameters.values()][1:-1]
    sigg = signature(g)
    paramsg = [*sigg.parameters.values()][1:]
    h = FunctionType(f.__code__, f.__globals__, name=f.__name__,
                     argdefs=f.__defaults__, closure=f.__closure__)
    h.__kwdefaults__ = f.__kwdefaults__
    all_params = [*paramsf, *[i.replace(kind=3) for i in paramsg]]
    params = []
    names = set()
    for i in tuple(all_params):
        if i.name in names: continue
        names.add(i.name)
        params.append(i)
    h.__signature__ = sigf.replace(parameters=params)
    h.__annotations__ = f.__annotations__ | g.__annotations__
    h.__wrapped__ = f
    return h

# %% Helpful tools

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
            

# %% Number functions

def factor(base_units, new_units):
    if base_units == new_units: return 1
    else: return tmo._Q(1, base_units).to(new_units).magnitude

def checkbounds(x, bounds):
    return bounds[0] < x < bounds[1]
    
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

def docround(obj, n=4):
    """Function to round recursively."""
    if isinstance(obj, np.ndarray):
        return np.round(obj, n)
    elif isinstance(obj, Iterable):
        values = [docround(i, n) for i in obj]
        return obj.__class__(values)
    else:
        return round(obj, n)   

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

# %% Array formulation

def dictionaries2array(dictionaries):
    keys = []
    shape = ()
    for dct in dictionaries: 
        keys.extend(dct)
        for sample in dct.values():
            sample = np.asarray(sample)
            shape = sample.shape
            break
    keys = (*frozenset(keys),)
    index = {j: i for i, j in enumerate(keys)}
    array = np.zeros([*shape, len(dictionaries), len(keys)], dtype=float)
    for i, dct in enumerate(dictionaries):
        for key, value in dct.items():
            j = index[key]
            array[..., i, j] = value
    return array, keys


# %% String functions

def list_available_names(names):
    *names, last = list(names)
    names = ', '.join([repr(i) for i in names]) + ', and ' + repr(last)
    return names

def strtuple(iterable):
    """Return string of all items in the tuple""" 
    string = ''
    function = type(strtuple)
    for i in iterable:
        if isinstance(i , function):
            string += i.__name__ + ', '
        else:
            string += str(i) + ', '
    string = string.rstrip(', ')
    string = '(' + string + ')'
    return string
        
def format_title(line):
    line = line.replace('_', ' ')
    words = []
    word = ''
    last = ''
    for i in line:
        if i.isupper() and last.isalpha() and not last.isupper():
            words.append(word)
            word = i
        else:
            word += i
        last = i
    words.append(word)
    line = ''
    for word in words:
        N_letters = len(word)
        if N_letters > 1:
            line += word + ' '
        else:
            line += word
    line = line.strip(' ')
    first_word, *rest = line.split(' ')
    words = [first_word[0].capitalize() + first_word[1:]]
    for word in rest:
        if not all([(i.isupper() or not last.isalpha()) for i in word]):
            word = word.lower()
        words.append(word)
    return ' '.join(words)

def format_unit_name(name):
    return ''.join([i[0].capitalize() + i[1:] for i in name.split(' ')])
    
def subgroup(items, size=5):
    return [items[size*i: size*(i+1)] for i in range(int(ceil(len(items) / size)))]

def repr_subgroups(subgroups):
    return [', '.join([str(i) for i in j]) for j in subgroups]

def repr_items(start, items, subgroup_size=5, brackets=None):
    N_spaces = len(start)
    if brackets:
        left, right = brackets
        N_spaces += 1
    else:
        left = ''; right = ''
    subgroups = repr_subgroups(subgroup(items, subgroup_size))
    dlim = ",\n" + " " * N_spaces
    return start + left + dlim.join(subgroups) + right

# %% Chemical management

def remove_undefined_chemicals(data: dict, chemicals: tmo.Chemicals):
    for i in tuple(data):
        if i not in chemicals: del data[i]

def default_chemical_dict(dct, chemicals, g, l, s, n=None):
    if n is None: n = l
    for i in chemicals:
        ID = i.ID
        if ID not in dct:
            locked_state = i.locked_state
            if locked_state == 'g': dct[ID] = g
            elif locked_state == 'l': dct[ID] = l
            elif locked_state == 's': dct[ID] = s
            elif locked_state is None: dct[ID] = n
            else: raise RuntimeError(f"unknown locked state '{locked_state}' of chemical '{i}'")
