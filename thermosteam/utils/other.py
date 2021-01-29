# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
__all__ = (
    'fill_like', 
    'getfields', 
    'setfields', 
    'copy_maybe', 
    'get_instance'
)

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