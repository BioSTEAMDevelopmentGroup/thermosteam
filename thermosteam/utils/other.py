# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
__all__ = ('fill_like', 'getfields', 'setfields', 'any_isinstance',
           'copy_maybe', 'get_dct_values', 'get_obj_values',
           'get_instance', 'define_from', 'Chaperone')

class Chaperone:
    __slots__ = ('item',)
    def __init__(self, item):
        self.item = item
    def get_item(self, item):
        return self.item
    def set_item(self, item):
        self.item = item
    def __rerp__(self):
        return f"{type(self).__name__}(item={self.item})"

def fill_like(A, B, fields):
    setfield = setattr
    getfield = getattr
    for i in fields: setfield(A, i, getfield(B, i))
    
def getfields(obj, fields, getfield=getattr):
    return [getfield(obj, i) for i in fields]

def setfields(obj, names, fields, setfield=setattr):
    for i,j in zip(names, fields): setfield(obj, i, j)

def any_isinstance(objs, cls):
    isa = isinstance
    for i in objs:
        if isa(i, cls): return True
    return False

def copy_maybe(obj):
    return obj.copy() if hasattr(obj, 'copy') else obj

def get_dct_values(dct, params):
    return [dct[key] for key in params]

def get_obj_values(obj, params):
    attr = getattr
    return [attr(obj, key) for key in params]

def _define_from(cls, other, names):
    getfield = getattr
    setfield = setattr
    for name in names: setfield(cls, name, getfield(other, name))
    return cls

def define_from(other, names):
    return lambda cls: _define_from(cls, other, names)

def get_instance(iterable, cls):
    """Return object that is an instance of given class."""
    isa = isinstance
    objs = [i for i in iterable if isa(i, cls)]
    N = len(objs)
    if N == 1:
        return objs[0]
    elif N == 0:
        raise ValueError('instance not found')
    else:
        raise ValueError('multiple instances')