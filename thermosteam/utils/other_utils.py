# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:59:17 2019

@author: yoelr
"""
__all__ = ('fill_like', 'getfields', 'setfields', 'any_isinstance',
           'copy_maybe', 'get_dct_values', 'get_obj_values')

def fill_like(A, B, fields):
    setfield = setattr
    getfield = getattr
    for i in fields: setfield(A, i, getfield(B, i))
    
def getfields(obj, fields):
    getfield = getattr
    return [getfield(obj, i) for i in fields]

def setfields(obj, names, fields):
    setfield = setattr
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