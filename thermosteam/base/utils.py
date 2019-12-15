# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:37:22 2019

@author: yoelr
"""
from .units_of_measure import units_of_measure

def copy_maybe(obj):
    return obj.copy() if hasattr(obj, 'copy') else obj

def var_with_units(var, units=units_of_measure):
    name, *phase = var.split(".")
    units = units.get(name, "")
    units = units and ' [' + str(units) + ']'
    return f"{var}{units}"

def any_isinstance(objs, cls):
    isa = isinstance
    for i in objs:
        if isa(i, cls): return True
    return False

def get_anyvar(models):
    attr = hasattr
    for i in models:
        if attr(i.evaluate, 'var'): return i.evaluate.var

def get_dct_values(dct, params):
    return [dct[key] for key in params]

def get_obj_values(obj, params):
    attr = getattr
    return [attr(obj, key) for key in params]
