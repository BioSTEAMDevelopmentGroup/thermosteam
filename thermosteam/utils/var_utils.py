# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:37:22 2019

@author: yoelr
"""

__all__ = ('var_with_units', 'get_anyvar')

def var_with_units(var, units):
    name, *phase = var.split(".")
    units = units.get(name, "")
    units = units and ' [' + str(units) + ']'
    return f"{var}{units}"

def get_anyvar(models):
    attr = hasattr
    for i in models:
        if attr(i.evaluate, 'var'): return i.evaluate.var


