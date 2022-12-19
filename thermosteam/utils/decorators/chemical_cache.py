# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:15:23 2022

@author: yrc2
"""
import thermosteam as tmo
from functools import cache

__all__ = ('chemical_cache',)

def chemical_cache(f):
    g = cache(f)
    def h(*args, cache=True, **kwargs):
        if cache:
            Chemical = tmo.Chemical
            caching = Chemical.cache
            Chemical.cache = True
            value = g(*args, **kwargs)
            Chemical.cache = caching
        else:
            value = f(*args, **kwargs)
        return value
    h.__name__ = f.__name__
    return h