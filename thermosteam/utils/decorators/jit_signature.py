# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 17:14:27 2025

@author: yoelr
"""
from inspect import signature
from numba import njit
from numba.types import Tuple

__all__ = ('JitSignature',)

class JitSignature:
    __slots__ = ('types',)
    def __init__(self, **types):
        self.types = types
        
    def __call__(self, output):
        types = self.types
        outputs = output.split(', ')
        if len(outputs) == 1:
            output = types[output]
        else:
            output = Tuple([types[i] for i in outputs])
        return lambda f: njit(output(*[types[i] for i in signature(f).parameters]), cache=True)(f)
        