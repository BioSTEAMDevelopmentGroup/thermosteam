# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 08:43:40 2021

@author: yrc2
"""
from numba import njit

__all__ = ('ideal_coefficient',)

def ideal(cls):
    cls.f = ideal_coefficient
    cls.args = ()
    return cls

@property
def ideal_coefficient(self):
    return _ideal_coefficient

@njit(cache=True)
def _ideal_coefficient(z=None, T=None, P=None):
    return 1.