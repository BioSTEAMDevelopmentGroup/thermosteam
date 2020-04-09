# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:08:07 2019

@author: yoelr
"""

__all__ = ('define_from',)

def _define_from(cls, other, names):
    getfield = getattr
    setfield = setattr
    for name in names: setfield(cls, name, getfield(other, name))
    return cls

def define_from(other, names):
    return lambda cls: _define_from(cls, other, names)