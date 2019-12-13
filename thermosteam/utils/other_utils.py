# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:59:17 2019

@author: yoelr
"""
__all__ = ('fill_like',)

def fill_like(A, B, fields):
    setfield = setattr
    getfield = getattr
    for i in fields: setfield(A, i, getfield(B, i))
