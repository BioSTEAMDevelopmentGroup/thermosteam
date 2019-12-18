# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:59:17 2019

@author: yoelr
"""
__all__ = ('fill_like', 'getfields', 'setfields')

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
