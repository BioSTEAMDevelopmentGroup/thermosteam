# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:59:17 2019

@author: yoelr
"""
__all__ = ('repr_kwargs', 'repr_kwarg', 'repr_couples')
    
def repr_kwargs(IDs, data, dlim=", ", start=None):
    return (start or dlim) + dlim.join([f"{ID}={i:.4g}" for ID, i in zip(IDs, data) if i])

def repr_kwarg(name, value, spec=None, dlim=", "):
    if value:
        if spec: value = format(value, spec)
        else: value = repr(value)
        return dlim + f"{name}={value}"
    else:
        return ""

def repr_couples(dlim, IDs, data):
    return dlim.join([f"('{ID}', {i:.4g})" for ID, i in zip(IDs, data) if i])