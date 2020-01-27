# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:59:17 2019

@author: yoelr
"""
__all__ = ('repr_IDs_data', 'repr_kwargs', 'repr_kwarg', 'repr_couples')
    
def repr_IDs_data(IDs, data, dlim=", ", start=None):
    return (start or dlim) + dlim.join([f"{ID}={i:.4g}" for ID, i in zip(IDs, data) if i])

def repr_kwargs(kwargs, dlim=", ", start=None):
    if kwargs:
        start = dlim if start is None else ""
        return start + dlim.join([f"{key}={value}" for key, value in kwargs.items()])
    else:
        return ""

def repr_args(args, dlim=", ", start=None):
    if args:
        return (start or dlim) + dlim.join([repr(value) for value in args])
    else:
        return ""

def repr_kwarg(name, value, spec=None, dlim=", "):
    if value:
        if spec: value = format(value, spec)
        else: value = repr(value)
        return dlim + f"{name}={value}"
    else:
        return ""

def repr_couples(dlim, IDs, data):
    return dlim.join([f"('{ID}', {i:.4g})" for ID, i in zip(IDs, data) if i])


