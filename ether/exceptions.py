# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:55:06 2019

@author: yoelr
"""
__all__ = ('UndefinedChemical', 'check_value')

def check_value(f, obj, name, type):
    if not f(obj, type):
        raise ValueError(f"{name} must be a '{type.__name__}' object, "
                         f"not a '{type(obj).__name__}'")

class UndefinedChemical(LookupError):
    """LookupError regarding undefined compounds."""
    def __init__(self, ID): super().__init__(f"'{ID}'")