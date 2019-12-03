# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:55:06 2019

@author: yoelr
"""
__all__ = ('UndefinedChemical',
           'UndefinedPhase',
           'assert_value')

class UndefinedChemical(AttributeError):
    """AttributeError regarding undefined compounds."""
    def __init__(self, ID): super().__init__(f"'{ID}'")
    
    
class UndefinedPhase(AttributeError):
    """AttributeError regarding undefined phases."""
    def __init__(self, phase): super().__init__(f"'{phase}'")


def assert_value(f, obj, name, type):
    assert f(obj, type), (f"{name} must be a '{type.__name__}' object, "
                          f"not a '{type(obj).__name__}'")
