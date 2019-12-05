# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:55:06 2019

@author: yoelr
"""
__all__ = ('UndefinedChemical',
           'UndefinedPhase')

class UndefinedChemical(IndexError):
    """IndexError regarding undefined compounds."""
    def __init__(self, ID): super().__init__(f"'{ID}'")
    
    
class UndefinedPhase(IndexError):
    """IndexError regarding undefined phases."""
    def __init__(self, phase): super().__init__(f"'{phase}'")


