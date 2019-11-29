# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:55:06 2019

@author: yoelr
"""
__all__ = ('UndefinedChemical',)

class UndefinedChemical(LookupError):
    """LookupError regarding undefined compounds."""
    def __init__(self, ID): super().__init__(f"'{ID}'")