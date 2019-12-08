# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:36:37 2019

@author: yoelr
"""
from ctypes import create_unicode_buffer

__all__ = ('PhaseContainer', 'phase_container')

isa = isinstance

def phase_container(phase):
     return phase if isa(phase, PhaseContainer) else create_unicode_buffer(phase)
 
PhaseContainer = type(create_unicode_buffer('l'))