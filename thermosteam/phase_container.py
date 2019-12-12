# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:36:37 2019

@author: yoelr
"""
from ctypes import create_unicode_buffer

__all__ = ('phase_container',)

isa = isinstance

def phase_container(phase):
     return create_unicode_buffer(phase) if isa(phase, str) else phase
 
def new_phase_container(phase):
    return create_unicode_buffer(phase if isa(phase, str) else phase[0])
