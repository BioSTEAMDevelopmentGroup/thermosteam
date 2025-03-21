# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
__all__ = ('AbstractMethod', 'NotImplementedMethod',
           'AbstractClassMethod', 'NotImplementedClassMethod')

class AbstractMethodType:
    __slots__ = ()
    
    @property
    def __name__(self): return "AbstractMethod"
    def __new__(self): return AbstractMethod
    def __call__(self): return NotImplemented
    def __bool__(self): return False
    def __repr__(self): return "AbstractMethod"
    
class AbstractClassMethodType:
    
    @property
    def __name__(self): return "AbstractClassMethod"
    def __new__(self): return AbstractClassMethod
    def __call__(self): return NotImplemented
    def __bool__(self): return False
    def __repr__(self): return "AbstractClassMethod"

AbstractMethod = NotImplementedMethod = object.__new__(AbstractMethodType)
AbstractClassMethod = NotImplementedClassMethod = object.__new__(AbstractClassMethodType)