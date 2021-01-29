# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from .thermo_model_handle import TDependentModelHandle, TPDependentModelHandle

__all__ = ('HandleBuilder', 'TDependentHandleBuilder', 'TPDependentHandleBuilder')

class HandleBuilder:
    __slots__ = ('function',)

    def __new__(cls, var, build=None):
        if not build: return lambda build: cls(var, build)
        self = super().__new__(cls)
        self.build = build
        self.var = var
        return self

    def __call__(self, data):
        handle = self.Handle(self.var)
        self.build(handle, *data)
        return handle
        
    def __repr__(self):
        return f"<[{type(self).__name__}] {self.build.__name__}(data)>"
    
class TDependentHandleBuilder(HandleBuilder):
    Handle = TDependentModelHandle

class TPDependentHandleBuilder(HandleBuilder):
    Handle = TPDependentModelHandle