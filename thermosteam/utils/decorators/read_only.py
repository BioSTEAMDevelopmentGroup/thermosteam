# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""

__all__ = ('read_only',)

def deny(self, *args, **kwargs):
    raise TypeError(f"'{type(self).__name__}' object is read-only")

def read_only(cls=None, methods=()):
    if not cls and methods:
        return lambda cls: read_only(cls, methods)
    else:
        for i in methods: setattr(cls, i, deny)
        cls.__delattr__ = deny
        cls.__setattr__ = deny
        return cls
 