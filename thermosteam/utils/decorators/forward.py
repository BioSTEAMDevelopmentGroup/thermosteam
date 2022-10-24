# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""

__all__ = ('forward', 'forward_to_module')

def forward_to_module(f, module):
    """Define function `f` in given module."""
    setattr(module, f.__name__, f)
    return f

def forward(module):
    """Decorator to define function `f` in given module."""
    return lambda f: forward_to_module(f, module)