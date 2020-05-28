# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""

__all__ = ('var_with_units', 'get_anyvar')

def var_with_units(var, units):
    name, *phase = var.split(".")
    units = units.get(name, "")
    units = units and ' [' + str(units) + ']'
    return f"{var}{units}"

def get_anyvar(models):
    attr = hasattr
    for i in models:
        if attr(i.evaluate, 'var'): return i.evaluate.var


