# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import thermosteam as tmo

__all__ = ('thermo_user',)

def thermo_user(cls):
    cls._load_thermo = _load_thermo
    cls.thermo = thermo
    cls.chemicals = chemicals
    cls.mixture = mixture
    return cls
    
def _load_thermo(self, thermo):
    self._thermo = thermo = tmo.settings.get_default_thermo(thermo)
    return thermo

@property
def thermo(self):
    return self._thermo
@property
def chemicals(self):
    return self._thermo.chemicals
@property
def mixture(self):
    return self._thermo.mixture