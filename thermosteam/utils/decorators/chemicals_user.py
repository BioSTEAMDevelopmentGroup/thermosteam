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

__all__ = ('chemicals_user',)

def chemicals_user(cls):
    cls._load_chemicals = _load_chemicals
    cls.chemicals = chemicals
    return cls

@property
def chemicals(self):
    return self._chemicals

def _load_chemicals(self, chemicals):
    self._chemicals = chemicals = tmo.settings.get_default_chemicals(chemicals)
    return chemicals