# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the t_dependent_property module from the thermo library:
# https://github.com/CalebBell/thermo
# Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>
#
# This module is under a dual license:
# 1. The UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
# 
# 2. The MIT open-source license. See
# https://github.com/CalebBell/chemicals/blob/master/LICENSE.txt for details.
from thermo import TDependentProperty, VolumeSolid
TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = True

# Remove cache from call
TDependentProperty.__call__ = TDependentProperty.T_dependent_property

# Backwards compatibility with past thermosteam versions
def add_model(self, f=None, *args, top_priority=True, **kwargs):
    if f is None:
        return lambda f: self.add_method(f, *args, **kwargs)
    else:
        self.add_method(f, *args, **kwargs)

TDependentProperty.add_model = add_model
def has_method(self):
    return bool(self._method)

TDependentProperty.__bool__ = has_method

def __call__(self, T, P):
    return self.T_dependent_property(T)

VolumeSolid.__call__ = __call__

def copy(self):
    cls = type(self)
    copy = cls.__new__(cls)
    copy.__dict__.update(self.__dict__)
    return copy

TDependentProperty.copy = copy