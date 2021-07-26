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
from warnings import warn
TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = True

# Remove cache from call
TDependentProperty.__call__ = TDependentProperty.T_dependent_property

### Backwards compatibility with past thermosteam versions

# Allow for decorators
def add_model(self, f=None, *args, top_priority=True, **kwargs):
    if f is None:
        return lambda f: self.add_method(f, *args, **kwargs)
    else:
        self.add_method(f, *args, **kwargs)

TDependentProperty.add_model = add_model

# Missing method 
def has_method(self):
    return bool(self._method)

TDependentProperty.__bool__ = has_method

# Consistency with other volume phases
def __call__(self, T, P=None):
    return self.T_dependent_property(T)

VolumeSolid.__call__ = __call__

# Shallow copy
def copy(self):
    cls = type(self)
    copy = cls.__new__(cls)
    copy.__dict__.update(self.__dict__)
    return copy

TDependentProperty.copy = copy

# Handling methods

@TDependentProperty.method.setter
def method(self, method):
    if method is not None:
        method, *_ = method.split('(')
        method = method.upper().replace(' ', '_').replace('_AND_', '_').strip('_').replace('SOLID', 'S')
        if method not in self.all_methods and method != 'POLY_FIT':
            raise ValueError("Method '%s' is not available for this chemical; "
                             "available methods are %s" %(method, self.all_methods))
    self._method = method

def set_model_priority(self, model, priority=0):
    if priority == 0:
        warn("'set_model_priority' is deprecated; set the 'method' "
             "attribute to change model", DeprecationWarning)
        self.method = model
    else:
        raise RuntimeError(
            "'set_model_priority' is deprecated; cannot set model "
            "priority as models are not cycled anymore")
        
def move_up_model_priority(self, model, priority=0):
    if priority == 0:
        warn("'move_up_model_priority' is deprecated; set the 'method' "
             "attribute to change model", DeprecationWarning)
        self.method = model
    else:
        raise RuntimeError(
            "'move_up_model_priority' is deprecated; cannot set model "
            "priority as models are not cycled anymore")

TDependentProperty.method = method
TDependentProperty.set_model_priority = set_model_priority
TDependentProperty.move_up_model_priority = move_up_model_priority