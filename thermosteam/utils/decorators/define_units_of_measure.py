# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import thermosteam as tmo
from typing import Optional

__all__ = ('define_units_of_measure',)

def define_units_of_measure(dct, cls=None):
    if cls is None:
        return lambda cls: define_units_of_measure(dct, cls)
    else:
        cls.define_property = define_property
        cls._units_of_measure = dct
        cls.get_property = get_property
        cls.set_property = set_property
    return cls
    
@classmethod
def define_property(cls, name, units, fget, fset=None):
    cls._units_of_measure[name] = tmo.units_of_measure.UnitsOfMeasure(units)
    if hasattr(cls, name): raise ValueError(f"property with name '{name}' already exists")
    setattr(cls, name, property(fget, fset))

def get_property(self, name: str, units: Optional[str]=None):
    """
    Return property in requested units.

    Parameters
    ----------
    name :
        Name of property.
    units : 
        Units of measure. Defaults to the property's original units of measure.

    """
    value = getattr(self, name)
    if units is None:
        return value
    else:
        units_dct = self._units_of_measure
        if name in units_dct:
            original_units = units_dct[name]
        else:
            raise ValueError(f"'{name}' is not a property")
        return original_units.convert(value, units)

def set_property(self, name: str, value: float, units: Optional[str]=None):
    """
    Set property in given units.

    Parameters
    ----------
    name : 
        Name of property.
    value : 
        New value of property.
    units : 
        Units of measure.

    """
    units_dct = self._units_of_measure
    if name in units_dct:
        if units is not None:
            original_units = units_dct[name]
            value = original_units.unconvert(value, units)
        setattr(self, name, value)
    else:
        raise ValueError(f"no property with name '{name}'")
