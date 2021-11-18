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

__all__ = ('units_of_measure',)

def units_of_measure(dct, cls=None):
    if cls is None:
        return lambda cls: units_of_measure(dct, cls)
    else:
        cls.define_property = define_property
        cls._units_of_measure = dct
        cls.get_property = get_property
        cls.set_property = set_property
    return cls
    
@classmethod
def define_property(cls, name, units, fget, fset=None):
    cls._units_of_measure[name] = tmo.units_of_measure.AbsoluteUnitsOfMeasure(units)
    if hasattr(cls, name): raise ValueError(f"property with name '{name}' already exists")
    setattr(cls, name, property(fget, fset))

def get_property(self, name, units):
    """
    Return property in requested units.

    Parameters
    ----------
    name : str
        Name of property.
    units : str
        Units of measure.

    """
    value = getattr(self, name)
    units_dct = self._units_of_measure
    if name in units_dct:
        original_units = units_dct[name]
    else:
        raise ValueError(f"'{name}' is not a property")
    return original_units.convert(value, units)

def set_property(self, name, value, units):
    """
    Set property in given units.

    Parameters
    ----------
    name : str
        Name of property.
    value : str
        New value of property.
    units : str
        Units of measure.

    """
    units_dct = self._units_of_measure
    if name in units_dct:
        original_units = units_dct[name]
        value = original_units.unconvert(value, units)
        setattr(self, name, value)
    else:
        raise ValueError(f"no property with name '{name}'")
