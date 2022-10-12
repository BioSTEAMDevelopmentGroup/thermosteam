# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from .utils import colors

__all__ = ('UndefinedChemicalAlias',
           'UndefinedChemical',
           'UndefinedPhase',
           'DimensionError',
           'InfeasibleRegion',
           'InvalidMethod',
           'NoEquilibrium',
           'message_with_object_stamp',
           'try_method_with_object_stamp',
           'raise_error_with_object_stamp')

class InfeasibleRegion(RuntimeError):
    """Runtime error regarding infeasible processes."""
    def __init__(self, region, msg=None): 
        self.region = region
        if msg is None: msg = region + ' is infeasible'
        super().__init__(msg)

class UndefinedChemicalAlias(Exception):
    """Exception regarding undefined chemicals."""
    def __init__(self, ID): 
        self.ID = ID
        super().__init__(repr(ID))
    
UndefinedChemical = UndefinedChemicalAlias # Alias for backwards compatibility
    
class UndefinedPhase(Exception):
    """Exception regarding undefined phases."""
    def __init__(self, phase): super().__init__(repr(phase))

class NoEquilibrium(Exception):
    """Exception regarding an attempt to solve phase equilibrium when not applicable."""

class DimensionError(ValueError):
    """ValueError regarding wrong dimensions."""

class InvalidMethod(ValueError):
    """ValueError regarding an attempt to evaluate an invalid method."""
    def __init__(self, method):
        super().__init__(repr(method))
    
def message_with_object_stamp(object, msg):
    object_name = str(repr(object))
    if object_name in msg:
        return msg
    else:
        return colors.violet(object_name) + ' ' + msg

def raise_error_with_object_stamp(object, error):
    try: 
        msg, *args = error.args
        error.args = (message_with_object_stamp(object, msg), *args)
    except: pass
    raise error

def try_method_with_object_stamp(object, method, args=()):
    try:
        return method(*args)
    except KeyError as error:
        raise error
    except Exception as error:
        raise_error_with_object_stamp(object, error)