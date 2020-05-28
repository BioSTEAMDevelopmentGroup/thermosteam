# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from .utils import colors
import flexsolve as flx
from flexsolve.exceptions import *

__all__ = (*flx.exceptions.__all__,
           'UndefinedChemical',
           'UndefinedPhase',
           'DimensionError',
           'DomainError',
           'InvalidMethod',
           'message_with_object_stamp',
           'try_method_with_object_stamp',
           'raise_error_with_object_stamp')

class UndefinedChemical(AttributeError):
    """AttributeError regarding undefined chemicals."""
    def __init__(self, ID): super().__init__(repr(ID))
    
class UndefinedPhase(AttributeError):
    """AttributeError regarding undefined phases."""
    def __init__(self, phase): super().__init__(repr(phase))

class DimensionError(ValueError):
    """ValueError regarding wrong dimensions."""

class DomainError(ValueError):
    """ValueError regarding an attempt to evaluate a model out of its domain."""

class InvalidMethod(ValueError):
    """ValueError regarding an attempt to evaluate an invalid method."""
    def __init__(self, method):
        super().__init__(self, repr(method))

def message_with_object_stamp(object, msg):
    object_name = str(repr(object))
    if object_name in msg:
        return msg
    else:
        return colors.violet(object_name) + ' ' + msg

def raise_error_with_object_stamp(object, error):
    msg, *args = error.args
    error.args = (message_with_object_stamp(object, msg), *args)
    raise error

def try_method_with_object_stamp(object, method, args=()):
    try:
        return method(*args)
    except KeyError as error:
        raise error
    except Exception as error:
        raise_error_with_object_stamp(object, error)