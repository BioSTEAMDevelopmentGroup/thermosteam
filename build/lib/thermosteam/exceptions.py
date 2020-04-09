# -*- coding: utf-8 -*-
from .utils import colors

__all__ = ('UndefinedChemical',
           'UndefinedPhase',
           'UndefinedPhaseOrChemical',
           'DimensionError',
           'message_with_object_stamp',
           'try_method_with_object_stamp',
           'raise_error_with_object_stamp')

class UndefinedChemical(AttributeError):
    """AttributeError regarding undefined chemicals."""
    def __init__(self, ID): super().__init__(repr(ID))
    
class UndefinedPhase(AttributeError):
    """AttributeError regarding undefined phases."""
    def __init__(self, phase): super().__init__(repr(phase))

class UndefinedPhaseOrChemical(AttributeError):
    """AttributeError regarding undefined phases or chemicals."""
    def __init__(self, phase_or_ID): super().__init__(repr(phase_or_ID))

class DimensionError(ValueError):
    """ValueError regarding wrong dimensions."""

def message_with_object_stamp(object, msg):
    object_name = repr(object)
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
    except Exception as error:
        raise_error_with_object_stamp(object, error)