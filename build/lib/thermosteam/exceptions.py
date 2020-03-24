# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:55:06 2019

@author: yoelr
"""
__all__ = ('UndefinedChemical',
           'UndefinedPhase',
           'UndefinedPhaseOrChemical',
           'DimensionError',
           'AutodocError',
           'DeprecatedAttribute')

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

class AutodocError(RuntimeError):
    """RuntimeError regarding automatic documentation."""
    
class DeprecatedAttribute(AttributeError):
    def __init__(self, old, new):
        super().__init__(f"'{old}' is deprecated; please use '{new}' instead")
        
        