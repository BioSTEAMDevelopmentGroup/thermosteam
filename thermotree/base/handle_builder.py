# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:00:55 2019

@author: yoelr
"""
from .thermo_model_handle import ThermoModelHandle, RegisteredHandles
from .utils import any_isinstance

__all__ = ('HandleBuilder',)

class HandleBuilder:
    __slots__ = ('function', 'params')
    
    def __init__(self, function):
        self.function = function
    
    def __call__(self, var, data):
        handle = ThermoModelHandle()
        self.function(handle, *data)
        models = handle.models
        for Handler in RegisteredHandles:    
            if any_isinstance(models, Handler._Model): break
        handle.__class__ = Handler
        handle.var = var
        return handle
        
    def __repr__(self):
        return f"<[{type(self).__name__}] {self.function.__name__}(data, var='')>"