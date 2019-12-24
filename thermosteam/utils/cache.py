# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 08:11:08 2019

@author: yoelr
"""
from .repr_utils import repr_args, repr_kwargs

__all__ = ('Cache',)

class Cache:
    __slots__ = ('loader', 'args', 'kwargs', 'obj')
    def __init__(self, loader, *args, **kwargs):
        self.loader = loader
        self.args = args
        self.kwargs = kwargs
        self.obj = None
        
    def retrieve(self):
        obj = self.obj
        if not obj:
            self.obj = obj = self.loader(*self.args, **self.kwargs)
        return obj
    
    def reload(self, *args, **kwargs):
        if args == self.args and kwargs == self.kwargs:
            obj = self.retrieve()
        else:
            self.args = args
            self.kwargs = kwargs
            self.obj = obj = self.loader(*self.args, **self.kwargs)
        return obj
    
    def __repr__(self):
        return f"{type(self).__name__}({repr(self.loader)}{repr_args(self.args)}{repr_kwargs(self.kwargs)})"