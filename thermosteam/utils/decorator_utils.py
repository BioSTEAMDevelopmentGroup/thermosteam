# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:59:17 2019

@author: yoelr
"""
import thermosteam as tmo

__all__ = ('read_only', 'chemicals_user')

def deny(self, *args, **kwargs):
    raise TypeError(f"'{type(self).__name__}' object is read-only")

def read_only(cls=None, methods=()):
    if not cls and methods:
        return lambda cls: read_only(cls, methods)
    else:
        for i in methods: setattr(cls, i, deny)
        cls.__delattr__ = deny
        cls.__setattr__ = deny
        return cls
    
# %% Decorator for chemicals users

def chemicals_user(cls):
    cls._load_chemicals = _load_chemicals
    cls.chemicals = chemicals
    return cls

@property
def chemicals(self):
    return self._chemicals

def _load_chemicals(self, chemicals):
    self._chemicals = tmo.settings.get_chemicals(chemicals)
    return chemicals