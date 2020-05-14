# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:59:17 2019

@author: yoelr
"""
import thermosteam as tmo

__all__ = ('chemicals_user',)

def chemicals_user(cls):
    cls._load_chemicals = _load_chemicals
    cls.chemicals = chemicals
    return cls

@property
def chemicals(self):
    return self._chemicals

def _load_chemicals(self, chemicals):
    self._chemicals = chemicals = tmo.settings.get_default_chemicals(chemicals)
    return chemicals