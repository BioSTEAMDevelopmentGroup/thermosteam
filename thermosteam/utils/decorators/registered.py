# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from ..registry import Registry

__all__ = ('registered',)


def registered(ticket_name):
    return lambda cls: _registered(cls, ticket_name)

def _registered(cls, ticket_name):
    cls.registry = Registry()
    cls.ticket_name = ticket_name
    cls.ticket_number = 0
    cls.unregistered_ticket_number = 0
    cls._take_unregistered_ticket = _take_unregistered_ticket
    cls._take_ticket = _take_ticket
    cls._unregister = _unregister
    cls._register = _register
    cls.ID = ID
    cls.__repr__ = __repr__
    cls.__str__ = __str__
    return cls

@classmethod
def _take_ticket(cls):
    cls.ticket_number += 1
    return cls.ticket_name + str(cls.ticket_number)
@classmethod
def _take_unregistered_ticket(cls):
    if cls.unregistered_ticket_number > 100:
       cls.unregistered_ticket_number = 1 
    else:
        cls.unregistered_ticket_number += 1
    return cls.ticket_name + '.' + str(cls.unregistered_ticket_number)

def _unregister(self):
    try: delattr(self.registry, self._ID)
    except: pass
    
def _register(self, ID):
    if ID:
        self.registry[ID] = self
        self._ID = ID
    elif ID == "":
        self._ID = ID = self._take_ticket()
        self.registry.__dict__[ID] = self
    else:
        self._ID = self._take_unregistered_ticket()
        
@property
def ID(self):
    """Unique identification (str). If set as '', it will choose a default ID."""
    return self._ID

@ID.setter
def ID(self, ID):
    self._unregister()
    self._register(ID)
    
def __repr__(self):
    return f"<{type(self).__name__}: {self.ID}>"

def __str__(self):
    return self.ID
