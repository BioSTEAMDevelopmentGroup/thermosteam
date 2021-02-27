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

__all__ = ('registered', 'unregistered')

def unregistered(cls):
    cls._register = _pretend_to_register
    return cls

def registered(ticket_name):
    return lambda cls: _registered(cls, ticket_name)

def _registered(cls, ticket_name):
    cls.registry = Registry()
    cls.ticket_name = ticket_name
    cls.ticket_numbers = {}
    cls.unregistered_ticket_number = 0
    cls._take_unregistered_ticket = _take_unregistered_ticket
    cls._take_ticket = _take_ticket
    cls._register = _register
    cls.ID = ID
    cls.__repr__ = __repr__
    cls.__str__ = __str__
    return cls

@classmethod
def _take_ticket(cls):
    ticket_numbers = cls.ticket_numbers
    ticket_name = cls.ticket_name 
    if ticket_name not in ticket_numbers:
        ticket_numbers[ticket_name] = n = 1
    else:
        ticket_numbers[ticket_name] = n = ticket_numbers[ticket_name] + 1
    return ticket_name + str(n)

@classmethod
def _take_unregistered_ticket(cls):
    if cls.unregistered_ticket_number > 100:
       cls.unregistered_ticket_number = 1 
    else:
        cls.unregistered_ticket_number += 1
    return cls.ticket_name + '.' + str(cls.unregistered_ticket_number)

def _register(self, ID):
    replace_ticket_number = isinstance(ID, int)
    if replace_ticket_number: self.__class__.ticket_number = ID
    if ID == "" or replace_ticket_number: 
        registry = self.registry
        data = registry.data
        while True:
            ID = self._take_ticket()
            if not ID in data: break
        registry.register(ID, self)
    elif ID:
        self.registry.register_safely(ID, self) 
    else:
        self._ID = self._take_unregistered_ticket()

def _pretend_to_register(self, ID):
    self._ID = ID

@property
def ID(self):
    """Unique identification (str). If set as '', it will choose a default ID."""
    return self._ID

@ID.setter
def ID(self, ID):
    self._register(ID)
    
def __repr__(self):
    return f"<{type(self).__name__}: {self.ID}>"

def __str__(self):
    return self.ID
