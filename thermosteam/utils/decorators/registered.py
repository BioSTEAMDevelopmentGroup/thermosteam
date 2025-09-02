# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from ..registry import Registry

__all__ = ('registered', 'unregistered', 'registered_franchise')

def unregistered(cls):
    cls._register = _pretend_to_register
    cls.register_alias = _pretend_to_register
    return cls

def registered(ticket_name):
    return lambda cls: _registered(cls, ticket_name)

def _registered(cls, ticket_name, autonumber=True):
    cls.registry = Registry()
    cls.ticket_name = ticket_name
    cls.ticket_numbers = {}
    cls.autonumber = autonumber
    cls.unregistered_ticket_number = 0
    cls._take_ticket = _take_ticket
    cls._register = _register
    cls.register_alias = register_alias
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

def _register(self, ID):
    registry = self.registry
    data = registry.data
    if ID is None:
        if hasattr(self, '_ID') and data.get(ID_old:=self._ID) is self: del data[ID_old]
        self._ID = ""
    elif isinstance(ID, str):
        if '.' in ID:
            if hasattr(self, '_ID') and data.get(ID_old:=self._ID) is self: del data[ID_old]
            self._ID = ID.lstrip('.')
        elif ID:
            registry.register_safely(ID, self) 
        elif self.autonumber:
            ID = self._take_ticket()
            while ID in data: ID = self._take_ticket()
            registry.register(ID, self)
        else:
            registry.register_safely(self.ticket_name, self) 
    elif isinstance(ID, int):
        self.ticket_numbers[self.ticket_name] = ID
        ID = self._take_ticket()
        while ID in data: ID = self._take_ticket()
        registry.register(ID, self)
    elif hasattr(ID, '__iter__'):
        ID, *aliases = ID
        self._register(ID)
        for i in aliases: self.register_alias(i)
    else:
        raise ValueError('invalid ID {ID!r}; ID must be a string, integer, or an interable of these')

def _pretend_to_register(self, ID):
    self._ID = ID

def ID(self):
    """Unique identification (str). If set as '', it will choose a default ID."""
    return self._ID
 
def register_alias(self, alias, override=None, safe=True):
    if safe:
        self.registry.register_alias_safely(alias, self, override) 
    else:
        self.registry.register_alias(alias, self, override) 
    
def __repr__(self):
    if self.ID:
        return f'<{type(self).__name__}: {self.ID}>'
    else:
        return f'<{type(self).__name__}>'

def __str__(self):
    return self.ID or '-'

def registered_franchise(parent):
    return lambda cls: _registered_franchise(cls, parent)

def _registered_franchise(cls, parent):
    cls._franchise_parent = parent
    cls._register = _franchise_register
    cls.ID = ID_franchise
    cls.__str__ = __str__
    cls.__repr__ = __repr__
    return cls

def _franchise_register(self, ID):
    parent = self._franchise_parent
    registry = parent.registry
    data = registry.data
    if ID is None:
        if hasattr(self, '_ID') and data.get(ID_old:=self._ID) is self: del data[ID_old]
        self._ID = ""
    else:
        replace_ticket_number = isinstance(ID, int)
        if replace_ticket_number: 
            parent.ticket_numbers[self.ticket_name] = ID
            ID = None
        if ID:
            if '.' in ID:
                if hasattr(self, '_ID') and data.get(ID_old:=self._ID) is self: del data[ID_old]
                self._ID = ID.lstrip('.')
            else:
                registry.register_safely(ID, self) 
        else:
            if parent.autonumber or replace_ticket_number:
                ID = parent._take_ticket()
                while ID in data: ID = parent._take_ticket()
                registry.register(ID, self)
            else:
                registry.register_safely(parent.ticket_name, self) 
                
ID_franchise = property(ID, _franchise_register)    
ID = property(ID, _register)