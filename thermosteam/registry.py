# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:47:33 2018

@author: Yoel Cortes-Pena
"""

__all__ = ('Registry', )

setitem = dict.__setitem__

class Registry(dict):
    
    def __setitem__(self, key, value):
        """Register object."""
        if key:
            ID_words = key.split('_')
            assert all(word.isalnum() for word in ID_words), (
                    'ID may only contain letters, numbers, and/or underscores; '
                    'no special characters or spaces')
            setitem(self, key, value)
        value._ID = key
    
    def __repr__(self):
        if self:
            return f'Register:\n ' + '\n '.join([repr(i) for i in self.values()])
        else:
            return f'Register: (Empty)'


# %% Register methods

registries = {}

def registered(ticket_name):
    return lambda cls: _registered(cls, ticket_name)

def _registered(cls, ticket_name):
    cls.registry = registry = Registry()
    cls.ticket_name = ticket_name
    cls.ticket_number = 0
    cls._take_ticket = _take_ticket
    cls._unregister = _unregister
    cls._register = _register
    cls.ID = ID
    cls.__repr__ = __repr__
    cls.__str__ = __str__
    registries[cls.__name__.lower()] = registry
    return cls

@classmethod
def _take_ticket(cls):
    cls.ticket_number += 1
    return cls.ticket_name + str(cls.ticket_number)

def _unregister(self):
    try: del self.registry[self._ID]
    except: pass
    
def _register(self, ID):
    if ID == "": ID = self._take_ticket()
    self.registry[ID] = self
    
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