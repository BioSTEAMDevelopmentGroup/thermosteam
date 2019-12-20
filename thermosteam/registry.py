# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:47:33 2018

@author: Yoel Cortes-Pena
"""

__all__ = ('Registry', )


class Registry:
    
    def __setattr__(self, ID, obj):
        self[ID] = obj
    
    def __getitem__(self, ID):
        return self.__dict__[ID]
    
    def __setitem__(self, ID, obj):
        """Register object."""
        if ID:
            assert isinstance(ID, str), f"ID must be a string, not a '{type(ID).__name__}' object"
            assert ID[0].isalpha(), "ID must start with a letter"
            ID_words = ID.split('_')
            assert all(word.isalnum() for word in ID_words), (
                    'ID may only contain letters, numbers, and/or underscores; '
                    'no special characters or spaces')
            self.__dict__[ID] = obj
        obj._ID = ID
    
    def __repr__(self):
        if self:
            return f'Register:\n ' + '\n '.join([repr(i) for i in self.__dict__.values()])
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
    try: delattr(self.registry, self._ID)
    except: pass
    
def _register(self, ID):
    if ID == "":
        self._ID = ID = self._take_ticket()
        self.registry.__dict__[ID] = self
    else:
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