# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:47:33 2018

@author: Yoel Cortes-Pena
"""

__all__ = ('Register', )

class Register:
    
    __getitem__ = object.__getattribute__
    
    def search(self, key):
        if key in self.__dict__:
            return self[key]
    
    def __bool__(self):
        return bool(self.__dict__)  
    
    def __setitem__(self, key, value):
        """Register object."""
        if key:
            ID_words = key.split('_')
            assert all(word.isalnum() for word in ID_words), (
                    'ID may only contain letters, numbers, and/or underscores; '
                    'no special characters or spaces')
            value._ID = key
            self.__dict__[key] = value
    
    def __iter__(self):
        yield from self.__dict__.values()
    
    def __repr__(self):
        if self.__dict__:
            return f'Register:\n ' + '\n '.join([repr(i) for i in self])
        else:
            return f'Register: (Empty)'

