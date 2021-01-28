# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""

__all__ = ('Registry', 'is_valid_ID', 'check_valid_ID')


def is_valid_ID(ID):
    if not isinstance(ID, str): return False
    if not ID[0].isalpha(): return False
    if not all([word.isalnum() for word in ID.split('_') if word]): return False
    return True

def check_valid_ID(ID):
    if not isinstance(ID, str):
        raise RuntimeError(f"ID must be a string, not a '{type(ID).__name__}' object")
    if not ID[0].isalpha():
        raise RuntimeError("ID must start with a letter")
    if not all([word.isalnum() for word in ID.split('_') if word]):
        raise RuntimeError(
            'ID may only contain letters, numbers, and/or underscores; '
            'no special characters or spaces'
        )

class Registry: # pragma: no cover
    
    def search(self, ID):
        return self.__dict__.get(ID)
    
    def __getitem__(self, ID):
        return self.__dict__[ID]
    
    def __setitem__(self, ID, obj):
        """Register object."""
        check_valid_ID(ID)
        self.__dict__[ID] = obj
    
    def __setattr__(self, ID, obj):
        self[ID] = obj
    
    def clear(self):
        self.__dict__.clear()
    
    def discard(self, ID):
        dct = self.__dict__
        try: dct[ID].disconnect()
        except AttributeError: pass
        del dct[ID]
    
    def get_IDs(self):
        return set(self.__dict__)
    
    def to_set(self):
        return set(self.__dict__.values())
    
    def __contains__(self, ID):
        return ID in self.__dict__
    
    def __iter__(self):
        return iter(self.__dict__.values())
    
    def __repr__(self):
        if self.__dict__:
            return 'Register:\n ' + '\n '.join([repr(i) for i in self.__dict__.values()])
        else:
            return 'Register: (Empty)'

