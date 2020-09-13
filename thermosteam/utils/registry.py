# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""

__all__ = ('Registry',)

class Registry: # pragma: no cover
    
    def search(self, ID):
        return self.__dict__.get(ID)
    
    def __getitem__(self, ID):
        return self.__dict__[ID]
    
    def __setitem__(self, ID, obj):
        """Register object."""
        assert isinstance(ID, str), f"ID must be a string, not a '{type(ID).__name__}' object"
        assert ID[0].isalpha(), "ID must start with a letter"
        ID_words = ID.split('_')
        assert all([word.isalnum() for word in ID_words if word]), (
                'ID may only contain letters, numbers, and/or underscores; '
                'no special characters or spaces')
        self.__dict__[ID] = obj
    
    def __setattr__(self, ID, obj):
        self[ID] = obj
    
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

