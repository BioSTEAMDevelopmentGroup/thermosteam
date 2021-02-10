# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from warnings import warn

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

    __slots__ = ('data', 'safe_to_replace')

    def mark_safe_to_replace(self, objs):
        self.safe_to_replace.update(objs)

    def unmark_safe_to_replace(self, objs):
        safe_to_replace = self.safe_to_replace
        data = self.data
        for obj in objs:
            safe_to_replace.discard(obj)
            ID = obj._ID
            if ID in data: 
                other = data[ID]
                if obj is not other:
                    *root, n = ID.split('_')
                    if n.isdigit(): 
                        root = '_'.join(root)
                        ID_new = ID + '_2'
                    else:
                        root = ID
                        n = 2
                    ID_new = self._suggest_non_conflicting_ID(root, n)
                    warning = RuntimeWarning(
                        f"{ID} already exists in registry; "
                        f"{type(obj).__name__} has been replaced"
                    )
                    warn(warning, stacklevel=getattr(obj, '_stacklevel', 5) - 1)
                    data[ID_new] = other
                    other._ID = ID_new
                    data[ID] = obj
                    obj._ID = ID
            else:
                data[ID] = obj
        
    def __init__(self):
        self.data = {}
        self.safe_to_replace = set()

    def search(self, ID):
        return self.data.get(ID)
    
    def __getattr__(self, ID):
        try:
            return self.data[ID]
        except KeyError:
            raise AttributeError(f'{repr(ID)} not in registry')
    __getitem__ = __getattr__
    
    def register_safely(self, ID, obj):
        """Register object safely, with checks and due warnings."""
        check_valid_ID(ID)
        data = self.data
        ID_old = getattr(obj, '_ID', None)
        if ID_old and data.get(ID_old) is obj: del data[ID_old]
        if ID in data:
            other = data[ID]
            if obj is not other and other not in self.safe_to_replace:
                if ID_old:
                    warn(RuntimeWarning(f"upon renaming, {obj} replaced {other} in registry"), 4)
                else:
                    warning = RuntimeWarning(
                        f"{ID} already exists in registry; {type(other).__name__} "
                        f"object has been replaced"
                    )
                    warn(warning, stacklevel=getattr(obj, '_stacklevel', 5) - 1)
        data[ID] = obj
        obj._ID = ID
        
    def register(self, ID, obj):
        """Register object without warnings or checks."""
        data = self.data
        ID_old = getattr(obj, '_ID', None)
        if ID_old in data: del data[ID_old]
        data[ID] = obj
        obj._ID = ID
        
    def clear(self):
        self.data.clear()
        self.safe_to_replace.clear()
    
    def discard(self, obj):
        data = self.data
        if hasattr(obj, '_ID'):
            ID = obj._ID
            if ID in data and data[ID] is obj: del data[ID]
        elif isinstance(obj, str):
            if ID in data: del data[ID]
    
    def get_IDs(self):
        return set(self.data)
    
    def to_set(self):
        return set(self.data.values())
    
    def __contains__(self, obj):
        data = self.data
        if hasattr(obj, '_ID'):
            ID = obj._ID
            return ID in data and data[ID] is obj
        else:
            return obj in data
    
    def __iter__(self):
        return iter(self.data.values())
    
    def __repr__(self):
        if self.data:
            return 'Register:\n ' + '\n '.join([repr(i) for i in self.data.values()])
        else:
            return 'Register: (Empty)'

