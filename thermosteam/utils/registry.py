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

    __slots__ = ('data', 'safe_to_replace',)

    def untrack(self, objs):
        """
        Mark objects safe to replace, so no warnings are issued if any are 
        replaced.
        
        """
        self.safe_to_replace.update(objs)

    def track(self, objs):
        """
        Reregister objects if they are not already in the registry and mark 
        objects as unsafe to replace, so that warnings can be issued if any are 
        replaced.
        
        """
        safe_to_replace = self.safe_to_replace
        data = self.data
        for obj in objs:
            safe_to_replace.discard(obj)
            ID = obj._ID
            if ID in data: 
                other = data[ID]
                if obj is not other:
                    warning = RuntimeWarning(
                        f"{repr(other)} has been replaced in registry"
                    )
                    warn(warning, stacklevel=getattr(obj, '_stacklevel', 5) - 1)
                    data[ID] = obj
                    obj._ID = ID
            else:
                data[ID] = obj
        
    def __init__(self, objs=None):
        self.data = {i.ID: i for i in objs} if objs else {}
        self.safe_to_replace = set()

    def search(self, ID):
        """Return object given ID. If ID not in registry, return None."""
        return self.data.get(ID)
    
    def __dir__(self):
        return [*self.data, 'search', 'register', 'register_safely', 'discard', 
                'clear', 'mark_safe_to_replace', 'unmark_safe_to_replace']
    
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
                    warning = RuntimeWarning(
                        f"upon renaming, {repr(obj)} replaced {repr(other)} "
                         "in registry"
                    )
                    warn(warning, 4)
                else:
                    warning = RuntimeWarning(
                        f"{repr(other)} has been replaced in registry"
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
        """Clear data."""
        self.data.clear()
    
    def discard(self, obj):
        """Remove object from data."""
        data = self.data
        if hasattr(obj, '_ID'):
            ID = obj._ID
            if ID in data and data[ID] is obj: del data[ID]
        elif isinstance(obj, str):
            if obj in data: del data[obj]
    
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
        return f"{type(self).__name__}([{', '.join([str(i) for i in self])}])"
    
    def show(self):
        if self.data:
            print('Register:\n ' + '\n '.join([repr(i) for i in self]))
        else:
            print('Register: (Empty)')
    
    _ipython_display_ = show

