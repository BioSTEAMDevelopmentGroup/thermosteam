# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import numba
from numba.experimental import jitclass
from numba.extending import as_numba_type

__all__ = ('jitdata',)

def jitdata(name_or_cls, /, **fields):
    """
    Create a jitclass with a lightweight __init__ method
    which saves input arguments as attributes.
        
    """
    for i, j in fields.items():
        if isinstance(j, str): fields[i] = eval(j, numba.__dict__)
    if isinstance(name_or_cls, str):
        name = name_or_cls
        cls = None
    elif isinstance(name_or_cls, type):
        cls = name_or_cls
        name = cls.__name__
        annotations = cls.__annotations__
        for i, j in annotations.items():
            if isinstance(j, str): annotations[i] = eval(j, numba.__dict__)
        all_fields = {**fields, **cls.__annotations__}
    else:
        raise TypeError('first argument must be a class or name')
    if not all_fields: raise ValueError('at least one field must be given')
    if cls is not None and '__init__' in cls.__dict__:
        # return cls
        return jitclass(cls, spec=[(i, as_numba_type(j)) for i, j in fields.items()])
    else:
        arguments = ', '.join(all_fields)
        init = [f"def __init__(self, {arguments}):"]
        for i in all_fields: init.append(f'self.{i} = {i}')
        init = '\n    '.join(init)
        dct = {}
        exec(init, dct)
        if cls is None: 
            cls = type(name, (), {'__init__': dct['__init__']})
        else:
            cls.__init__ = dct['__init__']
        # return cls
        return jitclass(cls, spec=[(i, as_numba_type(j)) for i, j in fields.items()])

