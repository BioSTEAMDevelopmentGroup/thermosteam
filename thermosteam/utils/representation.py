# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
__all__ = ('repr_IDs_data', 'repr_kwargs', 'repr_kwarg', 'repr_couples',
           'repr_listed_values', 'repr_obj')
    
def repr_IDs_data(IDs, data, dlim=", ", start=None): # Used for Indexer and Stream representation
    return (start or dlim) + dlim.join([f"{ID}={i:.4g}" for ID, i in zip(IDs, data) if i])

def repr_kwargs(kwargs, dlim=", ", start=None):
    """
    Represent key word arguments.

    Parameters
    ----------
    kwargs : dict[str: Any]
        Key workd arguments.
    dlim : str, optional
        Delimiter. The default is ", ".
    start : str, optional
        Start of return value. Defaults to delimiter value.

    Examples
    --------
    >>> repr_kwargs({'a': 1, 'b': 2})
    ', a=1, b=2'
    
    >>> repr_kwargs({'a': 1, 'b': 2}, start="")
    'a=1, b=2'
    
    """
    if kwargs:
        start = dlim if start is None else ""
        return start + dlim.join([f"{key}={value}" for key, value in kwargs.items()])
    else:
        return ""

def repr_args(args, dlim=", ", start=None):
    """
    Represent arguments.

    Parameters
    ----------
    args : Iterable[Any]
        Arguments.
    dlim : str, optional
        Delimiter. The default is ", ".
    start : str, optional
        Start of return value. Defaults to delimiter value.

    Examples
    --------
    >>> repr_args([1, 2])
    ', 1, 2'
    
    >>> repr_args([1, 2], start="")
    '1, 2'
    
    """
    if args:
        return (start if start is not None else dlim) + dlim.join([repr(value) for value in args])
    else:
        return ""

def repr_kwarg(name, value, spec=None, dlim=", "): 
    """
    Represent parameter for class __repr__ method.

    Parameters
    ----------
    name : str
        Name of parameter.
    value : Any
        Value being represented.
    spec : str, optional
        DESCRIPTION. The default is None.
    dlim : str, optional
        Delimiter. The default is ", ".

    Examples
    --------
    >>> repr_kwarg('price', 0.12324, spec='.2f', dlim=", ")
    ', price=0.12'
    
    """
    if value:
        if spec: value = format(value, spec)
        else: value = repr(value)
        return dlim + f"{name}={value}"
    else:
        return ""

def repr_couples(dlim, IDs, data):  # Used for Indexer and Stream representation
    return dlim.join([f"('{ID}', {i:.4g})" for ID, i in zip(IDs, data) if i])

def repr_listed_values(values):
    """
    Represent values for messages.
    
    Examples
    --------
    >>> repr_listed_values(['green', 'blue', 'yellow'])
    'green, blue and yellow'
    
    """
    *values, last = values 
    if values:
        return ", ".join(values) + ' and ' + last
    else:
        return last

def repr_obj(obj): 
    """
    Represent object.
    
    Examples
    --------
    >>> class Data:
    ...     def __init__(self, a=1, b=2):
    ...         self.a = a
    ...         self.b = b
    >>> repr_obj(Data())
    'Data(a=1, b=2)'
    
    """
    return f"{type(obj).__name__}({repr_kwargs(obj.__dict__, start='')})"




