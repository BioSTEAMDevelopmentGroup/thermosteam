# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from ..units_of_measure import chemical_units_of_measure
from .. import utils
from inspect import signature

__all__ = ("functor", "Functor",  "TFunctor", 
           "TPFunctor", "TIntegralFunctor", 'display_asfunctor', 
           'functor_lookalike', 'functor_matching_params', 
           'parse_var', 'get_units', 'var_with_units')

REGISTERED_ARGS = set()
REGISTERED_FUNCTORS = []

# %% Utilities

def functor_name(functor):
    return functor.__name__ if hasattr(functor, "__name__") else type(functor).__name__    

def display_asfunctor(functor, var=None, name=None, show_var=True):
    name = name or functor_name(functor)
    info = f"{name}{str(signature(functor)).replace('self, ', '')}"
    units = functor.units if hasattr(functor, 'units') else None
    var = var or (functor.var if hasattr(functor, 'var') else None)
    if var:
        if show_var:
            info += " -> " + var_with_units(var, units)
        else:
            units = get_units(var, units)
            if units: info += f" -> {units}"
    return info

def functor_lookalike(cls):
    cls.__repr__ = Functor.__repr__
    cls.__str__ = Functor.__str__
    return cls

def functor_matching_params(params):
    N_total = len(params)
    for base in REGISTERED_FUNCTORS:
        N = base._N_args
        if N <= N_total and base._args == params[:N]: return base
    raise ValueError("could not match function signature to registered functors")

def functor_arguments(params):
    base = functor_matching_params(params)
    return base, params[base._N_args:]

def parse_var(var):
    return var.split(".") if '.' in var else (var, "")

def get_units(var, units=None):
    name, phase = parse_var(var)
    if isinstance(units, dict):
        units = units.get(name)
    if not units:
        units = chemical_units_of_measure.get(name)
    return units

def var_with_units(var, units=None):
    name, _ = parse_var(var)
    units = get_units(name)
    if units: var += f' [{units}]'
    return var


# %% Decorator
  
def functor(f=None, var=None, units=None):
    """
    Decorate a function of temperature, or both temperature and pressure 
    to have an attribute, `functor`, that serves to create its functor counterpart.
    
    Parameters
    ----------
    f : function(T, *args) or function(T, P, *args)
        Function that calculates a thermodynamic property based on temperature,
        or both temperature and pressure.
    var : str, optional
        Name of variable returned (useful for bookkeeping).
    units : dict, optional
        Units of measure for functor signature.
    
    Returns
    -------
    f : function(T, *args) or function(T, P, *args)
        Same function, but with an attribute `functor` that can create 
        its `Functor` counterpart.
    
    Notes
    -----
    The functor decorator checks the signature of the function to find the 
    names of the parameters that should be stored as data. 
    
    Examples
    --------
	Create a functor of temperature that returns the vapor pressure
	in Pascal:
	
    >>> # Describe the return value with `var`.
    >>> # Thermosteam's chemical units of measure are always assumed.
    >>> @functor(var='Psat')
    ... def Antoine(T, a, b, c):
    ...     return 10.0**(a - b / (T + c))
    >>> f(T=373.15, a=10.116, b=1687.5, c=-42.98) # functional
    101047.25357066597
    >>> f = Antoine.functor(a=10.116, b=1687.5, c=-42.98) # as functor object
    >>> f
    Functor: Antoine(T, P=None) -> Psat [Pa]
     a: 10.116
     b: 1687.5
     c: -42.98
    >>> f(T=373.15)
    101047.25357066597
    
    """
    if f:
        params = tuple(signature(f).parameters)
        base, params = functor_arguments(params)
        dct = {'__slots__': (),
               'function': staticmethod(f),
               'params': params,
               'units': units,
               'var': var}
        name = f.__name__
        f.functor = type(name, (base,), dct)
    else:
        return lambda f: functor(f, var, units)
    return f


# %% Functors

class Functor:
    __slots__ = ('__dict__',)

    def __init_subclass__(cls, args=None):
        if args:
            cls._args = args = tuple(args)
            cls._N_args = N_args = len(args)
            assert args not in REGISTERED_ARGS, (
                f"abstract functor with args={args} already implemented")
            REGISTERED_ARGS.add(args)
            index = 0
            for index, Functor in enumerate(REGISTERED_FUNCTORS):
                if Functor._N_args <= N_args: break
            REGISTERED_FUNCTORS.insert(index, cls)
    
    def __init__(self, *args, **kwargs):
        for i, j in zip(self.params, args): kwargs[i] = j
        self.__dict__ = kwargs
    
    def set_value(self, var, value):
        if var in self.__dict__: self.__dict__[var] = value
    
    def copy(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__ = self.__dict__.copy()
        return new
    
    @classmethod
    def from_other(cls, other):
        self = cls.__new__(cls)
        self.__dict__ = other.__dict__
        return self
    
    @classmethod
    def from_args(cls, data):
        self = cls.__new__(cls)
        self.__dict__ = dict(zip(self.params, data))
        return self
    
    @classmethod
    def from_kwargs(cls, data):
        self = cls.__new__(cls)
        self.__dict__ = data
        return self
    
    @classmethod
    def from_obj(cls, data):
        self = cls.__new__(cls)
        self.__dict__ = dict(zip(self.params, utils.get_obj_values(data, self.params)))
        return self
    
    def __str__(self):
        return display_asfunctor(self)
    
    def __repr__(self):
        return f"<{self}>"
    
    def show(self):
        info = f"Functor: {self}"
        data = self.__dict__
        for key, value in data.items():
            if callable(value):
                value = display_asfunctor(value, show_var=False)
                info += f"\n {key}: {value}"
                continue
            try:
                info += f"\n {key}: {value:.5g}"
            except:
                info += f"\n {key}: {value}"
            else:
                key, *_ = key.split('_')
                u = get_units(key, self.units)
                if u: info += ' ' + str(u)
        print(info)
        
    _ipython_display_ = show


class TFunctor(Functor, args=('T',)):
    __slots__ = ()
    kind = "functor of temperature (T; in K)"
    def __call__(self, T, P=None):
        return self.function(T, **self.__dict__)

class TPFunctor(Functor, args=('T', 'P')):
    __slots__ = ()
    kind = "functor of temperature (T; in K) and pressure (P; in Pa)"
    def __call__(self, T, P):
        return self.function(T, P, **self.__dict__)

class TIntegralFunctor(Functor, args=('Ta', 'Tb')):
    __slots__ = ()
    kind = "temperature integral functor (Ta to Tb; in K)"
    def __call__(self, Ta, Tb, P=None):
        return self.function(Ta, Tb, **self.__dict__)



    
    
    