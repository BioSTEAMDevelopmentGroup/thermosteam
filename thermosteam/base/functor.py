# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from ..units_of_measure import chemical_units_of_measure, definitions, format_plot_units, convert
from .. import utils
from .. import functors
from inspect import signature
import matplotlib.pyplot as plt
import numpy as np

__all__ = ("functor", "Functor",  "TFunctor", 
           "TPFunctor", 'display_asfunctor', 
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

# %% Plotting utilities

def convert_var(value, var, units):
    if var and units:
        var, *_ = var.split('.')
        return convert(value, chemical_units_of_measure[var].units, units)
    else:
        return value

def describe_parameter(var, units):
    info = definitions.get(var) or var
    if info:
        var, *_ = var.split('.')
        units = units or chemical_units_of_measure.get(var)
        units = format_plot_units(units)
        if units: info += f" [{units}]"
        else: info += " [-]"
        return info

def create_axis_labels(Xvar, X_units, Yvar, Y_units):
    Xvar_description = describe_parameter(Xvar, X_units)
    Yvar_description = describe_parameter(Yvar, Y_units)
    plt.xlabel(Xvar_description)
    plt.ylabel(Yvar_description)

def plot_functors_vs_T(fs, T_range=None, T_units=None, units=None,
                        P=101325, label_axis=True, **plot_kwargs): # pragma: no cover
    for f in fs: f.plot_vs_T(T_range, T_units, units, P, False, **plot_kwargs)
    if label_axis: create_axis_labels('T', T_units, f.var, units)
    plt.legend()
    
def plot_functors_vs_P(fs, P_range=None, P_units=None, units=None,
                       T=298.15, label_axis=True, **plot_kwargs): # pragma: no cover
    for f in fs: f.plot_vs_P(P_range, P_units, units, T, False, **plot_kwargs)
    if label_axis: create_axis_labels('P', P_units, f.var, units)
    plt.legend()

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
    >>> import thermosteam as tmo
    >>> @tmo.functor(var='Psat')
    ... def Antoine(T, a, b, c):
    ...     return 10.0**(a - b / (T + c))
    >>> Antoine(T=373.15, a=10.116, b=1687.5, c=-42.98) # functional
    101157.148
    >>> f = Antoine.functor(a=10.116, b=1687.5, c=-42.98) # as functor object
    >>> f.show()
    Functor: Antoine(T, P=None) -> Psat [Pa]
     a: 10.116
     b: 1687.5
     c: -42.98
    >>> f(T=373.15)
    101157.148
    
    All functors are saved in the `functors` module:
    
    >>> tmo.functors.Antoine
    <class 'thermosteam.functors.Antoine'>
    
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
        f.functor = cls = type(name, (base,), dct)
        cls.__module__ = functors.__name__
        setattr(functors, name, cls)
    else:
        return lambda f: functor(f, var, units)
    return f


# %% Functors

class Functor:
    __slots__ = ('__dict__',)
    hook = None

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
        return display_asfunctor(self).replace(', **kwargs', '')
    
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
    
    def __call__(self, T, P=None, **kwargs):
        return self.function(T, **self.__dict__, **kwargs)
        
    def tabulate_vs_T(self, Tmin, Tmax, T_units=None, units=None, P=101325):
        Ts = np.linspace(Tmin, Tmax)
        Ys = np.array([self(T) for T in Ts])
        if T_units: Ts = convert_var(Ts, 'T', T_units)
        if units: Ys = convert_var(Ys, self.var, units)
        return Ts, Ys
    
    def tabulate_vs_P(self, Pmin, Pmax, P_units=None, units=None, T=298.15):
        Ps = np.linspace(Pmin, Pmax)
        Y = self(T)
        Ys = np.array([Y, Y])
        if P_units: Ps = convert_var(Ps, 'P', P_units)
        if units: Ys = convert_var(Ys, self.var, units)
        return Ps, Ys
    
    def plot_vs_T(self, Tmin, Tmax, T_units=None, units=None, P=101325,
                  label_axis=True, **plot_kwargs):
        Ts, Ys = self.tabulate_vs_T(Tmin, Tmax, T_units, units, P)
        plot_kwargs['label'] = plot_kwargs.get('label') or self.name
        plt.plot(Ts, Ys, **plot_kwargs)
        if label_axis: create_axis_labels('T', T_units, self.var, units)
        plt.legend()
    
    def plot_vs_P(self, Pmin, Pmax, P_units=None, units=None, T=298.15, 
                  label_axis=True, **plot_kwargs):
        Ps, Ys = self.tabulate_vs_P(Pmin, Pmax, P_units, units, T)
        plot_kwargs['label'] = plot_kwargs.get('label') or self.name
        plt.plot(Ps, Ys, **plot_kwargs)
        if label_axis: create_axis_labels('P', P_units, self.var, units)
        plt.legend()
    

class TPFunctor(Functor, args=('T', 'P')):
    __slots__ = ()
    kind = "functor of temperature (T; in K) and pressure (P; in Pa)"
    
    def __call__(self, T, P, **kwargs):
        return self.function(T, P, **self.__dict__, **kwargs)
    
    def tabulate_vs_T(self, Tmin, Tmax, T_units=None, units=None, P=101325):
        Ts = np.linspace(Tmin, Tmax)
        Ys = np.array([self(T, P) for T in Ts])
        if T_units: Ts = convert_var(Ts, 'T', T_units)
        if units: Ys = convert_var(Ys, self.var, units)
        return Ts, Ys
    
    def tabulate_vs_P(self, Pmin, Pmax, P_units=None, units=None, T=298.15):
        Ps = np.linspace(Pmin, Pmax)
        Ys = np.array([self(T, P) for P in Ps])
        if P_units: Ps = convert_var(Ps, 'P', P_units)
        if units: Ys = convert_var(Ys, self.var, units)
        return Ps, Ys
    
    plot_vs_T = TFunctor.plot_vs_T
    plot_vs_P = TFunctor.plot_vs_P


    
    
    