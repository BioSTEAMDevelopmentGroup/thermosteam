# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from math import log
from .functor import TFunctor, TPFunctor, Functor, \
                     display_asfunctor, functor_matching_params, functor_name
from ..units_of_measure import chemical_units_of_measure, definitions, format_plot_units, convert
from numpy import inf as infinity
from inspect import signature
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

__all__ = ('thermo_model', 'create_axis_labels', 
           'ThermoModel', 'TDependentModel', 'TPDependentModel', 
           'ConstantThermoModel', 'ConstantTDependentModel',
           'ConstantTPDependentModel', 'InterpolatedTDependentModel')

REGISTERED_MODELS = []

# %% Functions

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

def default_Tmax(Tmin, Tmax):
    return Tmin + 200 if (not Tmax or Tmax is infinity) else Tmax

def default_Pmax(Pmin, Pmax):
    return 5 * Pmin if (not Pmax or Pmax is infinity) else Pmax

def T_min_max(Tmin, Tmax):
    Tmin = Tmin or 298.15
    Tmax = default_Tmax(Tmin, Tmax)
    return np.array([Tmin + 1e-6, Tmax - 1e-6])

def P_min_max(Pmin, Pmax):
    Pmin = Pmin or 101325
    Pmax = default_Pmax(Pmin, Pmax)
    return np.array([Pmin + 1e-6, Pmax - 1e-6])

def T_linspace(Tmin, Tmax):
    return np.linspace(*T_min_max(Tmin, Tmax))

def P_linspace(Pmin, Pmax):
    return np.linspace(*P_min_max(Pmin, Pmax))

def create_axis_labels(Xvar, X_units, Yvar, Y_units):
    Xvar_description = describe_parameter(Xvar, X_units)
    Yvar_description = describe_parameter(Yvar, Y_units)
    plt.xlabel(Xvar_description)
    plt.ylabel(Yvar_description)


# %% Interfaces

def model_matching_function(f):
    isa = isinstance
    if isa(f, Functor):
        for Model in REGISTERED_MODELS:
            if isa(f, Model._Functor):
                return Model
    else:
        parameters = tuple(signature(f).parameters)
        cls = functor_matching_params(parameters)
        for Model in REGISTERED_MODELS:
            if cls is Model._Functor:
                return Model
    raise ValueError('no available model matching signature')

def thermo_model(evaluate,
                 Tmin=None, Tmax=None, 
                 Pmin=None, Pmax=None,
                 name=None, var=None,
                 **kwargs):
    if not callable(evaluate):
        if Pmin or Pmax:
            ConstantModel = ConstantTPDependentModel
        elif Tmin or Tmax:
            ConstantModel = ConstantTDependentModel
        else:
            ConstantModel = ConstantThermoModel
        return ConstantModel(evaluate, Tmin, Tmax, Pmin, Pmax,
                             name, var, **kwargs)
    Model = model_matching_function(evaluate)
    return Model(evaluate, Tmin, Tmax, Pmin, Pmax, name, var, **kwargs)


# %% Thermo models

class ThermoModel:
    __slots__ = ()
    def __init_subclass__(cls, Functor=None):
        if Functor:
            REGISTERED_MODELS.append(cls)
            cls._Functor = Functor
    
    def plot_vs_T(self, T_range=None, T_units=None, units=None, P=101325,
                  label_axis=True, **plot_kwargs):
        Ts, Ys = self.tabulate_vs_T(T_range, T_units, units, P)
        plot_kwargs['label'] = plot_kwargs.get('label') or self.name
        plt.plot(Ts, Ys, **plot_kwargs)
        if label_axis: create_axis_labels('T', T_units, self.var, units)
        plt.legend()
    
    def plot_vs_P(self, P_range=None, P_units=None, units=None, T=298.15, 
                  label_axis=True, **plot_kwargs):
        Ps, Ys = self.tabulate_vs_P(P_range, P_units, units, T)
        plot_kwargs['label'] = plot_kwargs.get('label') or self.name
        plt.plot(Ps, Ys, **plot_kwargs)
        if label_axis: create_axis_labels('P', P_units, self.var, units)
        plt.legend()
    
    @property
    def __call__(self):
        return self.evaluate
    
    def __str__(self):
        return display_asfunctor(self.evaluate, self.var, type(self).__name__)
    
    def __repr__(self):
        return f"<{self}>"


class TDependentModel(ThermoModel, Functor=TFunctor):
    __slots__ = ('name', 'var', 'evaluate', 'Pmin', 'Pmax', 'Tmin', 'Tmax',
                 'integrate_by_T', 'integrate_by_T_over_T')
    
    def __init__(self, evaluate,
                 Tmin=None, Tmax=None,
                 Pmin=None, Pmax=None,
                 name=None, var=None,
                 integrate_by_T=None,
                 integrate_by_T_over_T=None):
        self.name = name or functor_name(evaluate).replace('_', ' ')
        self.var = var or evaluate.var
        self.evaluate = evaluate
        self.Pmin = Pmin or 0.
        self.Pmax = Pmax or infinity
        self.Tmin = Tmin or 0.
        self.Tmax = Tmax or infinity
        self.integrate_by_T = integrate_by_T or self.numerically_integrate_by_T
        self.integrate_by_T_over_T = integrate_by_T_over_T or self.numerically_integrate_by_T_over_T
    
    def set_value(self, var, value):
        isa = isinstance
        for f in (self.evaluate, self.integrate_by_T, self.integrate_by_T_over_T):
            if isa(f, Functor): f.set_value(var, value)
    
    def tabulate_vs_T(self, T_range=None, T_units=None, units=None, P=101325):
        if not T_range: T_range = (self.Tmin, self.Tmax)
        Ts = T_linspace(*T_range)
        Ys = np.array([self(T) for T in Ts])
        if T_units: Ts = convert_var(Ts, 'T', T_units)
        if units: Ys = convert_var(Ys, self.var, units)
        return Ts, Ys
    
    def tabulate_vs_P(self, P_range=None, P_units=None, units=None, T=298.15):
        if not P_range: P_range = (self.Pmin, self.Pmax)
        Ps = P_min_max(*P_range)
        Y = self(T)
        Ys = np.array([Y, Y])
        if P_units: Ps = convert_var(Ps, 'P', P_units)
        if units: Ys = convert_var(Ys, self.var, units)
        return Ps, Ys
    
    def indomain(self, T, P=None):
        return self.Tmin < T < self.Tmax
     
    def numerically_integrate_by_T(self, Ta, Tb, P=None):
        return self.evaluate((Tb+Ta)/2.)*(Tb - Ta)
    
    def numerically_integrate_by_T_over_T(self, Ta, Tb, P=None): 
        return self.evaluate((Tb+Ta)/2.)*log(Tb/Ta)

    def differentiate_by_T(self, T, P=None, dT=1e-12):
        return (self.evaluate(T+dT) - self.evaluate(T))/dT
    
    def differentiate_by_P(self, T, P=None, dP=1e-12):
        return 0
    
    def integrate_by_P(self, Pa, Pb, T):
        return (Pb - Pa) * self.evaluate(T)
    
    def show(self):
        print(f"{self}\n"
              f" name: {self.name}\n"
              f" Tmin: {self.Tmin:.5g} K\n"
              f" Tmax: {self.Tmax:.5g} K")
        
    _ipython_display_ =  show


class TPDependentModel(ThermoModel, Functor=TPFunctor):
    __slots__ = TDependentModel.__slots__
    __init__ = TDependentModel.__init__
    set_value = TDependentModel.set_value
    
    def tabulate_vs_T(self, T_range=None, T_units=None, units=None, P=101325):
        if not T_range: T_range = (self.Tmin, self.Tmax)
        Ts = T_linspace(*T_range)
        Ys = np.array([self(T, P) for T in Ts])
        if T_units: Ts = convert_var(Ts, 'T', T_units)
        if units: Ys = convert_var(Ys, self.var, units)
        return Ts, Ys
    
    def tabulate_vs_P(self, P_range=None, P_units=None, units=None, T=298.15):
        if not P_range: P_range = (self.Pmin, self.Pmax)
        Ps = P_linspace(*P_range)
        Ys = np.array([self(T, P) for P in Ps])
        if P_units: Ps = convert_var(Ps, 'P', P_units)
        if units: Ys = convert_var(Ys, self.var, units)
        return Ps, Ys
    
    def indomain(self, T, P):
        return self.Tmin < T < self.Tmax and self.Pmin < P < self.Pmax
    
    def numerically_integrate_by_T(self, Ta, Tb, P):
        return self.evaluate((Tb+Ta)/2, P)*(Tb - Ta)
    
    def numerically_integrate_by_T_over_T(self, Ta, Tb, P): 
        return self.evaluate((Tb+Ta)/2., P)*log(Tb/Ta)
    
    def integrate_by_P(self, Pa, Pb, T):
        return self.evaluate(T, (Pb+Pa)/2)*(Pb - Pa)

    def differentiate_by_T(self, T, P, dT=1e-12):
        return (self.evaluate(T+dT, P) - self.evaluate(T, P))/dT
    
    def differentiate_by_P(self, T, P, dP=1e-12):
        return (self.evaluate(T, P+dP) - self.evaluate(T, P))/dP

    def show(self):
        print(f"{self}\n"
              f" name: {self.name}\n"
              f" Tmin: {self.Tmin:.5g} K\n"
              f" Tmax: {self.Tmax:.5g} K\n"
              f" Pmin: {self.Pmin:.5g} Pa\n"
              f" Pmax: {self.Pmax:.5g} Pa")
        
    _ipython_display_ = show
    

class ConstantThermoModel(ThermoModel):
    __slots__ = ('value', 'name', 'var', 'Pmin', 'Pmax', 'Tmin', 'Tmax')
    def __init__(self, value, 
                 Tmin=None, Tmax=None,
                 Pmin=None, Pmax=None,
                 name=None, var=""):
        self.value = float(value)
        self.name = name or "Constant"
        self.var = var
        self.Pmin = Pmin or 0.
        self.Pmax = Pmax or infinity
        self.Tmin = Tmin or 0.
        self.Tmax = Tmax or infinity

    def set_value(self, var, value): pass

    def tabulate_vs_T(self, T_range=None, T_units=None, units=None, P=101325):
        if not T_range: T_range = (self.Tmin, self.Tmax)
        Ts = T_min_max(*T_range)
        Ys = np.array([self.value, self.value])
        if T_units: Ts = convert_var(Ts, 'T', T_units)
        if units: Ys = convert_var(Ys, self.var, units)
        return Ts, Ys
    
    def tabulate_vs_P(self, P_range=None, P_units=None, units=None, T=298.15):
        if not P_range: P_range = (self.Pmin, self.Pmax)
        Ps = P_min_max(*P_range)
        Ys = np.array([self.value, self.value])
        if P_units: Ps = convert_var(Ps, 'P', P_units)
        if units: Ys = convert_var(Ys, self.var, units)
        return Ps, Ys
    
    def indomain(self, T=None, P=None):
        return True
    
    def evaluate(self, T=None, P=None):
        return self.value
    
    def integrate_by_T(self, Ta, Tb, P=None):
        return self.value*(Tb - Ta)
    
    def integrate_by_T_over_T(self, Ta, Tb, P=None): 
        return self.value*log(Tb/Ta)    
    
    def integrate_by_P(self, Pa, Pb, T=None):
        return self.value*(Pb - Pa)
    
    def differentiate_by_T(self, T=None, P=None, dT=None):
        return 0
    
    def differentiate_by_P(self, T=None, P=None, dP=None):
        return 0

    def show(self):
        print(f"{self}\n"
              f" name: {self.name}\n"
              f" value: {self.value}\n"
              f" Tmin: {self.Tmin:.5g} K\n"
              f" Tmax: {self.Tmax:.5g} K\n"
              f" Pmin: {self.Pmin:.5g} Pa\n"
              f" Pmax: {self.Pmax:.5g} Pa")
        
    _ipython_display_ = show


class ConstantTDependentModel(ConstantThermoModel):
    indomain = TDependentModel.indomain
    

class ConstantTPDependentModel(ConstantThermoModel):
    indomain = TPDependentModel.indomain


class InterpolatedTDependentModel(ThermoModel):
    __slots__ = ('extrapolator', 'spline', 
                 'Tmin', 'Tmax', 'T_lb', 'T_ub', 
                 'name', 'var')
    
    
    def __init__(self, Ts, Ys, Tmin=None, Tmax=None, 
                 kind='cubic', name='Interpolated', var=None):
        # Only allow linear extrapolation, but with whatever transforms are specified
        self.extrapolator = interp1d(Ts, Ys, fill_value='extrapolate')
        # If more than 5 property points, create a spline interpolation
        self.spline = interp1d(Ts, Ys, kind=kind) if len(Ts)>5 else self.extrapolator
        self.Tmin = Tmin or 0.
        self.Tmax = Tmax or infinity
        self.T_lb = Ts[0]
        self.T_ub = Ts[-1]
        self.name = name
        self.var = var
    
    def evaluate(self, T, P=None):
        return self.spline(T) if (self.T_lb <= T <= self.T_ub) else self.extrapolator(T)
    
    set_value = ConstantThermoModel.set_value
    tabulate_vs_T = TDependentModel.tabulate_vs_T
    tabulate_vs_P = TDependentModel.tabulate_vs_P
    indomain = TDependentModel.indomain
    integrate_by_T = TDependentModel.numerically_integrate_by_T
    integrate_by_T_over_T = TDependentModel.numerically_integrate_by_T_over_T
    integrate_by_P = TDependentModel.integrate_by_P
    differentiate_by_T = TDependentModel.differentiate_by_T
    differentiate_by_P = TDependentModel.differentiate_by_P
    
    def show(self):
        print(f"{self}\n"
              f" name: {self.name}"
              f" Tmin: {self.Tmin:.2f}\n"
              f" Tmax: {self.Tmax:.2f}")

    _ipython_display_ = show