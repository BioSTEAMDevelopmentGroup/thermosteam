# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:23:44 2019

@author: yoelr
"""
from math import log
from scipy.interpolate import interp1d
from .functor import TFunctor, TPFunctor, Functor, \
                     display_asfunctor, functor_matching_params, functor_name
from .units_of_measure import chemical_units_of_measure, definitions, format_plot_units, convert
from numpy import inf as infinity
from inspect import signature
import matplotlib.pyplot as plt
import numpy as np

__all__ = ('thermo_model', 'create_axis_labels', 
           'ThermoModel', 'TDependentModel', 'TPDependentModel', 
           'ConstantThermoModel', 'ConstantTDependentModel',
           'ConstantTPDependentModel', 'InterpolatedTDependentModel')

RegisteredModels = []

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
        else: info += f" [-]"
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

def thermo_model(evaluate,
                 Tmin=None, Tmax=None,
                 Pmin=None, Pmax=None,
                 name=None, var=None,
                 **funcs):
    if not callable(evaluate):
        if Pmin or Pmax:
            ConstantModel = ConstantTPDependentModel
        elif Tmin or Tmax:
            ConstantModel = ConstantTDependentModel
        else:
            ConstantModel = ConstantThermoModel
        return ConstantModel(evaluate, Tmin, Tmax, Pmin, Pmax, name, var)
    
    found_model = False
    if isinstance(evaluate, Functor):
        for Model in RegisteredModels:
            if isinstance(evaluate, Model._Functor):
                found_model = True
                break
    else:
        parameters = tuple(signature(evaluate).parameters)
        cls = functor_matching_params(parameters)
        for Model in RegisteredModels:
            if cls is Model._Functor:
                found_model = True
                break
        
    assert found_model, 'no available model matching signature'
    return Model(evaluate, Tmin, Tmax, Pmin, Pmax, name, var, **funcs)


# %% Thermo models

class ThermoModel:
    Tmin = Pmin = 0.0
    Tmax = Pmax = infinity
    IGNORE_DOMAIN = False
    
    def __init_subclass__(cls, Functor=None):
        if Functor:
            RegisteredModels.append(cls)
            cls._Functor = Functor
    
    def __init__(self, evaluate,
                 Tmin=None, Tmax=None,
                 Pmin=None, Pmax=None,
                 name=None, var=None, **funcs):
        self.var = var or evaluate.var
        self.name = name or functor_name(evaluate)
        self.evaluate = evaluate
        self.__dict__.update(funcs)
        if Pmin: self.Pmin = Pmin
        if Pmax: self.Pmax = Pmax
        if Tmin: self.Tmin = Tmin
        if Tmax: self.Tmax = Tmax
    
    def plot_vs_T(self, T_range=None, T_units=None, units=None, P=101325,
                  label_axis=True, **plot_kwargs):
        Ts, Ys = self.tabulate_vs_T(T_range, T_units, units, P)
        plot_kwargs['label'] = plot_kwargs.get('label') or self.name.replace('_', " ")
        plt.plot(Ts, Ys, **plot_kwargs)
        if label_axis: create_axis_labels('T', T_units, self.var, units)
    
    def plot_vs_P(self, P_range=None, P_units=None, units=None, T=298.15, 
                  label_axis=True, **plot_kwargs):
        Ps, Ys = self.tabulate_vs_P(P_range, P_units, units, T)
        plot_kwargs['label'] = plot_kwargs.get('label') or self.name.replace('_', " ")
        plt.plot(Ps, Ys, **plot_kwargs)
        if label_axis: create_axis_labels('P', P_units, self.var, units)
    
    @property
    def __call__(self):
        return self.evaluate
    
    def __repr__(self):
        return f"<{type(self).__name__}: {self.name}>"


class TDependentModel(ThermoModel, Functor=TFunctor):
    
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
        return self.IGNORE_DOMAIN or self.Tmin < T < self.Tmax
    
    def integrate_by_T(self, Ta, Tb, P=None):
        return self.evaluate((Tb+Ta)/2.)*(Tb - Ta)
    
    def integrate_by_T_over_T(self, Ta, Tb, P=None): 
        return self.evaluate((Tb+Ta)/2.)*log(Tb/Ta)

    def differentiate_by_T(self, T, P=None, dT=1e-12):
        return (self.evaluate(T+dT) - self.evaluate(T))/dT
    
    def differentiate_by_P(self, T, P=None, dP=1e-12):
        return 0
    
    def integrate_by_P(self, Pa, Pb, T):
        return (Pb - Pa) * self(T)
    
    def show(self):
        print(f"{type(self).__name__}: {self.name}\n"
              f" evaluate: {display_asfunctor(self.evaluate)}\n"
              f" Tmin: {self.Tmin:.5g} K\n"
              f" Tmax: {self.Tmax:.5g} K")
        
    _ipython_display_ =  show


class TPDependentModel(ThermoModel, Functor=TPFunctor):
    
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
        return self.IGNORE_DOMAIN or (self.Tmin < T < self.Tmax
                                      and self.Pmin < P < self.Pmax)
    
    def integrate_by_T(self, Ta, Tb, P):
        return self.evaluate((Tb+Ta)/2, P)*(Tb - Ta)
    
    def integrate_by_T_over_T(self, Ta, Tb, P): 
        return self.evaluate((Tb+Ta)/2., P)*log(Tb/Ta)
    
    def integrate_by_P(self, Pa, Pb, T):
        return self.evaluate((Pb+Pa)/2, T)*(Pb - Pa)

    def differentiate_by_T(self, T, P, dT=1e-12):
        return (self.evaluate(T+dT, P) - self.evaluate(T, P))/dT
    
    def differentiate_by_P(self, T, P, dP=1e-12):
        return (self.evaluate(T, P+dP) - self.evaluate(T, P))/dP

    def show(self):
        print(f"{type(self).__name__}: {self.name}\n"
              f" evaluate: {display_asfunctor(self.evaluate)}\n"
              f" Tmin: {self.Tmin:.5g} K\n"
              f" Tmax: {self.Tmax:.5g} K\n"
              f" Pmin: {self.Pmin:.5g} Pa\n"
              f" Pmax: {self.Pmax:.5g} Pa")
        
    _ipython_display_ = show
    

class ConstantThermoModel(ThermoModel):
    
    def __init__(self, value, 
                 Tmin=None, Tmax=None,
                 Pmin=None, Pmax=None,
                 name=None, var=""):
        self.value = value
        self.name = name or "Constant"
        self.var = var
        if Pmin: self.Pmin = Pmin
        if Pmax: self.Pmax = Pmax
        if Tmin: self.Tmin = Tmin
        if Tmax: self.Tmax = Tmax

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
        if self.var :
            var = f' ({self.var})'
            var_no_phase, *_ = self.var.split('.')
            if var_no_phase in chemical_units_of_measure:
                units = " " + str(chemical_units_of_measure[var_no_phase])
            else:
                units = ""
        else:
            var = units = ""
        print(f"{type(self).__name__}: {self.name}\n"
              f" value{var}: {self.value}{units}\n"
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
    
    def __init__(self, Ts, Ys, Tmin=None, Tmax=None, 
                 kind='cubic', name='Interpolated', var=None):
        # Only allow linear extrapolation, but with whatever transforms are specified
        self.extrapolator = interp1d(Ts, Ys, fill_value='extrapolate')
        # If more than 5 property points, create a spline interpolation
        self.spline = interp1d(Ts, Ys, kind=kind) if len(Ts)>5 else self.extrapolator
        if Tmin: self.Tmin = Tmin
        if Tmax: self.Tmax = Tmax
        self.T_lb = Ts[0]
        self.T_ub = Ts[-1]
        self.name = name
        self.var = var
    
    def evaluate(self, T, P=None):
        return self.spline(T) if (self.T_lb <= T <= self.T_ub) else self.extrapolator(T)
    
    tabulate_vs_T = TDependentModel.tabulate_vs_T
    tabulate_vs_P = TDependentModel.tabulate_vs_P
    indomain = TDependentModel.indomain
    integrate_by_T = TDependentModel.integrate_by_T
    integrate_by_T_over_T = TDependentModel.integrate_by_T_over_T
    integrate_by_P = TDependentModel.integrate_by_P
    differentiate_by_T = TDependentModel.differentiate_by_T
    differentiate_by_P = TDependentModel.differentiate_by_P
    
    def show(self):
        print(f"{type(self).__name__}: {self.name}\n"
              f" evaluate: {display_asfunctor(self.evaluate, self.var)}"
              f" Tmin: {self.Tmin:.2f}\n"
              f" Tmax: {self.Tmax:.2f}")

    _ipython_display_ = show