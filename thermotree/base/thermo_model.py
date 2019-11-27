# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:23:44 2019

@author: yoelr
"""
from math import log
from scipy.interpolate import interp1d
from .functor import TFunctor, TPFunctor, Functor, \
                     display_asfunctor, functor_matching_params, functor_name
from .units_of_measure import units_of_measure
from numpy import inf as infinity
from inspect import signature

__all__ = ('thermo_model', 'ThermoModel',
           'TDependentModel', 'TPDependentModel', 
           'ConstantThermoModel', 'ConstantTDependentModel',
           'ConstantTPDependentModel', 'InterpolatedTDependentModel')


RegisteredModels = []

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
        
    assert found_model, 'no available model matching evaluate signature'
    return Model(evaluate, Tmin, Tmax, Pmin, Pmax, name, **funcs)


# %% Thermo models

class ThermoModel:
    Tmin = Pmin = 0.0
    Tmax = Pmax = infinity
    
    def __init_subclass__(cls, Functor=None):
        if Functor:
            RegisteredModels.append(cls)
            cls._Functor = Functor
    
    def __init__(self, evaluate,
                 Tmin=None, Tmax=None,
                 Pmin=None, Pmax=None,
                 name=None, **funcs):
        self.name = name or functor_name(evaluate)
        self.evaluate = evaluate
        self.__dict__.update(funcs)
        if Pmin: self.Pmin = Pmin
        if Pmax: self.Pmax = Pmax
        if Tmin: self.Tmin = Tmin
        if Tmax: self.Tmax = Tmax
    
    def __repr__(self):
        return f"<{type(self).__name__}: {self.name}>"


class TDependentModel(ThermoModel, Functor=TFunctor):
    
    @property
    def var(self):
        try: return self.evaluate.var
        except: return None
    
    def indomain(self, T, P=None):
        return self.Tmin < T < self.Tmax
    
    def integrate_by_T(self, Ta, Tb, P=None):
        return self.evaluate((Tb+Ta)/2.)*(Tb - Ta)
    
    def integrate_by_T_over_T(self, Ta, Tb, P=None): 
        return self.evaluate((Tb+Ta)/2.)*log(Tb/Ta)

    def differentiate_by_T(self, T, dT=1e-12, P=None):
        return (self.evaluate(T+dT) - self.evaluate(T))/dT
    
    def differentiate_by_P(self, T, P=None):
        return 0
    
    def integrate_by_P(self, Pa, Pb, T):
        return (Pb - Pa) * self(T)
    
    def show(self):
        print(f"{type(self).__name__}: {self.name}\n"
              f" evaluate: {display_asfunctor(self.evaluate)}\n"
              f" Tmin: {self.Tmin:.2f}\n"
              f" Tmax: {self.Tmax:.2f}")
        
    _ipython_display_ =  show


class TPDependentModel(ThermoModel, Functor=TPFunctor):
    
    var = TDependentModel.var
    
    def indomain(self, T, P):
        return (self.Tmin < T < self.Tmax) and (self.Pmin < P < self.Pmax)
    
    def integrate_by_T(self, Ta, Tb, P):
        return self.evaluate((Tb+Ta)/2, P)*(Tb - Ta)
    
    def integrate_by_T_over_T(self, Ta, Tb, P=101325.): 
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
    
    def indomain(self, T, P=None):
        return True
    
    def evaluate(self, T=None, P=None):
        return self.value
    
    def integrate_by_T(self, Ta, Tb, P=101325.):
        return self.value*(Tb - Ta)
    
    def integrate_by_T_over_T(self, Ta, Tb, P=101325.): 
        return self.value*log(Tb/Ta)    
    
    def integrate_by_P(self, Pa, Pb, T):
        return self.value*(Pb - Pa)
    
    def differentiate_by_T(self, T, P=101325.):
        return 0
    
    def differentiate_by_P(self, T, P):
        return 0

    def show(self):
        var = self.var
        if var in units_of_measure:
            units = " " + units_of_measure[var]
        else:
            units = ""
        var = f' ({self.var})' if self.var else ""
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