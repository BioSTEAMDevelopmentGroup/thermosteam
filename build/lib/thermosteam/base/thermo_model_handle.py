# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:02:53 2019

@author: yoelr
"""
import matplotlib.pyplot as plt
from collections import deque
from numpy import inf as infinity
from .thermo_model import (ThermoModel,
                           TDependentModel,
                           TPDependentModel,
                           thermo_model,
                           create_axis_labels)
from .units_of_measure import definitions
from .functor import functor_lookalike

__all__ = ('ThermoModelHandle',
           'TDependentModelHandle',
           'TPDependentModelHandle')

# %% Utilities
    
def find_model_by_name(models, name):
    for model in models:
        if model.name == name: return model
    raise LookupError(f'no model with name {name}')
    
def find_model_index_by_name(models, name):
    for index, model in enumerate(models):
        if model.name == name: return index
    raise LookupError(f'no model with name {name}')
    
def no_valid_model(chemical, var):
    definition = definitions.get(var)
    msg = f"{chemical} has no valid " if chemical else "no valid "
    msg += f"{definition.lower()} model" if definition else "model"
    return msg


# %% Handles

@functor_lookalike
class ThermoModelHandle:
    __slots__ = ('chemical', 'var', 'models',)
    
    def plot_vs_T(self, T_range=None, T_units=None, units=None, 
                  P=101325, label_axis=True, **plot_kwargs):
        Ts, Ys = self.tabulate_vs_T(T_range, T_units, units, P)
        plt.plot(Ts, Ys, **plot_kwargs)
        if label_axis: create_axis_labels('T', T_units, self.var, units)
        plt.legend()
        
    def plot_vs_P(self, P_range=None, P_units=None, units=None,
                  T=298.15, label_axis=True, **plot_kwargs):
        Ps, Ys = self.tabulate_vs_P(P_range, P_units, units, T)
        plt.plot(Ps, Ys, **plot_kwargs)
        if label_axis: create_axis_labels('P', P_units, self.var, units)
        plt.legend()
    
    def plot_models_vs_T(self, T_range=None, T_units=None, units=None,
                         P=101325, label_axis=True, **plot_kwargs):
        for model in self: model.plot_vs_T(T_range, T_units, units, P, False, **plot_kwargs)
        if label_axis: create_axis_labels('T', T_units, self.var, units)
        plt.legend()
    
    def plot_models_vs_P(self, P_range=None, P_units=None, units=None,
                         T=298.15, label_axis=True, **plot_kwargs):
        for model in self: model.plot_vs_P(P_range, P_units, units, T, False, **plot_kwargs)
        if label_axis: create_axis_labels('P', P_units, self.var, units)
        plt.legend()
    
    def __init__(self, var, models=None):
        self.chemical = None
        self.var = var
        self.models = deque(models) if models else deque()
    
    def __getitem__(self, index):
        return self.models[index]
    
    def __setitem__(self, index, model):
        assert isinstance(model, ThermoModel), (
            "a 'ThermoModelHandle' object may only "
            "contain 'ThermoModel' objects")
        self.models[index] = model
	
    def __iter__(self):
        return iter(self.models)
    
    def __bool__(self):
        return bool(self.models)
    
    def copy(self):
        return self.__class__(self.var, self.models.copy())
    __copy__ = copy
    
    def set_model_priority(self, key, priority=0):
        models = self.models
        isa = isinstance
        if isa(key, int):
            index = key
            model = models[index]
        elif isa(key, str):
            name = key
            model = find_model_by_name(models, name)
        elif isa(key, ThermoModel):
            model = key
        else:
            raise ValueError(
                '`key` must be either the index [int], name [str], '
                'or the model itself [ThermoModel]')
        models.remove(model)
        models.insert(priority, model)
    
    def move_up_model_priority(self, key, priority=0):
        models = self.models
        isa = isinstance
        if isa(key, int):
            index = key
        elif isa(key, str):
            name = key
            index = find_model_index_by_name(models, name)
        elif isa(key, ThermoModel):
            index = models.index(key)
        else:
            raise ValueError(
                '`key` must be either the index [int], name [str], '
                'or the model itself [ThermoModel]')
        models.rotate(priority - index)
    
    def add_model(self, evaluate=None,
                  Tmin=None, Tmax=None,
                  Pmin=None, Pmax=None,
                  name=None, *, top_priority=False, **funcs):
        if evaluate is None:
            return lambda evaluate: self.add_model(evaluate,
                                                   Tmin, Tmax,
                                                   Pmin, Pmax,
                                                   name,
                                                   top_priority=top_priority,
                                                   **funcs)
        if isinstance(evaluate, ThermoModel):
            model = evaluate
        else:
            model = thermo_model(evaluate, Tmin, Tmax, Pmin,
                                 Pmax, name, self.var, **funcs)
        if top_priority:
            self.models.appendleft(model)
        else:
            self.models.append(model)    
        return evaluate
       
    def show(self):
        info = f"{self}\n"
        if self.models:
            models = "\n".join([f'[{i}] {model.name}'
                                for i, model in enumerate(self.models)])
        else:
            models = "(no models available)"
        print(info + models)
        
    _ipython_display_ = show

    
class TDependentModelHandle(ThermoModelHandle):
    __slots__ = ()
    Pmin = 0
    Pmax = infinity
    tabulate_vs_T = TDependentModel.tabulate_vs_T
    tabulate_vs_P = TDependentModel.tabulate_vs_P
        
    @property
    def Tmin(self):
        return min([i.Tmin for i in self.models])
    @property
    def Tmax(self):
        return max([i.Tmax for i in self.models])
    
    def __call__(self, T, P=None):
        for model in self.models:
            if model.indomain(T): return model.evaluate(T)
        raise ValueError(f"{no_valid_model(self.chemical, self.var)} at T={T:.2f} K")
        
    def differentiate_by_T(self, T, P=None, dT=1e-12):
        for model in self.models:
            if model.indomain(T): return model.differentiate_by_T(T, dT=dT)
        raise ValueError(f"{no_valid_model(self.chemical, self.var)} at T={T:.2f} K")
        
    def differentiate_by_P(self, T, P=None, dP=1e-12):
        return 0
        
    def integrate_by_T(self, Ta, Tb, P=None):
        integral = 0.
        defined = hasattr
        for model in self.models:
            if not defined(model, 'integrate_by_T'): continue
            Tmax = model.Tmax
            Tmin = model.Tmin
            lb_satisfied = Ta > Tmin
            ub_satisfied = Tb < Tmax
            if lb_satisfied:
                if ub_satisfied:
                    try:
                        return integral + model.integrate_by_T(Ta, Tb)
                    except:
                        import pdb
                        pdb.set_trace()
                elif Ta < Tmax:
                    integral += model.integrate_by_T(Ta, Tmax)
                    Ta = Tmax
            elif ub_satisfied and Tmin < Tb:
                integral += model.integrate_by_T(Tmin, Tb)
                Tb = Tmin
        raise ValueError(f"{no_valid_model(self.chemical, self.var)} between T={Ta:.2f} to {Tb:.2f} K")
    
    def integrate_by_P(self, Pa, Pb, T):
        return (Pb - Pa) * self(T)
    
    def integrate_by_T_over_T(self, Ta, Tb, P=None):
        integral = 0.
        defined = hasattr
        for model in self.models:
            if not defined(model, 'integrate_by_T_over_T'): continue
            Tmax = model.Tmax
            Tmin = model.Tmin
            lb_satisfied = Ta > Tmin
            ub_satisfied = Tb < Tmax
            if lb_satisfied:
                if ub_satisfied:
                    return integral + model.integrate_by_T_over_T(Ta, Tb)
                elif Ta < Tmax:
                    integral += model.integrate_by_T_over_T(Ta, Tmax)
                    Ta = Tmax
            elif ub_satisfied and Tmin < Tb:
                integral += model.integrate_by_T_over_T(Tmin, Tb)
                Tb = Tmin
        raise ValueError(f"{no_valid_model(self.chemical, self.var)} between T={Ta:.2f} to {Tb:.2f} K")
    
    
class TPDependentModelHandle(ThermoModelHandle):
    __slots__ = ()
    
    Tmin = TDependentModelHandle.Tmin
    Tmax = TDependentModelHandle.Tmax
    
    tabulate_vs_T = TPDependentModel.tabulate_vs_T
    tabulate_vs_P = TPDependentModel.tabulate_vs_P
    
    @property
    def Pmin(self):
        return min([i.Pmin for i in self.models])
    @property
    def Pmax(self):
        return max([i.Pmax for i in self.models])
    
    def __call__(self, T, P):
        for model in self.models:
            if model.indomain(T, P): return model.evaluate(T, P)
        raise ValueError(f"{no_valid_model(self.chemical, self.var)} at T={T:.2f} K and P={P:.0f} Pa")

    def differentiate_by_T(self, T, P):
        for model in self.models:
            if model.indomain(T, P): return model.differentiate_by_T(T, P)
        raise ValueError(f"{no_valid_model(self.chemical, self.var)} at T={T:.2f} K and P={P:.0f} Pa")
            
    def differentiate_by_P(self, T, P):
        for model in self.models:
             if model.indomain(T, P): return model.differentiate_by_P(T, P)
        raise ValueError(f"{no_valid_model(self.chemical, self.var)} at T={T:.2f} K and P={P:.0f} Pa")

    def integrate_by_T(self, Ta, Tb, P):
        integral = 0
        defined = hasattr
        for model in self.models:
            if not (defined(model, 'integrate_by_T') and model.Pmin < P < model.Pmax): continue
            Tmax = model.Tmax
            Tmin = model.Tmin    
            lb_satisfied = Ta > Tmin
            ub_satisfied = Tb < Tmax
            if lb_satisfied:
                if ub_satisfied:
                    return integral + model.integrate_by_T(Ta, Tb, P)
                elif Ta < Tmax:
                    integral += model.integrate_by_T(Ta, Tmax, P)
                    Ta = Tmax
            elif ub_satisfied and Tmin < Tb:
                integral += model.integrate_by_T(Tmin, Tb, P)
                Tb = Tmin
        raise ValueError(f"{no_valid_model(self.chemical, self.var)} between T={Ta:.2f} to {Tb:.2f} K at P={P:.0f} Pa")
    
    def integrate_by_P(self, Pa, Pb, T):
        integral = 0
        defined = hasattr
        for model in self.models:
            if not (defined(model, 'integrate_by_P')
                    and model.Tmin < T < model.Tmax): continue
            Pmin = model.Pmin
            Pmax = model.Pmax
            lb_satisfied = Pa > Pmin
            ub_satisfied = Pb < Pmax
            if lb_satisfied:
                if ub_satisfied:
                    return integral + model.integrate_by_P(Pa, Pb, T)
                elif Pa < Pmax:
                    integral += model.integrate_by_P(Pa, Pmax, T)
                    Pa = Pmax
            elif ub_satisfied and Pmin < Pb:
                integral += model.integrate_by_P(Pmin, Pb, T)
                Pb = Pmin
        raise ValueError(f"{no_valid_model(self.chemical, self.var)} between P={Pa:5g} to {Pb:5g} Pa ast T={T:.2f}")
    
    def integrate_by_T_over_T(self, Ta, Tb, P):
        integral = 0
        defined = hasattr
        for model in self.models:
            if not (defined(model, 'integrate_by_T_over_T')
                    and model.Pmin < P < model.Pmax): continue
            Tmax = model.Tmax
            Tmin = model.Tmin    
            lb_satisfied = Ta > Tmin
            ub_satisfied = Tb < Tmax
            if lb_satisfied:
                if ub_satisfied:
                    return integral + model.integrate_by_T_over_T(Ta, Tb, P)
                elif Ta < Tmax:
                    integral += model.integrate_by_T_over_T(Ta, Tmax, P)
                    Ta = Tmax
            elif ub_satisfied and Tmin < Tb:
                integral += model.integrate_by_T_over_T(Tmin, Tb, P)
                Tb = Tmin
        raise ValueError(f"{no_valid_model(self.chemical, self.var)} between T={Ta:.2f} to {Tb:.2f} K")
        
            
    