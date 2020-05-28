# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from collections import deque
from numpy import inf as infinity
from .thermo_model import (ThermoModel,
                           TDependentModel,
                           TPDependentModel,
                           thermo_model,
                           create_axis_labels)
from ..exceptions import DomainError
from ..units_of_measure import definitions
from .functor import functor_lookalike
import matplotlib.pyplot as plt

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
    msg = f"{chemical} (CAS: {chemical.CAS}) has no valid " if chemical else "no valid "
    msg += f"{definition.lower()} model" if definition else "model"
    return msg

def as_model_index(models, key):
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
    return index

def as_model(models, key):
    """Return model given the index, name, or the model itself."""
    isa = isinstance
    if isa(key, int):
        index = key
        model = models[index]
    elif isa(key, str):
        name = key
        model = find_model_by_name(models, name)
    elif isa(key, ThermoModel):
        model = key
        assert model in models, "''"
    else:
        raise ValueError(
            '`key` must be either the index [int], name [str], '
            'or the model itself [ThermoModel]')
    return model


# %% Handles

@functor_lookalike
class ThermoModelHandle:
    __slots__ = ('_chemical', '_var', '_models',)
    
    @property
    def chemical(self):
        """[Chemical] Parent chemical."""
        return self._chemical
    @property
    def var(self):
        """[str] Return variable."""
        return self._var
    @property
    def models(self):
        """Deque[ThermoModel] All models."""
        return self._models
    
    def set_value(self, var, value):
        for model in self._models: model.set_value(var, value)
    
    def plot_vs_T(self, T_range=None, T_units=None, units=None, 
                  P=101325, label_axis=True, **plot_kwargs):
        Ts, Ys = self.tabulate_vs_T(T_range, T_units, units, P)
        plt.plot(Ts, Ys, **plot_kwargs)
        if label_axis: create_axis_labels('T', T_units, self._var, units)
        plt.legend()
        
    def plot_vs_P(self, P_range=None, P_units=None, units=None,
                  T=298.15, label_axis=True, **plot_kwargs):
        Ps, Ys = self.tabulate_vs_P(P_range, P_units, units, T)
        plt.plot(Ps, Ys, **plot_kwargs)
        if label_axis: create_axis_labels('P', P_units, self._var, units)
        plt.legend()
    
    def plot_models_vs_T(self, T_range=None, T_units=None, units=None,
                         P=101325, label_axis=True, **plot_kwargs):
        for model in self: model.plot_vs_T(T_range, T_units, units, P, False, **plot_kwargs)
        if label_axis: create_axis_labels('T', T_units, self._var, units)
        plt.legend()
    
    def plot_models_vs_P(self, P_range=None, P_units=None, units=None,
                         T=298.15, label_axis=True, **plot_kwargs):
        for model in self: model.plot_vs_P(P_range, P_units, units, T, False, **plot_kwargs)
        if label_axis: create_axis_labels('P', P_units, self._var, units)
        plt.legend()
    
    def __init__(self, var, models=None):
        self._chemical = None
        self._var = var
        self._models = deque(models) if models else deque()
    
    def __getitem__(self, index):
        return self._models[index]
    
    def __setitem__(self, index, model):
        assert isinstance(model, ThermoModel), (
            "a 'ThermoModelHandle' object may only "
            "contain 'ThermoModel' objects")
        self._models[index] = model
	
    def __iter__(self):
        return iter(self._models)
    
    def __bool__(self):
        return bool(self._models)
    
    def copy(self):
        return self.__class__(self._var, self._models.copy())
    __copy__ = copy
    
    def set_model_priority(self, key, priority=0):
        """
        Set model priority.

        Parameters
        ----------
        key : int, str, or ThermoModel
            Index, name, or the model itself.
        priority : int, optional
            Index to place model. The default is 0.

        """
        models = self._models
        model = as_model(models, key)
        models.remove(model)
        models.insert(priority, model)
    
    def move_up_model_priority(self, key, priority=0):
        """
        Move up model priority.
        
        Parameters
        ----------
        key : int, str, or ThermoModel
            Index, name, or the model itself.
        priority : int, optional
            Index to place model. The default is 0.
            
        """
        index = as_model_index(self._models, key)
        self._models.rotate(priority - index)
    
    def add_model(self, evaluate=None,
                  Tmin=None, Tmax=None,
                  Pmin=None, Pmax=None,
                  name=None, top_priority=False, 
                  **kwargs):
        """
        Add model to handle.

        Parameters
        ----------
        evaluate : ThermoModel, function, or float
            Model function or constant value.
        Tmin : float, optional
            Minimum temperature the that model can be used [K]. Defaults to 0.
        Tmax : float, optional
            Maximum temperature the that model can be used [K]. Defaults to infinity.
        Pmin : float, optional
            Minimum pressure the that model can be used [Pa]. Defaults to 0.
        Pmax : float, optional
            Maximum pressure the that model can be used [Pa]. Defaults to infinity.
        name : str, optional
            Name of the model. Defaults to function name.
        top_priority : int, optional
            Whether to add the model in 0th index/priority. The default is False.
        
        Other Parameters
        ----------------
        integrate_by_T : function(Ta, Tb, ...)
            Analytical integration function (to replace numerical integration).
        integrate_by_T_over_T : function(Ta, Tb, ...)
            Analytical integration function (to replace numerical integration).
        
        Notes
        -----
        This method can be used as a function decorator, but the function is
        not replaced with the model.

        """
        if evaluate is None:
            return lambda evaluate: self.add_model(evaluate,
                                                   Tmin, Tmax,
                                                   Pmin, Pmax,
                                                   name, top_priority,
                                                   **kwargs)
        if isinstance(evaluate, ThermoModel):
            model = evaluate
        elif isinstance(evaluate, ThermoModelHandle):
            raise ValueError("ThermoModelHandle object cannot be added as a "
                             "model; only ThermoModel objects, functions, "
                             "and constants can be added")
        else:
            model = thermo_model(evaluate, Tmin, Tmax, Pmin, Pmax,
                                 name, self._var, **kwargs)
        if top_priority:
            self._models.appendleft(model)
        else:
            self._models.append(model)    
        return evaluate
       
    def remove(self, key):
        """
        Remove model
        
        Parameters
        ----------
        key : int, str, or ThermoModel
            Index, name, or the model itself.
            
        """
        model = as_model(self._models, key)
        self._models.remove(model)
       
    def show(self):
        info = f"{self}\n"
        if self._models:
            models = "\n".join([f'[{i}] {model.name}'
                                for i, model in enumerate(self._models)])
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
        return min([i.Tmin for i in self._models])
    @property
    def Tmax(self):
        return max([i.Tmax for i in self._models])
    
    def __call__(self, T, P=None):
        for model in self._models:
            if model.indomain(T): return model.evaluate(T)
        raise DomainError(f"{no_valid_model(self._chemical, self._var)} "
                         f"at T={T:.2f} K")
    
    at_T = __call__
    
    def try_out(self, T, P=None):
        for model in self._models:
            if model.indomain(T): return model.evaluate(T)
    
    def differentiate_by_T(self, T, P=None, dT=1e-12):
        for model in self._models:
            if model.indomain(T): return model.differentiate_by_T(T, dT=dT)
        raise DomainError(f"{no_valid_model(self._chemical, self._var)} "
                         f"at T={T:.2f} K")
        
    def differentiate_by_P(self, T, P=None, dP=1e-12):
        return 0
        
    def integrate_by_T(self, Ta, Tb, P=None):
        integral = 0.
        defined = hasattr
        for model in self._models:
            if not defined(model, 'integrate_by_T'): continue
            Tmax = model.Tmax
            Tmin = model.Tmin
            lb_satisfied = Ta > Tmin
            ub_satisfied = Tb < Tmax
            if lb_satisfied:
                if ub_satisfied:
                    return integral + model.integrate_by_T(Ta, Tb)
                elif Ta < Tmax:
                    integral += model.integrate_by_T(Ta, Tmax)
                    Ta = Tmax
            elif ub_satisfied and Tmin < Tb:
                integral += model.integrate_by_T(Tmin, Tb)
                Tb = Tmin
        raise DomainError(f"{no_valid_model(self._chemical, self._var)} "
                         f"between T={Ta:.2f} to {Tb:.2f} K")
    
    def integrate_by_P(self, Pa, Pb, T):
        return (Pb - Pa) * self(T)
    
    def integrate_by_T_over_T(self, Ta, Tb, P=None):
        integral = 0.
        defined = hasattr
        for model in self._models:
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
        raise DomainError(f"{no_valid_model(self._chemical, self._var)} "
                         f"between T={Ta:.2f} to {Tb:.2f} K")
    
    
class TPDependentModelHandle(ThermoModelHandle):
    __slots__ = ()
    
    Tmin = TDependentModelHandle.Tmin
    Tmax = TDependentModelHandle.Tmax
    
    tabulate_vs_T = TPDependentModel.tabulate_vs_T
    tabulate_vs_P = TPDependentModel.tabulate_vs_P
    
    @property
    def Pmin(self):
        return min([i.Pmin for i in self._models])
    @property
    def Pmax(self):
        return max([i.Pmax for i in self._models])
    
    def __call__(self, T, P):
        for model in self._models:
            if model.indomain(T, P): return model.evaluate(T, P)
        raise DomainError(f"{no_valid_model(self._chemical, self._var)} "
                          f"at T={T:.2f} K and P={P:.0f} Pa")

    def at_T(self, T):
        isa = isinstance
        for model in self._models:
            if isa(model, TDependentModel) and model.indomain(T):
                return model.evaluate(T)
        raise DomainError(f"{no_valid_model(self._chemical, self._var)} "
                          f"at T={T:.2f} K")

    def try_out(self, T, P):
        for model in self._models:
            if model.indomain(T, P): return model.evaluate(T, P)

    def differentiate_by_T(self, T, P):
        for model in self._models:
            if model.indomain(T, P): return model.differentiate_by_T(T, P)
        raise DomainError(f"{no_valid_model(self._chemical, self._var)} "
                          f"at T={T:.2f} K and P={P:.0f} Pa")
            
    def differentiate_by_P(self, T, P):
        for model in self._models:
             if model.indomain(T, P): return model.differentiate_by_P(T, P)
        raise DomainError(f"{no_valid_model(self._chemical, self._var)} "
                          f"at T={T:.2f} K and P={P:.0f} Pa")

    def integrate_by_T(self, Ta, Tb, P):
        integral = 0
        defined = hasattr
        for model in self._models:
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
        raise DomainError(f"{no_valid_model(self._chemical, self._var)} "
                          f"between T={Ta:.2f} to {Tb:.2f} K at P={P:.0f} Pa")
    
    def integrate_by_P(self, Pa, Pb, T):
        integral = 0
        defined = hasattr
        for model in self._models:
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
        raise DomainError(f"{no_valid_model(self._chemical, self._var)} "
                          f"between P={Pa:5g} to {Pb:5g} Pa ast T={T:.2f}")
    
    def integrate_by_T_over_T(self, Ta, Tb, P):
        integral = 0
        defined = hasattr
        for model in self._models:
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
        raise DomainError(f"{no_valid_model(self._chemical, self._var)} "
                          f"between T={Ta:.2f} to {Tb:.2f} K")
        
            
    