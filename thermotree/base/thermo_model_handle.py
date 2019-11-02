# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:02:53 2019

@author: yoelr
"""
from numpy import inf as infinity
from .thermo_model import TPDependentModel, TDependentModel, ThermoModel, thermo_model
from .functor import display_asfunctor
from .units_of_measure import units_of_measure

__all__ = ('ThermoModelHandle', 'TDependentModelHandle',
           'TPDependentModelHandle', 'RegisteredHandles')


# %% Handles

RegisteredHandles = []

class ThermoModelHandle:
    __slots__ = ('var', 'models')
    
    def __init_subclass__(cls, Model=None, before=None, after=None):
        if Model:
            if before:
                index = RegisteredHandles.index(before)
            elif after:
                index = RegisteredHandles.index(after) + 1
            else:
                index = 0
            RegisteredHandles.insert(index, cls)
            cls._Model = Model
            
    def __init__(self, var=None):
        self.var = var
        self.models = []
    
    def model(self, evaluate,
              Tmin=None, Tmax=None,
              Pmin=None, Pmax=None,
              name=None, var=None,
              **funcs):
        self.models.append(evaluate if isinstance(evaluate, ThermoModel)
                           else thermo_model(evaluate, Tmin, Tmax, Pmin, Pmax, name, var, **funcs))
    
    def __repr__(self):
        return f"<{display_asfunctor(self)}>"
       
    def show(self):
        if self.models:
            models = ("\n").join([f'[{i}] {model.name}'
                                  for i, model in enumerate(self.models)])
        else:
            models = "(no models available)"
        print(f"{display_asfunctor(self)}\n"
              f"{models}")
        
    _ipython_display_ = show
        
    
class TDependentModelHandle(ThermoModelHandle, Model=TDependentModel):
    __slots__ = ()
    
    Pmin = 0.0
    Pmax = infinity
    
    @property
    def Tmin(self):
        return min([i.Tmin for i in self.models])
    @property
    def Tmax(self):
        return max([i.Tmax for i in self.models])
    
    def __call__(self, T, P=None):
        for model in self.models:
            if model.indomain(T): return model.evaluate(T)
        raise ValueError(f"no valid model at T={T:.2f} K")
            
    def differentiate_by_T(self, T, P=None):
        for model in self.models:
            if model.indomain(T): return model.differentiate_by_T(T)
        raise ValueError(f"no valid model at T={T:.2f} K")
    
    def differentiate_by_P(self, T, P=None):
        return 0
        
    def integrate_by_T(self, Ta, Tb, P=None):
        integral = 0
        defined = hasattr
        for model in self.models:
            if not defined(model, 'integrate_by_T'): continue
            lb_satisfied = Ta > model.Tmin
            ub_satisfied = Tb < model.Tmax
            if lb_satisfied:
                if ub_satisfied:
                    return integral + model.integrate_by_T(Ta, Tb)
                else:
                    Ti = model.Tmax
                    integral += model.integrate_by_T(Ta, Ti)
                    Ta = Ti
            elif ub_satisfied:
                Ti = model.Tmin
                integral += model.integrate_by_T(Ti, Tb)
                Tb = Ti
        raise ValueError(f"no valid model between T={Ta:.2f} to {Tb:.2f} K")
    
    def integrate_by_P(self, Pa, Pb, T):
        return (Pb - Pa) * self(T)
    
    def integrate_by_T_over_T(self, Ta, Tb, P=None):
        integral = 0
        defined = hasattr
        for model in self.models:
            if not defined(model, 'integrate_by_T_over_T'): continue
            lb_satisfied = Ta >= model.Tmin
            ub_satisfied = Tb <= model.Tmax
            if lb_satisfied:
                if ub_satisfied:
                    return integral + model.integrate_by_T_over_T(Ta, Tb)
                else:
                    Ti = model.Tmax
                    integral += model.integrate_by_T_over_T(Ta, Ti)
                    Ta = Ti
            elif ub_satisfied:
                Ti = model.Tmin
                integral += model.integrate_by_T_over_T(Ti, Tb)
                Tb = Ti
        raise ValueError(f"no valid model between T={Ta:.2f} to {Tb:.2f} K")
    
    
class TPDependentModelHandle(ThermoModelHandle, Model=TPDependentModel):
    __slots__ = ()
    
    Tmin = TDependentModelHandle.Tmin
    Tmax = TDependentModelHandle.Tmax
    
    @property
    def Pmin(self):
        return min([i.Pmin for i in self.models])
    @property
    def Pmax(self):
        return max([i.Pmax for i in self.models])
    
    def __call__(self, T, P=101325.):
        for model in self.models:
            if model.indomain(T, P): return model.evaluate(T, P)
        raise ValueError(f"no valid model at T={T:.2f} K and P={P:5g} Pa")

    def differentiate_by_T(self, T, P=101325.):
        for model in self.models:
            if model.indomain(T, P): return model.differentiate_by_T(T, P)
        raise ValueError(f"no valid model at T={T:.2f} K and P={P:5g} Pa")
            
    def differentiate_by_P(self, T, P=101325.):
        for model in self.models:
             if model.indomain(T, P): return model.differentiate_by_P(T, P)
        raise ValueError(f"no valid model at T={T:.2f} K and P={P:5g} Pa")

    def integrate_by_T(self, Ta, Tb, P=None):
        integral = 0
        defined = hasattr
        for model in self.models:
            if not (defined(model, 'integrate_by_T') and model.Pmin < P < model.Pmax): continue
            lb_satisfied = Ta > model.Tmin
            ub_satisfied = Tb < model.Tmax
            if lb_satisfied:
                if ub_satisfied:
                    return integral + model.integrate_by_T(Ta, Tb, P)
                else:
                    Ti = model.Tmax
                    integral += model.integrate_by_T(Ta, Ti, P)
                    Ta = Ti
            elif ub_satisfied:
                Ti = model.Tmin
                integral += model.integrate_by_T(Ti, Tb, P)
                Tb = Ti
        raise ValueError(f"no valid model between T={Ta:.2f} to {Tb:.2f} K at P={P:5g} Pa")
    
    def integrate_by_P(self, Pa, Pb, T):
        integral = 0
        defined = hasattr
        for model in self.models:
            if not (defined(model, 'integrate_by_P')
                    and model.Tmin < T < model.Tmax): continue
            lb_satisfied = Pa > model.Pmin
            ub_satisfied = Pb < model.Pmax
            if lb_satisfied:
                if ub_satisfied:
                    return integral + model.integrate_by_P(Pa, Pb, T)
                else:
                    Pi = model.Tmax
                    integral += model.integrate_by_P(Pa, Pi, T)
                    Pa = Pi
            elif ub_satisfied:
                Pi = model.Tmin
                integral += model.integrate_by_P(Pi, Pb, T)
                Pb = Pi
        raise ValueError(f"no valid model between P={Pa:5g} to {Pb:5g} Pa ast T={T:.2f}")
    
    def integrate_by_T_over_T(self, Ta, Tb, P):
        integral = 0
        defined = hasattr
        for model in self.models:
            if not (defined(model, 'integrate_by_T_over_T')
                    and model.Pmin < P < model.Pmax): continue
            lb_satisfied = Ta >= model.Tmin
            ub_satisfied = Tb <= model.Tmax
            if lb_satisfied:
                if ub_satisfied:
                    return integral + model.integrate_by_T_over_T(Ta, Tb, P)
                else:
                    Ti = model.Tmax
                    integral += model.integrate_by_T_over_T(Ta, Ti, P)
                    Ta = Ti
            elif ub_satisfied:
                Ti = model.Tmin
                integral += model.integrate_by_T_over_T(Ti, Tb, P)
                Tb = Ti
        raise ValueError(f"no valid model between T={Ta:.2f} to {Tb:.2f} K")
            
    