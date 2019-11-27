# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:02:53 2019

@author: yoelr
"""
from .thermo_model import ThermoModel, ConstantThermoModel, thermo_model
from .functor import display_asfunctor

__all__ = ('ThermoModelHandle', 'TDependentModelHandle',
           'TPDependentModelHandle')

# %% Utilities

def find_constant_model(models):
    isa = isinstance
    CTM = ConstantThermoModel
    for model in models:
        if isa(model, CTM): return model
            


# %% Handles

class ThermoModelHandle:
    __slots__ = ('models',)
    
    @property
    def var(self):
        for i in self.models:
            var = i.var
            if var: return var
         
    def __init__(self):
        self.models = []
    
    def __bool__(self):
        return bool(self.models)
    
    def model(self, evaluate,
              Tmin=None, Tmax=None,
              Pmin=None, Pmax=None,
              name=None, var=None,
              *, first=False, **funcs):
        if isinstance(evaluate, ThermoModel):
            model = evaluate
        else:
            model = thermo_model(evaluate, Tmin, Tmax, Pmin,
                                   Pmax, name, var, **funcs)
        if first:
            self.models.insert(0, model)
        else:
            self.models.append(model)    
    
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

    
class TDependentModelHandle(ThermoModelHandle):
    __slots__ = ()
    
    def to_TP(self, T, P=None):
        models = self.models
        constant_model = find_constant_model(models)
        if constant_model:
            models.remove(constant_model)
        else:
            constant_model = ConstantThermoModel(self(T))
        models.insert(0, constant_model)
        
    @property
    def Tmin(self):
        return min([i.Tmin for i in self.models])
    @property
    def Tmax(self):
        return max([i.Tmax for i in self.models])
    
    def __call__(self, T):
        for model in self.models:
            if model.indomain(T): return model.evaluate(T)
        # return model.evaluate(T)
        raise ValueError(f"no valid model at T={T:.2f} K")
            
    def differentiate_by_T(self, T):
        for model in self.models:
            if model.indomain(T): return model.differentiate_by_T(T)
        # return model.differentiate_by_T(T) 
        raise ValueError(f"no valid model at T={T:.2f} K")
        
    def integrate_by_T(self, Ta, Tb):
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
        # return integral + model.integrate_by_T(Ta, Tb)
        raise ValueError(f"no valid model between T={Ta:.2f} to {Tb:.2f} K")
    
    def integrate_by_P(self, Pa, Pb, T):
        return (Pb - Pa) * self(T)
    
    def integrate_by_T_over_T(self, Ta, Tb):
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
        # return integral + model.integrate_by_T_over_T(Ta, Tb)
        raise ValueError(f"no valid model between T={Ta:.2f} to {Tb:.2f} K")
    
    
class TPDependentModelHandle(ThermoModelHandle):
    __slots__ = ()
    
    Tmin = TDependentModelHandle.Tmin
    Tmax = TDependentModelHandle.Tmax
    
    def to_TP(self, T, P=None):
        models = self.models
        constant_model = find_constant_model(models)
        if constant_model:
            models.remove(constant_model)
        else:
            constant_model = ConstantThermoModel(self(T))
        models.insert(0, constant_model)
    
    @property
    def Pmin(self):
        return min([i.Pmin for i in self.models])
    @property
    def Pmax(self):
        return max([i.Pmax for i in self.models])
    
    def __call__(self, T, P=101325.):
        for model in self.models:
            if model.indomain(T, P): return model.evaluate(T, P)
        # return model.evaluate(T, P)
        raise ValueError(f"no valid model at T={T:.2f} K and P={P:5g} Pa")

    def differentiate_by_T(self, T, P=101325.):
        for model in self.models:
            if model.indomain(T, P): return model.differentiate_by_T(T, P)
        # return model.differentiate_by_T(T, P) 
        raise ValueError(f"no valid model at T={T:.2f} K and P={P:5g} Pa")
            
    def differentiate_by_P(self, T, P=101325.):
        for model in self.models:
             if model.indomain(T, P): return model.differentiate_by_P(T, P)
        # return model.differentiate_by_P(T, P) 
        raise ValueError(f"no valid model at T={T:.2f} K and P={P:5g} Pa")

    def integrate_by_T(self, Ta, Tb, P=101325.):
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
        # return integral + model.integrate_by_T(Ta, Tb, P) 
        raise ValueError(f"no valid model between T={Ta:.2f} to {Tb:.2f} K at P={P:5g} Pa")
    
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
        # return integral + model.integrate_by_P(Pa, Pb, T) 
        raise ValueError(f"no valid model between P={Pa:5g} to {Pb:5g} Pa ast T={T:.2f}")
    
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
        # return integral + model.integrate_by_T_over_T(Ta, Tb, P) 
        raise ValueError(f"no valid model between T={Ta:.2f} to {Tb:.2f} K")
        
            
    