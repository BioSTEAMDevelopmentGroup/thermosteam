# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 04:26:20 2019

@author: yoelr
"""
from .units_of_measure import units_of_measure
from .utils import var_with_units, get_obj_values
from inspect import signature, isclass
from numba.targets.registry import CPUDispatcher
from numba import njit
import numpy as np

__all__ = ("Functor", "MixtureFunctor",
           "TFunctor", "TPFunctor",
           "zTFunctor", "zTPFunctor",
           "functor", 'H', 'S', 'V', 'Cp',
           'Psat', 'Hvap', 'display_asfunctor',
           'functor_matching_params', 'functor_base_and_params')

RegisteredArgs = set()
RegisteredFunctors = []
function_functors = {}

# %% Utilities

def functor_name(functor):
    return functor.__name__ if hasattr(functor, "__name__") else type(functor).__name__

def display_asfunctor(functor, var=None, name=None, show_var=True):
    name = name or functor_name(functor)
    info = f"{name}{str(signature(functor)).replace('self, ', '')}"
    var = var or (functor.var if hasattr(functor, 'var') else None)
    units = functor.units_of_measure if hasattr(functor, "units_of_measure") else units_of_measure
    if var:
        if show_var:
            info += f" -> " + var_with_units(var, units)
        else:
            u = units.get(var)
            if u: info += f" -> {u}"
    return info

def functor_matching_params(params):
    N_total = len(params)
    for base in RegisteredFunctors:
        args = base._args
        N = len(args)
        if N > N_total: continue
        if args == params[:N]: return base
    raise ValueError("could not match function signature to registered functors")

def functor_base_and_params(function):
    params = tuple(signature(function).parameters)
    base = functor_matching_params(params)
    return base, params[len(base._args):]


# %% Interfaces

def functor(function=None, data=None, kwargs=None, var=None,
            njitcompile=True, wrap=None):
    if function:
        cls = function if isclass(function) else factory(function, var, njitcompile, wrap)
        return cls(data, kwargs) if data else cls
    else:
        return lambda function: factory(function, var, njitcompile, wrap)
  
def factory(function=None, var=None, njitcompile=True,
            wrap=None):
    if function:
        if (function in function_functors): return function_functors[function]
        base, params = functor_base_and_params(function)
        if njitcompile and not isinstance(function, CPUDispatcher): 
            function = njit(function)
        dct = {'__slots__': (),
               'function': staticmethod(function),
               'params': params,
               'var': var}
        function_functors[function] = cls = type(function.__name__, (base,), dct)
        if wrap: cls.wrapper(wrap)
        return cls
    else:
        return lambda function: factory(function, var, njitcompile, wrap)


# %% Decorators
    
class FunctorFactory:
   
    def __init__(self, var):
        self.var = var
    
    def __call__(self, function=None, data=None, kwargs=None, njitcompile=True, wrap=None):
        return functor(function, data, kwargs, self.var, njitcompile, wrap)
    
    def s(self, function=None, data=None, kwargs=None, njitcompile=True, wrap=None):
        return functor(function, data, kwargs, self.var + '.s', njitcompile, wrap)
    
    def l(self, function=None, data=None, kwargs=None, njitcompile=True, wrap=None):
        return functor(function, data, kwargs, self.var + '.l', njitcompile, wrap)
    
    def g(self, function=None, data=None, kwargs=None, njitcompile=True, wrap=None):
        return functor(function, data, kwargs, self.var + '.g', njitcompile, wrap)
    
    def __repr__(self):
        return f"{type(self).__name__}: {var_with_units(self.var)}"
    
H, S, Cp, V, k, mu, Psat, Hvap = [FunctorFactory(i) for i in
                                 ('H', 'S', 'Cp', 'V', 'k', 'mu', 'Psat', 'Hvap')]


# %% Functors

class Functor: 
    __slots__ = ()
    units_of_measure = units_of_measure
    
    def __init_subclass__(cls, args=None, before=None, after=None):
        if args and not hasattr(cls, 'function'):
            args = tuple(args)
            assert args not in RegisteredArgs, f"abstract functor with args={args} already implemented"
            if before:
                index = RegisteredFunctors.index(before)
            elif after:
                index = RegisteredFunctors.index(after) + 1
            else:
                index = 0
            RegisteredFunctors.insert(index, cls)
            RegisteredArgs.add(args)
            cls._args = args
    
    def __repr__(self):
        return f"<[Functor] {display_asfunctor(self)}>"


class PureComponentFunctor(Functor):
    __slots__ = ('kwargs', 'data')
    
    def __init__(self, data, kwargs=None):
        if isinstance(data, dict):
            self.data = data
        elif hasattr(data, '__iter__'):
            self.data = data = dict(zip(self.params, data))
        else:
            self.data = data = dict(zip(self.params, get_obj_values(data, self.params)))
        self.kwargs = kwargs or (self.wrap(**data) if hasattr(self, 'wrap') else data)
    
    @classmethod
    def wrapper(cls, kwargs):
        cls.params = tuple(signature(kwargs).parameters)
        cls.wrap = staticmethod(kwargs)
        return cls
    
    def show(self):
        info = f"Functor: {display_asfunctor(self)}"
        data = self.data
        units = self.units_of_measure
        for key, value in data.items():
            if value is None:
                info += f"\n {key}: {value}"
                continue
            else:
                try:
                    info += f"\n {key}: {value:.5g}"
                except:
                    info += f"\n {key}: {value}"    
                else:
                    units = units_of_measure.get(key, "")
                    if units: info += ' ' + units
        print(info)
        
    _ipython_display_ = show


class TFunctor(PureComponentFunctor, args=('T',)):
    __slots__ = ()
    
    def __call__(self, T, P=None):
        return self.function(T, **self.kwargs)


class TIntegralFunctor(PureComponentFunctor, args=('Ta', 'Tb')):
    __slots__ = ()
    
    def __call__(self, Ta, Tb, P=None):
        return self.function(Ta, Tb, **self.kwargs)

class PIntegralFunctor(PureComponentFunctor, args=('Pa', 'Pb', 'T')):
    __slots__ = ()
    
    def __call__(self, Pa, Pb, T):
        return self.function(Pa, Pb, T, **self.kwargs)


class TPFunctor(PureComponentFunctor, args=('T', 'P')):
    __slots__ = ()
    
    def __call__(self, T, P):
        return self.function(T, P, **self.kwargs)
    

class IdealMixtureFunctor(Functor):
    __slots__ = ('name', 'species', 'cached')
    
    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.TP = (0., 0.)
        self.data = None
    
    def __call__(self, z, T, P):
        if (T, P) != self.TP:
            attr = getattr
            name = self.name
            self.data = np.array([attr(i, name)(T, P) for i in self.species], dtype=float)
        return z * self.data

    def show(self):
        print(f"Functor: {display_asfunctor(self)}"
              f" species: {', '.join([i.ID for i in self.species])}")


class MixtureFunctor(Functor):
    __slots__ = ('kwargs', '_species')
    
    def __init__(self, species, kwargs=None):
        species = tuple(species)
        if kwargs:
            self._species = species
            self.kwargs = kwargs
        else:
            self.species = species
    
    @classmethod
    def wrap(cls, kwargs):
        cls.calculate_kwargs = staticmethod(kwargs)
        return cls
    
    @property
    def species(self):
        return self._species
    
    @species.setter
    def species(self, species):
        if species == self._species: return
        self._species = species = tuple(species)
        if species in self.cached:
            self.kwargs = self.cached[species]
        else:
            self.cached[species] = self.kwargs = self.calculate_kwargs(self.species)
        
    show = IdealMixtureFunctor.show
        
    _ipython_display_ = show
            
    
class zTFunctor(MixtureFunctor, args=('z', 'T')): 
    __slots__ = ()
    
    def __init_subclass__(self):
        self.cached = {}
    
    def __call__(self, z, T, P=None):
        return self.function(z, T, **self.kwargs)
        

class zTPFunctor(MixtureFunctor, args=('z', 'T', 'P')):
    __slots__ = ()
    
    def __init_subclass__(self):
        self.cached = {}
    
    def __call__(self, z, T, P):
        return self.function(z, T, P, **self.kwargs)
    
    