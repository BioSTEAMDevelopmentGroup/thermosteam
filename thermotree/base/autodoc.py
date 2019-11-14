# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:06:46 2019

@author: yoelr
"""
from .units_of_measure import units_of_measure, definitions, types

# %% Utilities

class FunctorVariableDescriber:
    __slots__ = ('defs', 'units', 'types', 'output')
    def __init__(self, functor):
        self.defs = functor.definitions
        self.units = functor.units_of_measure
        self.types = functor.types
        var = functor.var
        definition = self.get_def(var)
        units = self.get_units(var)
        output = definition.lower() + f" ({var})" if definition else var
        if units: output += f" in {units}"
        self.output = output

    def get_def(self, var):
        return self.defs.get(var, "") or definitions.get(var, "")
    
    def get_units(self, var):
        return self.units.get(var, "") or units_of_measure.get(var, "")
    
    def get_type(self, var):
        return self.types.get(var) or 'float'

    def describe(self, var):
        info = self.get_def(var)
        if info:
            units = self.get_units(var)
            if units: info += f" in {units}"
        return info
    

# %% Autodoc

def autodoc_functor(functor, base, math, refs):
    f = FunctorVariableDescriber(functor)
    header = f"Create a {base.__name__}.{functor.__name__} object that returns the {f.output}.\n" + math + '\n'
    params = functor.params
    if params:
        parameters = ("Parameters\n"
                      "----------\n")
        for p in params:
            parameters += (f"{p} : {f.get_type(p)}\n"
                           f"{f.describe(p)}.\n")
        parameters += '\n'
    else:
        parameters = ""
    if refs:
        references = ("References\n"
                      "----------\n") + refs
    else:
        references = ""
    functor.__doc__ = header + parameters + references
    
    
    
