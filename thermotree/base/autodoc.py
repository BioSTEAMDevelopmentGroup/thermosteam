# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:06:46 2019

@author: yoelr
"""
from .units_of_measure import units_of_measure, definitions, types

# %% Utilities

class VariableDescriber:
    __slots__ = ('defs', 'units', 'types')
    
    def __init__(self, definitions, units_of_measure, types):
        self.defs = definitions
        self.units = units_of_measure
        self.types = types

    def get_output(self, var):
        definition = self.get_def(var)
        units = self.get_units(var)
        output = definition.lower() + f" ({var})" if definition else var
        if units: output += f" in {units}"
        return output

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
            info += '.'
        return info

describer = VariableDescriber(definitions, units_of_measure, types)

# %% Autodoc

def autodoc_functor(functor, base, math, refs):
    f = VariableDescriber(functor.definitions, functor.units_of_measure,
                          functor.types)
    out = f.get_output(functor.var)
    header = f"Create a {base.__name__}.{functor.__name__} object that returns the {out}.\n" + math + '\n'
    params = functor.params
    if params:
        parameters = ("Parameters\n"
                      "----------\n")
        for p in params:
            parameters += (f"{p} : {f.get_type(p)}\n"
                           f"{f.describe(p)}\n")
        parameters += '\n'
    else:
        parameters = ""
    if refs:
        references = ("References\n"
                      "----------\n") + refs
    else:
        references = ""
    functor.__doc__ = header + parameters + references
    
    
    
