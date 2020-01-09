# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:06:46 2019

@author: yoelr
"""
from .units_of_measure import chemical_units_of_measure, definitions, types
from ..utils import MathString, MathSection
from ..exceptions import AutodocError

__all__ = ('Documenter', 'autodoc_functor')

# %% Utilities

class Documenter:
    __slots__ = ('definitions', 'units', 'types')
    
    def __init__(self, units_of_measure, definitions, types):
        self.units = units_of_measure
        self.definitions = definitions
        self.types = types

    def get_definition(self, var):
        return self.definitions.get(var, "") or definitions.get(var, "")
    
    def get_units(self, var):
        var, *_ = var.split(".")
        return self.units.get(var) or chemical_units_of_measure.get(var)
    
    def get_type(self, var):
        return self.types.get(var) or types.get(var) or 'float'

    def describe_functor(self, functor, equation, ref):
        var = functor.var
        if var:
            return_value = self.describe_return_value(var)
            description = f"Create a {functor.kind} that calcuates {return_value}"
            if equation:
                description += f" using the {equation} equation"
            if ref:
                description += f", as described in {ref}"
        else:
            description = f"Create a {functor.kind}"
            if equation:
                description += f"based on the {equation} equation"
            if ref:
                description += f", as described in {ref}"
        return description + "."

    def describe_parameter_type(self, var):
        return f"{var} : {self.get_type(var)}"
    
    def describe_all_parameters(self, vars, new_line):
        parameters = ("Parameters" + new_line 
                     +"----------" + new_line)
        sub_line = new_line + "    "
        definitions = self.definitions
        defined_vars = []
        coefficients = []
        for var in vars:
            if var in definitions: defined_vars.append(var)
            else: coefficients.append(var)
        for var in defined_vars:
            parameters += (self.describe_parameter_type(var) + sub_line
                         + self.describe_parameter(var) + '.' + new_line)
        if coefficients:
            parameters += (", ".join(coefficients) + " : float" + sub_line
                         + "Regressed coefficients.")
        return parameters
    
    def describe_return_value(self, var):
        definition = self.get_definition(var)
        units = self.get_units(var)
        if definition:
            info = definition.lower()
            info += f" ({var}; in {units})" if units else f" ({var})"
        else:
            info = f"{var} in {units}" if units else var
        return info

    def describe_parameter(self, var):
        info = self.get_definition(var)
        if info:
            units = self.get_units(var)
            if units: info += f" [{units}]"
            else: info += f" [-]"
        return info


# %% Autodoc

def autodoc_functor(functor, equation=None, ref=None, tabs=1):
    if not functor.var: return
    autodoc = Documenter(functor.units_of_measure, functor.definitions, functor.types)
    function = functor.function
    header = autodoc.describe_functor(functor,
                                      equation or function.__name__.replace('_', ' '),
                                      ref)
    params = functor.params
    new_line = "\n" + (tabs * 4) * " "
    parameters = autodoc.describe_all_parameters(params, new_line) if params else "" 
        
    doc = function.__doc__
    function.__doc__ = None
    if doc:
        if doc[:2] == '*\n':
            doc = _join_sections(header, parameters, new_line) + doc[1:]
        else:
            try:
                doc = doc.format(Header=header, Parameters=parameters)
            except Exception as error:
                raise AutodocError('formatting issue => '
                                  f'{type(error).__name__}: {error}')
    else:
        doc = _join_sections(header, parameters, new_line)
    functor.__doc__ = doc

def _join_sections(header, parameters, new_line):
    double_new_line = 2 * new_line
    doc = header + double_new_line 
    if parameters:
        doc += parameters + double_new_line
    return doc