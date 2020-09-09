# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from ..units_of_measure import chemical_units_of_measure, definitions, types

__all__ = ('Documenter', 'autodoc_functor')

# %% Utilities

class Documenter: # pragma: no cover
    __slots__ = ('units_of_measure', 'definitions', 'types')
    
    def __init__(self, units_of_measure, definitions, types):
        self.units_of_measure = units_of_measure
        self.definitions = definitions
        self.types = types

    def get_definition(self, var):
        user_defs = self.definitions
        if user_defs:
            return user_defs.get(var, "") or definitions.get(var, "")
        else:
            return definitions.get(var, "")
    
    def get_units(self, var):
        var, *_ = var.split(".")
        units_of_measure = self.units_of_measure
        if units_of_measure:
            return self.units_of_measure.get(var) or chemical_units_of_measure.get(var)
        else:
            return chemical_units_of_measure.get(var)
    
    def get_type(self, var):
        user_types = self.types
        if user_types:
            return user_types.get(var) or types.get(var, 'float')
        else:
            return types.get(var, 'float')

    def describe_functor(self, functor, method, ref):
        var = functor.var
        if var:
            return_value = self.describe_return_value(var)
            description = (f"Create a {functor.kind} that estimates the "
                           f"{return_value} of a chemical")
            if method:
                description += f" using the {method} method"
            if ref:
                description += f", as described in {ref}"
        else:
            description = f"Create a {functor.kind}"
            if method:
                description += f" based on the {method} method"
            if ref:
                description += f", as described in {ref}"
        return description + "."

    def describe_parameter_type(self, var):
        return f"{var} : {self.get_type(var)}"
    
    def describe_all_parameters(self, vars, new_line):
        parameters = ("Parameters" + new_line 
                     +"----------" + new_line)
        sub_line = new_line + "    "
        defs = self.definitions or definitions
        defined_vars = []
        coefficients = []
        for var in vars:
            if var in defs: defined_vars.append(var)
            else: coefficients.append(var)
        for var in defined_vars:
            parameters += (self.describe_parameter_type(var) + sub_line
                         + self.describe_parameter(var) + '.' + new_line)
        if coefficients:
            N_coefficients = len(coefficients)
            if N_coefficients == 1:
                parameters += (coefficients[0] + " : float" + sub_line
                               + "Regressed coefficient." + new_line)
            else:
                parameters += (",".join(coefficients) + " : float" + sub_line
                               + "Regressed coefficients." + new_line)
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
            else: info += " [-]"
        return info


# %% Autodoc

def autodoc_functor(functor, doc='auto-merge', method=None, ref=None, tabs=1,
                    units_of_measure=None, definitions=None, types=None,
                    other_params=None): # pragma: no cover
    auto = merge = header = param = False
    if doc == 'auto-doc':
        auto = True
    elif doc == 'auto-merge':
        merge = True
    elif doc == 'auto-header': 
        header = True
    elif doc == 'auto-param':
        param = True
    else:
        raise ValueError("`doc` key-word argument must be either 'auto-doc', 'auto-merge', 'auto-header', or 'auto-param'")
    
    autodoc = Documenter(units_of_measure, definitions, types)
    function = functor.function
    header = autodoc.describe_functor(functor,
                                      method or function.__name__.replace('_', ' '),
                                      ref)
    function_doc = function.__doc__ or ""
    function.__doc__ = None
    new_line = "\n" + (tabs * 4) * " "
    params = functor.params
    if header:
        if other_params:
            params = list(params)
            for i in other_params: params.remove(i)
        parameters = autodoc.describe_all_parameters(params, new_line) if params else "" 
    
    if auto:
        functor.__doc__ = _join_sections(header, parameters, new_line)
        return
    elif merge:
        functor.__doc__ = _join_sections(header, parameters, new_line) + function_doc
    elif header:
        functor.__doc__ = header + new_line + function_doc    
    elif param:
        functor.__doc__ = function.__doc__.replace('[Parameters]\n', parameters + "\n")
    
def _join_sections(header, parameters, new_line): # pragma: no cover
    doc = header + new_line 
    if parameters:
        doc += new_line + parameters
    return doc