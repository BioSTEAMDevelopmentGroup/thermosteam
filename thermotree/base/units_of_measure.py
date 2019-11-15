# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:02:53 2019

@author: yoelr
"""
__all__ = ('units_of_measure',)

units_of_measure = {'MW': 'g/mol',
                    'T': 'K',
                    'Tm': 'K',
                    'Tb': 'K',
                    'Tt': 'K',
                    'Tc': 'K',
                    'P': 'Pa',
                    'Pc': 'Pa',
                    'Psat': 'Pa',
                    'Pt': 'K',
                    'V': 'm^3/mol',
                    'Vc': 'm^3/mol',
                    'Cp': 'J/mol/K',
                    'rho': 'kg/m^3', 
                    'rhoc': 'kg/m^3',
                    'nu': 'm^2/s',
                    'mu': 'Pa*s',
                    'sigma': 'N/m' ,
                    'k': 'W/m/K',
                    'alpha': 'm^2/s', 
                    'Hvap': 'J/mol',
                    'H': 'J/mol',  
                    'Hf':'J/mol', 
                    'Hfus': 'J/mol', 
                    'Hsub': 'J/mol',
                    'S': 'J/mol', 
                    'G': 'J/mol', 
                    'U': 'J/mol', 
                    'A': 'J/mol',
                    'H_excess': 'J/mol', 
                    'S_excess': 'J/mol',
                    'R': 'J/mol/K'
}

definitions = {'MW': 'Molecular weight',
               'T': 'Temperature',
               'Tm': 'Melting point temperature',
               'Tb': 'Boiling point temperature',
               'Tt': 'Triple point temperature',
               'Tc': 'Critical point temperature',
               'P': 'Pressure',
               'Pc': 'Critical point pressure',
               'Psat': 'Saturated vapor pressure',
               'Pt': 'Triple point pressure',
               'V': 'Molar volume',
               'Vc': 'Critical point volume',
               'Cp': 'Molar heat capacity',
               'Cp.s': 'Molar heat capacity of a solid',
               'Cp.l': 'Molar heat capacity of a liquid',
               'Cp.g': 'Molar heat capacity of a gas',
               'rho': 'Density',
               'rhoc': 'Critical point density',
               'nu': 'Kinematic viscosity',
               'mu': 'Hydrolic viscosity',
               'sigma': 'Surface tension',
               'k': 'Thermal conductivity',
               'alpha': 'Thermal diffusivity',
               'Hvap': 'Heat of vaporization',
               'H': 'Enthalpy',
               'Hf': 'Enthalpy of formation',
               'Hfus': 'Heat of fusion',
               'Hsub': 'Heat of sublimation',
               'S': 'Entropy',
               'G': 'Gibbs free energy',
               'U': 'Internal energy',
               'A': 'Helmholtz energy',
               'H_excess': 'Excess enthalpy',
               'S_excess': 'Excess entropy',
               'R': 'Universal gas constant',
               'Zc': 'Critical compressibility',
               'omega': 'Acentric factor',
}

types = {}
# Synonyms
for i, j in [('ω', 'omega')]:
    definitions[i] = definitions[j]
    if j in units_of_measure: units_of_measure[i] = units_of_measure[j]

# Phase properties
for var in ('Cp', 'H', 'S', 'V', 'k', 'H_excess', 'S_excess'):
    units = units_of_measure[var]
    definition = definitions[var].lower()
    for tag, phase in zip(('s', 'l', 'g'), ('Solid ', 'Liquid ', 'Gas ')):
        for tag in ('_' + tag, '.'+tag):
            phasevar = var + tag
            units_of_measure[phasevar] = units
            definitions[phasevar] = phase + definition
