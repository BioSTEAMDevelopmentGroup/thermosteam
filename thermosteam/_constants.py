# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:09:05 2019

@author: yoelr
"""
from math import pi

__all__ = ('R', 'k', 'g', 'pi', 'N_A', 'calorie', 'epsilon_0')


# Obtained from SciPy 0.19 (2014 CODATA)
# Included here so calculations are consistent across SciPy versions

#: Universal gas constant [J/mol/K]
R = 8.3144598

#: Boltzman constant [m^2 * kg * s^-2 *K^-1]
k = 1.38064852e-23

#: Avogadro's number
N_A = 6.022140857e+23

#: Conversion factor [J/calorie]
calorie = 4.184

#: Permittivity of a vacuum [C/V/m]
epsilon_0 = 8.854187817620389e-12

#: Acceleration due to gravity [m/s^2]
g = 9.80665