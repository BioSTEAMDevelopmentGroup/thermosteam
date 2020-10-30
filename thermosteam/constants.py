# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from math import pi

__all__ = ('R', 'k', 'g', 'pi', 'N_A', 'calorie', 'epsilon_0')


# Obtained from SciPy 0.19 (2014 CODATA)
# Included here so calculations are consistent across SciPy versions

#: Universal gas constant [J/mol/K]
R = 8.3144598

#: Boltzman constant [m^2 * kg * s^-2 * K^-1]
k = 1.38064852e-23

#: Avogadro's number
N_A = 6.022140857e+23

#: Conversion factor [J/calorie]
calorie = 4.184

#: Permittivity of a vacuum [C/V/m]
epsilon_0 = 8.854187817620389e-12

#: Acceleration due to gravity [m/s^2]
g = 9.80665

#: Planck's constant [m^2 * kg / s]
h = 6.62607004e-34

#: Speed of light in a vacuum [m /2]
c = 299792458