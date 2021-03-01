# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import flexsolve as flx
from .. import functional as fn
from ..exceptions import InfeasibleRegion
from .fugacity_coefficients import IdealFugacityCoefficients
import numpy as np

__all__ = ('solve_x', 'solve_y')

def x_iter(x, x_gamma_poyinting, gamma, poyinting, T):
    x = fn.normalize(x)
    # Add back trace amounts for activity coefficients at infinite dilution
    mask = x < 1e-32
    x[mask] = 1e-32
    denominator = gamma(x, T) * poyinting(x, T)
    try:
        x = x_gamma_poyinting / denominator
    except FloatingPointError: 
        raise InfeasibleRegion('liquid phase composition')
    if (np.abs(x) > 1e16).any():
        raise InfeasibleRegion('liquid phase composition')
    return x

def solve_x(x_gamma_poyinting, gamma, poyinting, T, x_guess):
    args = (x_gamma_poyinting, gamma, poyinting, T)
    try:
        x = flx.wegstein(x_iter, x_guess, 1e-10, args=args, checkiter=False,
                         checkconvergence=False, convergenceiter=3)
    except InfeasibleRegion:
        x = x_gamma_poyinting
    return x
        
def y_iter(y, y_phi, phi, T, P):
    y = fn.normalize(y)
    return y_phi / phi(y, T, P)

def solve_y(y_phi, phi, T, P, y_guess):
    if isinstance(phi, IdealFugacityCoefficients): return y_phi
    return flx.wegstein(y_iter, y_phi, 1e-9, args=(y_phi, phi, T, P), 
                        checkiter=False,
                        checkconvergence=False, 
                        convergenceiter=3)