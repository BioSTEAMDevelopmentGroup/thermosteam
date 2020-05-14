# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:40:59 2019

@author: yoelr
"""
import flexsolve as flx
from ..functional import normalize
from .fugacity_coefficients import IdealFugacityCoefficients
import numpy as np

__all__ = ('solve_x', 'solve_y')

def x_iter(x, x_gamma_poyinting, gamma, poyinting, T):
    x = normalize(x)
    # Add back trace amounts for activity coefficients at infinite dilution
    mask = x < 1e-16
    x[mask] = 1e-16
    denominator = gamma(x, T) * poyinting(x, T)
    try:
        x = x_gamma_poyinting / denominator
    except FloatingPointError: 
        raise flx.InfeasibleRegion('liquid phase composition is infeasible')
    if not np.isfinite(x).all():
        raise flx.InfeasibleRegion('liquid phase composition is infeasible')
    return x

def solve_x(x_gamma_poyinting, gamma, poyinting, T, x_guess):
    if x_guess is None: x_guess = x_gamma_poyinting
    args = (x_gamma_poyinting, gamma, poyinting, T)
    try:
        x = flx.aitken(x_iter, x_guess, 1e-8, args=args)
    except flx.SolverError as solver_error:
        try: x = flx.fixed_point(x_iter, solver_error.x, 1e-8, args=args)
        except: x = x_gamma_poyinting
    except flx.InfeasibleRegion:
        x = x_gamma_poyinting
    return x
        
def y_iter(y, y_phi, phi, T, P):
    y = normalize(y)
    return y_phi / phi(y, T, P)

def solve_y(y_phi, phi, T, P, y_guess):
    if isinstance(phi, IdealFugacityCoefficients): return y_phi
    if y_guess is None: y_guess = y_phi
    return flx.aitken(y_iter, y_phi, 1e-8, args=(y_phi, phi, T, P))