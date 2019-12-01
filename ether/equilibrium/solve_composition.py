# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:40:59 2019

@author: yoelr
"""
from flexsolve import aitken

__all__ = ('solve_x', 'solve_y')

def x_error(x, x_gamma_poyinting, gamma, poyinting, T):
    x = x/x.sum()    
    return x_gamma_poyinting / (gamma(x, T) * poyinting(x, T))

def solve_x(x_gamma_poyinting, gamma, poyinting, T, x_guess=None):
    if x_guess is None: x_guess = x_gamma_poyinting
    return aitken(x_error, x_guess, 1e-5, args=(x_gamma_poyinting, gamma, poyinting, T))

def y_error(y, y_phi, phi, T, P):
    return y_phi / phi(y/y.sum(), T, P)

def solve_y(y_phi, phi, T, P, y_guess):
    if y_guess is None: y_guess = y_phi
    return aitken(y_error, y_guess, 1e-5, args=(y_phi, phi, T, P))