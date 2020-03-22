# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:40:59 2019

@author: yoelr
"""
from flexsolve import wegstein

__all__ = ('solve_x', 'solve_y')

def x_iter(x, x_gamma_poyinting, gamma, poyinting, T):
    x = x/x.sum()  
    return x_gamma_poyinting / (gamma(x, T) * poyinting(x, T))

def solve_x(x_gamma_poyinting, gamma, poyinting, T, x_guess=None):
    if x_guess is None: x_guess = x_gamma_poyinting
    return wegstein(x_iter, x_guess, 1e-5, args=(x_gamma_poyinting, gamma, poyinting, T))

def y_iter(y, y_phi, phi, T, P):
    y_sum = y.sum()
    if y_sum > 1e-5: y /= y_sum
    return y_phi / phi(y, T, P)

def solve_y(y_phi, phi, T, P, y_guess):
    if y_guess is None: y_guess = y_phi
    return wegstein(y_iter, y_phi, 1e-5, args=(y_phi, phi, T, P))