# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# A significant portion of this module originates from:
# Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
# Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>
# 
# This module is under a dual license:
# 1. The UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
# 
# 2. The MIT open-source license. See
# https://github.com/CalebBell/thermo/blob/master/LICENSE.txt for details.
"""
"""
import numpy as np
import flexsolve as flx
from numba import njit

__all__ = ('phase_fraction', 
           'solve_phase_fraction_Rashford_Rice',
           'phase_composition',
           'compute_phase_fraction_2N', 
           'compute_phase_fraction_3N')

@njit(cache=True)
def as_valid_fraction(x):
    """Ensure that x is between 0 and 1."""
    if x < 0.:
        x = 0.
    elif x > 1.:
        x = 1.
    return x

# @njit(cache=True)
def phase_fraction(zs, Ks, guess=None, za=0., zb=0.):
    """Return phase fraction for binary phase equilibrium."""
    N = zs.size
    if za or zb or N > 3:
        phase_fraction = solve_phase_fraction_Rashford_Rice(zs, Ks, guess, za, zb)
    elif N == 2:
        phase_fraction = compute_phase_fraction_2N(zs, Ks)
    elif N == 3:
        phase_fraction = compute_phase_fraction_3N(zs, Ks)
    else:
        raise ValueError('number of chemicals in equilibrium must be 2 or more '
                         'to find phase fraction')
    return as_valid_fraction(phase_fraction)

@njit(cache=True)
def phase_composition(zs, Ks, phi):
    return zs * Ks / (phi * Ks + (1. - phi))

# @njit(cache=True)
def solve_phase_fraction_Rashford_Rice(zs, Ks, guess, za, zb):
    """
    Return phase fraction for N-component binary equilibrium by
    numerically solving the Rashford Rice equation.
    
    """
    if Ks.max() < 1.0 and not za: return 0.
    if Ks.min() > 1.0 and not zb: return 1.
    K_minus_1 = Ks - 1.
    args = (- zs * K_minus_1, K_minus_1, za, zb)
    f = phase_fraction_objective_function
    x0 = 0.
    x1 = 1.
    y0 = -np.inf if za else f(x0, *args) 
    y1 = np.inf if zb else f(x1, *args)
    if y0 > y1 > 0.: return 1
    if y1 > y0 > 0.: return 0.
    if y0 < y1 < 0.: return 1.
    if y1 < y0 < 0.: return 0.
    x0, x1, y0, y1 = flx.find_bracket(f, x0, x1, y0, y1, args, tol=5e-8)
    if abs(x1 - x0) < 1e-6: return (x0 + x1) / 2.
    return flx.IQ_interpolation(f, x0, x1, y0, y1,
                                guess, 1e-16, 1e-16,
                                args, checkiter=False)

@njit(cache=True)
def phase_fraction_objective_function(phi, negative_zs_K_minus_1, K_minus_1, za, zb):
    """Phase fraction objective function."""
    denominator = 1. + phi * K_minus_1
    a = za/phi if za > 0. else 0.
    b = zb/(1. - phi) if zb > 0. else 0.
    phi = (negative_zs_K_minus_1 / denominator).sum()
    return phi - a + b

@njit(cache=True)
def compute_phase_fraction_2N(zs, Ks):
    """Return phase fraction for 2-component binary equilibrium."""
    z1, z2 = zs
    K1, K2 = Ks
    K1z1 = K1*z1
    K1z2 = K1*z2
    K2z1 = K2*z1
    K2z2 = K2*z2
    K1K2 = K1*K2
    K1K2z1 = K1K2*z1
    K1K2z2 = K1K2*z2
    z1_z2 = z1 + z2
    K1z1_K2z2 = K1z1 + K2z2
    return (-K1z1_K2z2 + z1_z2)/(K1K2z1 + K1K2z2 - K1z2 - K1z1_K2z2 - K2z1 + z1_z2)

@njit(cache=True)
def compute_phase_fraction_3N(zs, Ks):
    """Return phase fraction for 3-component binary equilibrium."""
    z1, z2, z3 = zs
    K1, K2, K3 = Ks
    
    K1z1 = K1*z1
    K1z2 = K1*z2
    K1z3 = K1*z3
    
    K2z1 = K2*z1
    K2z2 = K2*z2
    K2z3 = K2*z3
    
    K3z1 = K3*z1
    K3z2 = K3*z2
    K3z3 = K3*z3
    
    K1K2 = K1*K2
    K1K3 = K1*K3
    K2K3 = K2*K3
    
    K1K2z1 = K1K2*z1
    K1K2z2 = K1K2*z2
    K1K3z1 = K1K3*z1
    K1K3z3 = K1K3*z3
    K2K3z2 = K2K3*z2
    K2K3z3 = K2K3*z3
    
    K12 = K1 * K1
    K22 = K2 * K2
    K32 = K3 * K3
    
    K12K2 = K12*K2
    K12K3 = K12*K3
    K12K22 = K12*K22
    K12K2K3 = K12*K2*K3
    K1K22 = K1*K22
    K1K22K3 = K1K22*K3
    
    K1K2K3 = K1K2*K3
    K22K3 = K22*K3
    K1K32 = K1*K32
    K2K32 = K2*K32
    K22K32 = K22*K32
    K12K32 = K12K3*K3
    
    
    z1_z2_z3 = z1 + z2 + z3
    z12 = z1 * z1
    z22 = z2 * z2
    z32 = z3 * z3
    z1z2 = z1*z2
    z1z3 = z1*z3
    z2z3 = z2*z3
    
    K1K2K32 = K1K2K3*K3
    K12K2K3 = K12K2*K3
    K12K22z12 = K12K22*z12
    K12K22z1z2 = K12K22*z1z2
    K12K22z22 = K12K22*z22
    K12K2K3z12 = K12K2K3*z12
    K12K2K3z1z2 = K12K2K3*z1z2
    K12K2K3z1z3 = K12K2K3*z1z3
    K12K2K3z2z3 = K12K2K3*z2z3
    K12K2z1z2 = K12K2*z1z2
    K12K2z1z3 = K12K2*z1z3
    K12K2z22 = K12K2*z22
    K12K2z2z3 = K12K2*z2z3
    K12K32z12 = K12K32*z12
    K12K32z1z3 = K12K32*z1z3
    K12K32z32 = K12K32*z32
    K12K3z1z2 = K12K3*z1z2
    K12K3z1z3 = K12K3*z1z3
    K12K3z2z3 = K12K3*z2z3
    K12K3z32 = K12K3*z32
    K12z2z3 = K12*z2z3
    K12z22 = K12*z22
    K12z32 = K12*z32
    K1K22K3z1z2 = K1K22K3*z1z2
    K1K22K3z1z3 = K1K22K3*z1z3
    K1K22K3z22 = K1K22K3*z22
    K1K22K3z2z3 = K1K22K3*z2z3
    K1K22z12 = K1K22*z12
    K1K22z1z2 = K1K22*z1z2
    K1K22z1z3 = K1K22*z1z3
    K1K22z2z3 = K1K22*z2z3
    K1K2K32z1z2 = K1K2K32*z1z2
    K1K2K32z1z3 = K1K2K32*z1z3
    K1K2K32z2z3 = K1K2K32*z2z3
    K1K2K32z32 = K1K2K32*z32
    K1K2K3z12 = K1K2K3*z12
    K1K2K3z1z2 = K1K2K3*z1z2
    K1K2K3z1z3 = K1K2K3*z1z3
    K1K2K3z22 = K1K2K3*z22
    K1K2K3z2z3 = K1K2K3*z2z3
    K1K2K3z32 = K1K2K3*z32
    K1K2z1z2 = K1K2*z1z2
    K1K2z1z3 = K1K2*z1z3
    K1K2z2z3 = K1K2*z2z3
    K1K2z32 = K1K2*z32
    K1K32z12 = K1K32*z12
    K1K2z1z3 = K1K2*z1z3
    K1K32z1z2 = K1K32*z1z2
    K1K32z1z3 = K1K32*z1z3
    K1K32z2z3 = K1K32*z2z3
    K1K3z1z2 = K1K3*z1z2
    K1K3z1z3 = K1K3*z1z3
    K1K3z22 = K1K3*z22
    K1K3z2z3 = K1K3*z2z3
    K22K32z22 = K22K32*z22
    K22K3z1z2 = K22K3*z1z2
    K22K32z2z3 = K22K32*z2z3
    K22K32z32 = K22K32*z32
    return ((-K1K2z1/2 - K1K2z2/2 - K1K3z1/2 - K1K3z3/2 + K1z1
              + K1z2/2 + K1z3/2 - K2K3z2/2 - K2K3z3/2 + K2z1/2 + K2z2
              + K2z3/2 + K3z1/2 + K3z2/2 + K3z3 - z1_z2_z3
              - (K12K22z12 + 2*K12K22z1z2 + K12K22z22
                - 2*K12K2K3z12 - 2*K12K2K3z1z2 - 2*K12K2K3z1z3
                + 2*K12K2K3z2z3 - 2*K12K2z1z2 + 2*K12K2z1z3
                - 2*K12K2z22 - 2*K12K2z2z3 + K12K32z12
                + 2*K12K32z1z3 + K12K32z32 + 2*K12K3z1z2
                - 2*K12K3z1z3 - 2*K12K3z2z3 - 2*K12K3z32
                + K12z22 + 2*K12z2z3 + K12z32
                - 2*K1K22K3z1z2 + 2*K1K22K3z1z3 - 2*K1K22K3z22
                - 2*K1K22K3z2z3 - 2*K1K22z12 - 2*K1K22z1z2
                - 2*K1K22z1z3 + 2*K1K22z2z3 + 2*K1K2K32z1z2
                - 2*K1K2K32z1z3 - 2*K1K2K32z2z3 - 2*K1K2K32z32
                + 4*K1K2K3z12 + 4*K1K2K3z1z2 + 4*K1K2K3z1z3
                + 4*K1K2K3z22 + 4*K1K2K3z2z3 + 4*K1K2K3z32
                + 2*K1K2z1z2 - 2*K1K2z1z3 - 2*K1K2z2z3 - 2*K1K2z32
                - 2*K1K32z12 - 2*K1K32z1z2 - 2*K1K32z1z3
                + 2*K1K32z2z3 - 2*K1K3z1z2 + 2*K1K3z1z3 - 2*K1K3z22
                - 2*K1K3z2z3 + K22K32z22 + 2*K22K32z2z3
                + K22K32z32 + 2*K22K3z1z2 - 2*K22K3*z1z3
                - 2*K22K3*z2z3 - 2*K22K3*z32 + K22*z12
                + 2*K22*z1z3 + K22*z32 - 2*K2K32*z1z2
                + 2*K2K32*z1z3 - 2*K2K32*z22 - 2*K2K32*z2z3
                - 2*K2K3*z12 - 2*K2K3*z1z2 - 2*K2K3*z1z3
                + 2*K2K3*z2z3 + K32*z12 + 2*K32*z1z2 + K32*z22)**0.5/2)
                / (K1K2K3*z1 + K1K2K3*z2 + K1K2K3*z3 - K1K2*z1 - K1K2*z2
                    - K1K2*z3 - K1K3*z1 - K1K3*z2 - K1K3*z3 + K1z1 + K1z2
                    + K1z3 - K2K3*z1 - K2K3*z2 - K2K3*z3 + K2z1 + K2z2 + K2z3
                    + K3z1 + K3z2 + K3z3 - z1_z2_z3))

# # @njit(cache=True)
# def solve_phase_fraction_iteration(zs, Ks, guess=0.5, za=0., zb=0.):
#     """
#     Return phase fraction for N-component binary phase equilibrium by 
#     accelerated fixed-point iteration. 
    
#     Notes
#     -----
#     This iterative method was developed by Yoel Cortes-Pena. It can handle 
#     chemicals which do not partition. za and zb are the fraction of 
#     non-partitioning chemicals in phases a and b, respectively. 
    
#     Examples
#     --------
#     Find the vapor fraction of a mixture of CO2, air, and water. Assume that 
#     air is always a gas, water is always a liquid, but CO2 can partition in both.
        
#     >>> import numpy as np
#     >>> from thermosteam.equilibrium import (
#     ...     solve_phase_fraction_iteration,
#     ... )
#     >>> F_air = 1
#     >>> F_water = 1
#     >>> F_CO2 = 1
#     >>> F_total = F_air + F_water + F_CO2
#     >>> z_air = F_air / F_total
#     >>> z_water = F_water / F_total
#     >>> zs = np.array([0.333]) # CO2
#     >>> Ks = np.array([0.999]) # CO2
#     >>> phi = solve_phase_fraction_iteration(
#     ...     zs, Ks, za=z_air, zb=z_water
#     ... )
#     >>> phi
#     0.4998
    
#     """
#     if Ks.max() < 1.0 and not za: return 0.
#     if Ks.min() > 1.0 and not zb: return 1.
#     args = (zs, Ks, za, zb)
#     x0 = 0.
#     x1 = 1.
#     f = compute_phase_fraction_iter
#     y0 = -np.inf if za else f(x0, *args) 
#     y1 = np.inf if zb else f(x1, *args)
#     if y0 > y1 > 0.: return 1
#     if y1 > y0 > 0.: return 0.
#     if y0 < y1 < 0.: return 1.
#     if y1 < y0 < 0.: return 0.
#     if not 0. < guess < 1.: guess = 0.5
#     phi = np.array([guess, 1. - guess])
#     zc = np.ones([2, 1]); zc[:, 0] = [za, zb]
#     N = zs.shape[0]
#     Ks_2d = np.ones([2, N])
#     zs_2d = np.ones([1, N])
#     zs_2d[0, :] = zs
#     Ks_2d[0, :] = Ks
#     Ks_2d[1, :] = 1. / Ks
#     phi = flx.wegstein(f, phi, 1e-16, 
#                        args=(zs_2d, Ks_2d, zc), checkiter=False)
#     return phi[0] / phi.sum()

# @njit(cache=True)
# def compute_phase_fraction_iter(phi, zs, Ks, zc):
#     ys = phase_composition(zs, Ks, phi)
#     new_phi = np.zeros(phi.shape)
#     shape = Ks.shape
#     M = shape[0]
#     N = shape[1]
#     for i in range(M):
#         isum = 0.
#         for j in range(N): isum += ys[i, j]
#         new_phi[i] = isum * phi[i] + zc[i, 0]
#     return new_phi
