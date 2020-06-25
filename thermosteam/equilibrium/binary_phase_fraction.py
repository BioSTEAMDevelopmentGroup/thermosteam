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
import flexsolve as flx

__all__ = ('phase_fraction', 'solve_phase_fraction',
           'compute_phase_fraction_2N', 'compute_phase_fraction_3N')

@flx.njitable(cache=True)
def as_valid_fraction(x):
    if x < 0.:
        x = 0.
    elif x > 1.:
        x = 1.
    return x

def phase_fraction(zs, Ks, guess=None):
    """Return phase fraction for binary equilibrium."""
    N = zs.size
    if N == 2:
        phase_fraction = compute_phase_fraction_2N(zs, Ks)
    elif N == 3:
        phase_fraction = compute_phase_fraction_3N(zs, Ks)
    else:
        phase_fraction = solve_phase_fraction(zs, Ks, guess)
    return as_valid_fraction(phase_fraction)

def solve_phase_fraction(zs, Ks, guess):
    """
    Return phase fraction for N-component binary equilibrium through
    numerically solving an objective function.
    """
    args = (zs, Ks)
    f = phase_fraction_objective_function
    f_min = f(0., *args)
    f_max = f(1., *args)
    if f_min > f_max > 0.: return 1.
    if f_max < f_min < 0.: return 0.
    return flx.IQ_interpolation(f, 0., 1.,
                                f_min, f_max,
                                guess, 1e-16, 1e-16,
                                args, checkiter=False)

@flx.njitable(cache=True)
def phase_fraction_objective_function(V, zs, Ks):
    """Phase fraction objective function."""
    Kterm = Ks - 1.
    numerator = zs * Kterm
    denominator = 1. + V * Kterm
    denominator[denominator < 1e-16] = 1e-16
    return (numerator / denominator).sum()    

@flx.njitable(cache=True)
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

@flx.njitable(cache=True)
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
    
    K12 = K1**2
    K22 = K2**2
    K32 = K3**2
    
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
    z12 = z1**2
    z22 = z2**2
    z32 = z3**2
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