# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from chemicals import *
from fluids.core import Pr, alpha
from numba import njit
from thermosteam.base import functor
import numpy as np

@functor
def horner(T, coeffs):
    tot = 0
    for c in coeffs: tot = tot * T + c
    return tot 

@njit(cache=True)
def normalize(array, sum_array=None, minimum=1e-16):
    """
    Return a normalized array to a magnitude of 1.
    If magnitude is zero, all fractions will have equal value.
    """
    if sum_array is None: sum_array = array.sum()
    if sum_array < minimum:
        size = array.size
        return np.ones(size)/size
    else:
        return array/sum_array
    
def first_true_index(x):
    """
    Return index of first true value.
    
    """
    for index, i in enumerate(x):
        if i: return index
    
@njit(cache=True)
def mixing_simple(z, y):
    r'''
    Return a weighted average of `y` given the weights, `z`.
    
    Examples
    --------
    >>> import numpy as np
    >>> mixing_simple(np.array([0.1, 0.9]), np.array([0.01, 0.02]))
    0.019000000000000003
    
    '''
    return (z * y).sum()

@njit(cache=True)
def mixing_logarithmic(z, y):
    r'''
    Return the logarithmic weighted average `y` given weights, `z`.
    
    .. math::
    
        y = \sum_i z_i \cdot \log(y_i)
    
    Notes
    -----
    Does not work on negative values.
    Examples
    --------
    >>> import numpy as np
    >>> mixing_logarithmic(np.array([0.1, 0.9]), np.array([0.01, 0.02]))
    0.01866065983073615
    
    '''
    return np.exp((z*np.log(y)).sum())

@njit(cache=True)
def mu_to_nu(mu, rho):
    r"""
    Return the kinematic viscosity (nu) given the dynamic viscosity (mu) and 
    density (rho).
    
    .. math::
    
        \nu = \frac{\mu}{\rho}
    
    Examples
    --------
    >>> mu_to_nu(0.000998, 998.)
    1.0e-06
    
    """
    return mu/rho

@njit(cache=True)
def V_to_rho(V, MW):
    r'''
    Return the density (rho) in kg/m^3 given the molar volume (V) in
    m^3/mol and molecular weight (MW) in g/mol.
    
    .. math::
        \rho = \frac{MW}{1000\cdot V}
    
    Parameters
    ----------
    V : float
        Molar volume, [m^3/mol]
    MW : float
        Molecular weight, [g/mol]
    
    Returns
    -------
    rho : float
        Density, [kg/m^3]
    
    Examples
    --------
    >>> V_to_rho(0.000132, 86.18)
    652.878...
    
    '''
    return MW/V/1000.

@njit(cache=True)
def rho_to_V(rho, MW):
    r'''
    Return the molar volume (V) in m^3/mol given the density (rho) in
    kg/m^3 and molecular weight (MW) in g/mol.
    
    .. math::
        V = \left(\frac{1000 \rho}{MW}\right)^{-1}
    
    Parameters
    ----------
    rho : float
        Density, [kg/m^3]
    MW : float
        Molecular weight, [g/mol]
    
    Returns
    -------
    V : float
        Molar volume, [m^3/mol]
    
    Examples
    --------
    >>> rho_to_V(652.9, 86.18)
    0.0001319957...
    
    '''
    return MW/rho/1000.

@njit(cache=True)
def remove_negligible_negative_values(material: np.ndarray):
    negative = material < 0.
    if negative.any():
        material_sum = np.abs(material).sum()
        if material_sum > 1e-16:
            negligible = material / material_sum > -1e-16
            material[negative & negligible] = 0. 
        else:
            material[negative] = 0. 

del njit