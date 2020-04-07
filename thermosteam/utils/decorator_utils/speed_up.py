# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:25:27 2020

@author: yoelr
"""
from numba import njit
import sys

__all__ = ('njitable', 'speed_up')

#: All njitable functions
njitables = []

def njitable(f):
    """
    Decorate function as 'njitable'. All 'njitable' functions must be able to 
    be compiled by Numba's njit decorator.
    """
    njitables.append(f)
    return f

def speed_up():
    """
    Speed up simulations by jit compiling all functions registered as 'njitable'.
    
    See also
    --------
    njitable
    
    Notes
    -----
    Several computation-heavy functions in Thermosteam and BioSTEAM are already marked as 'njitable'.
    This function serves to cut down the time required to perform Monte Carlo analysis.
    
    """
    for i in njitables:
        module = sys.modules[i.__module__]
        setattr(module, i.__name__, njit(i))
    njitables.clear()