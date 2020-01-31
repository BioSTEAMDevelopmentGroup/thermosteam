#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 08:58:41 2018

@author: Yoel Rene Cortes-Pena
"""
import numpy as np
from .unifac_data import DOUFSG, DOUFIP2016, UFIP, UFSG
from numba import njit

__all__ = ('ActivityCoefficients',
           'GroupActivityCoefficients',
           'DortmundActivityCoefficients',
           'UNIFACActivityCoefficiencts')

# %% Utilities

def chemgroup_array(chemgroups, index):
    M = len(chemgroups)
    N = len(index)
    array = np.zeros((M, N))
    for i, groups in enumerate(chemgroups):
        for group, count in groups.items():
            array[i, index[group]] = count
    return array

@njit
def group_activity_coefficients(x, chemgroups, loggammacs,
                                Qs, psis, cQfs, gpsis):
    weighted_counts = chemgroups.transpose() @ x
    Q_fractions = Qs * weighted_counts 
    Q_fractions /= Q_fractions.sum()
    Q_psis = psis * Q_fractions
    sum1 = Q_psis.sum(1)
    sum2 = -(psis.transpose() / sum1) @ Q_fractions
    loggamma_groups = Qs * (1. - np.log(sum1) + sum2)
    sum1 = cQfs @ gpsis.transpose()
    sum1 = np.where(sum1==0, 1., sum1)
    fracs = - cQfs / sum1
    sum2 = fracs @ gpsis
    chem_loggamma_groups = Qs*(1. - np.log(sum1) + sum2)
    loggammars = ((loggamma_groups - chem_loggamma_groups) * chemgroups).sum(1)
    return np.exp(loggammacs + loggammars)

def get_interaction(all_interactions, i, j, no_interaction):
    if i==j:
        return no_interaction
    try:
        return all_interactions[i][j]
    except:
        return no_interaction

# %% Activity Coefficients

class ActivityCoefficients:
    """Abstract class for the estimation of activity coefficients. Non-abstract subclasses should implement the following methods:
        
    __init__(self, chemicals: Iterable[Chemicals]):
        Should use pure component data from chemicals to setup future calculations of activity coefficients.
    
    __call__(self, x: 1d array, T: float):
        Should accept an array of liquid molar compositions `x`, and temperature `T` (in Kelvin), and return an array of activity coefficients. Note that the molar compositions must be in the same order as the chemicals defined when creating the ActivityCoefficients object.
    
    """
    __slots__ = ('_chemicals',)
    
    @property
    def chemicals(self):
        """tuple[Chemical] All chemicals involved in the calculation of activity coefficients."""
        return self._chemicals
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"

    
class IdealActivityCoefficients(ActivityCoefficients):
    """Create an IdealActivityCoefficients object that estimates all activity coefficients to be 1 when called with a composition and a temperature (K).
    
    Parameters
    ----------
    chemicals : Iterable[Chemical]
    
    """
    __slots__ = ()
    
    def __init__(self, chemicals):
        self._chemicals = tuple(chemicals)
    
    def __call__(self, xs, T):
        return 1.
    

class GroupActivityCoefficients(ActivityCoefficients):
    """Abstract class for the estimation of activity coefficients using group contribution methods.
    
    Parameters
    ----------
    
    chemicals : Iterable[Chemical]
    
    """
    __slots__ = ('_rs', '_qs', '_Qs','_chemgroups',
                 '_group_psis',  '_chem_Qfractions',
                 '_group_mask', '_interactions',
                 '_chemicals')
    
    def __new__(cls, chemicals):
        chemicals = tuple(chemicals)
        if chemicals in cls._cached:
            return cls._cached[chemicals]
        else:
            self = super().__new__(cls)
        getfield = getattr
        field = self.group_name
        chemgroups = [getfield(s, field) for s in chemicals]
        all_groups = set()
        for groups in chemgroups: all_groups.update(groups)
        index = {group:i for i,group in enumerate(all_groups)}
        chemgroups = chemgroup_array(chemgroups, index)
        all_subgroups = self.all_subgroups
        subgroups = [all_subgroups[i] for i in all_groups]
        main_group_ids = [i.main_group_id for i in subgroups]
        self._Qs = Qs = np.array([i.Q for i in subgroups])
        Rs = np.array([i.R for i in subgroups])
        self._rs = chemgroups @ Rs
        self._qs = chemgroups @ Qs
        self._chemgroups = chemgroups
        chem_Qs = Qs * chemgroups
        self._chem_Qfractions = cQfs = chem_Qs/chem_Qs.sum(1, keepdims=True)
        all_interactions = self.all_interactions
        N_groups = len(all_groups)
        group_shape = (N_groups, N_groups)
        no_interaction = self._no_interaction
        self._interactions = np.array(
            [[get_interaction(all_interactions, i, j, no_interaction)
              for i in main_group_ids]
             for j in main_group_ids])
        # Psis array with only symmetrically available groups
        self._group_psis = np.zeros(group_shape, dtype=float)
        # Make mask for retrieving symmetrically available groups
        rowindex = np.arange(N_groups, dtype=int)
        indices = [rowindex[rowmask] for rowmask in cQfs != 0]
        self._group_mask = group_mask = np.zeros(group_shape, dtype=bool)
        for index in indices:
            for i in index:
                group_mask[i, index] = True
        self._cached[chemicals] = self
        self._chemicals = chemicals
        return self
    
    def __reduce__(self):
        return type(self), (self.chemicals,)
    
    def __call__(self, x, T):
        """Return UNIFAC coefficients.
        
        Parameters
        ----------
        x : array_like
            Molar fractions
        T : float
            Temperature (K)
        
        """
        x = np.asarray(x)
        psis = self.psi(T, self._interactions.copy())
        self._group_psis[self._group_mask] =  psis[self._group_mask]
        gamma =  group_activity_coefficients(x, self._chemgroups,
                                             self.loggammacs(self._qs, self._rs, x),
                                             self._Qs, psis,
                                             self._chem_Qfractions,
                                             self._group_psis)
        gamma[np.isnan(gamma)] = 1
        return gamma
    
    
class UNIFACActivityCoefficiencts(GroupActivityCoefficients):
    """Create a UNIFACActivityCoefficiencts that estimates activity coefficients using the UNIFAC group contribution method when called with a composition and a temperature (K).
    
    Parameters
    ----------
    
    chemicals : Iterable[Chemical]
    
    """
    all_subgroups = UFSG
    all_interactions = UFIP
    group_name = 'UNIFAC'
    _no_interaction = 0.
    _cached = {}
    @staticmethod
    @njit
    def loggammacs(qs, rs, x):
        r_net = (x*rs).sum()
        q_net = (x*qs).sum()  
        Vs = rs/r_net
        Fs = qs/q_net
        Vs_over_Fs = Vs/Fs
        return 1. - Vs - np.log(Vs) - 5.*qs*(1. - Vs_over_Fs + np.log(Vs_over_Fs))
    
    @staticmethod
    @njit
    def psi(T, a):
        return np.exp(-a/T)


class DortmundActivityCoefficients(GroupActivityCoefficients):
    """Create a DortmundActivityCoefficients that estimates activity coefficients using the Dortmund UNIFAC group contribution method when called with a composition and a temperature (K).
    
    Parameters
    ----------
    
    chemicals : Iterable[Chemical]
    
    """
    __slots__ = ()
    all_subgroups = DOUFSG
    all_interactions = DOUFIP2016
    group_name = 'Dortmund'
    _no_interaction = (0., 0., 0.)
    _cached = {}
    
    @staticmethod
    @njit
    def loggammacs(qs, rs, x):
        r_net = (x*rs).sum()
        q_net = (x*qs).sum()
        rs_p = rs**0.75
        r_pnet = (rs_p*x).sum()
        Vs = rs/r_net
        Fs = qs/q_net
        Vs_over_Fs = Vs/Fs
        Vs_p = rs_p/r_pnet
        return 1. - Vs_p + np.log(Vs_p) - 5.*qs*(1. - Vs_over_Fs + np.log(Vs_over_Fs))
    
    @staticmethod
    @njit
    def psi(T, abc):
        abc[:, :, 0] /= T
        abc[:, :, 2] *= T
        return np.exp(-abc.sum(2)) 
    
    



