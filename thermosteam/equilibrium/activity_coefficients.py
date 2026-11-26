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
from warnings import warn
from thermo import unifac as thermo_unifac
from .unifac import DOUFSG, UFSG, NISTUFSG, PSRKSG
from numba import njit, prange
from .ideal import ideal
import thermosteam as tmo
from .interaction_parameters import get_kijs
from fluids.constants import R, R_inv
from thermo import eos_mix
import numpy as np

__all__ = ('ActivityCoefficients',
           'IdealActivityCoefficients',
           'GroupActivityCoefficients',
           'UNIFACActivityCoefficients',
           'UNIFAC2ActivityCoefficients',
           'PSRKActivityCoefficients',
           'ModifiedUNIFACActivityCoefficients',
           'DortmundActivityCoefficients',
           'ModifiedUNIFAC2ActivityCoefficients',
           'NISTActivityCoefficients',
           'EOSActivityCoefficients',
           'PRActivityCoefficients',
           'PR78ActivityCoefficients',
           'SRKActivityCoefficients',
           'activity_coefficient_classes')

# %% Utilities

def chemgroup_array(chemgroups, index):
    M = len(chemgroups)
    N = len(index)
    array = np.zeros((M, N))
    for i, groups in enumerate(chemgroups):
        for group, count in groups.items():
            array[i, index[group]] = count
    return array

@njit(cache=True)
def group_activity_coefficients(x, chem_Qs, loggammacs, psis, cQfs, gpsis):
    # chem_Qs[i, k] = Q_k * nu_ki
    Q_fractions = x @ chem_Qs
    Q_fractions /= Q_fractions.sum()
    Q_psis = psis * Q_fractions
    sum1 = Q_psis.sum(1)
    sum2 = -(psis.transpose() / sum1) @ Q_fractions
    loggamma_groups = 1. - np.log(sum1) + sum2
    sum1 = cQfs @ gpsis.transpose()
    sum1 = np.where(sum1==0, 1., sum1)
    fracs = - cQfs / sum1
    sum2 = fracs @ gpsis
    chem_loggamma_groups = 1. - np.log(sum1) + sum2
    loggammars = ((loggamma_groups - chem_loggamma_groups) * chem_Qs).sum(1)
    return np.exp(loggammacs + loggammars)

def get_interaction(all_interactions, i, j, no_interaction):
    if i==j:
        return no_interaction
    try:
        return all_interactions[i][j]
    except:
        return no_interaction

def get_chemgroups(chemicals, field):
    getfield = getattr
    chemgroups = []
    index = []
    for i, chemical in enumerate(chemicals): 
        group = getfield(chemical, field)
        if group:
            chemgroups.append(group)
            index.append(True)
        else:
            index.append(False)
            warn(f"{chemical} has no defined {field} groups; "
                  "functional group interactions are ignored",
                  RuntimeWarning, stacklevel=3)
    return np.where(index)[0], chemgroups

@njit(cache=True)
def loggammacs_UNIFAC(qs, rs, x):
    r_net = np.dot(x, rs)
    q_net = np.dot(x, qs)
    Vs = rs/r_net
    Fs = qs/q_net
    Vs_over_Fs = Vs/Fs
    return 1. - Vs + np.log(Vs) - 5.*qs*(1. - Vs_over_Fs + np.log(Vs_over_Fs))

@njit(cache=True)
def loggammacs_modified_UNIFAC(qs, rs, rs_p, x):
    # rs_p = rs**0.75
    r_net = np.dot(x, rs)
    q_net = np.dot(x, qs)
    r_pnet = np.dot(rs_p, x)
    Vs = rs/r_net
    Fs = qs/q_net
    Vs_over_Fs = Vs/Fs
    Vs_p = rs_p/r_pnet
    return 1. - Vs_p + np.log(Vs_p) - 5.*qs*(1. - Vs_over_Fs + np.log(Vs_over_Fs))

@njit(cache=True)
def psi_abc(T, abc):
    # psi = exp(-a/T - b - c*T); interaction parameters are modified in place.
    abc[:, :, 0] /= T
    abc[:, :, 2] *= T
    return np.exp(-abc.sum(2))

psi_modified_UNIFAC = psi_abc

@njit(cache=True)
def psi_UNIFAC(T, a):
    return np.exp(-a/T)

@njit(cache=True)
def fill_group_psis(group_psis, psis, group_mask):
    M = psis.shape[0]
    N = psis.shape[1]
    for i in range(M):
        for j in range(N):
            if group_mask[i, j]: 
                group_psis[i, j] = psis[i, j]
            else:
                group_psis[i, j] = 0.

@njit(cache=True)
def gamma_UNIFAC(x, T, interactions,
                 group_psis, group_mask, qs, rs,
                 chem_Qs, chem_Qfractions, index, P=None):
    N_chemicals = index.size
    gamma = np.ones(x.size)
    if N_chemicals > 1:
        x_sub = np.ones(N_chemicals)
        for i, j in enumerate(index): x_sub[i] = x[j]
        xsum = x_sub.sum()
        if xsum != 0:
            x_sub /= xsum
            psis = psi_UNIFAC(T, interactions)
            fill_group_psis(group_psis, psis, group_mask)
            gamma_sub = group_activity_coefficients(x_sub, chem_Qs,
                                        loggammacs_UNIFAC(qs, rs, x_sub),
                                        psis, chem_Qfractions, group_psis)
        for i, j in enumerate(index):
            value = gamma_sub[i]
            if np.isnan(value): continue
            gamma[j] = value
    return gamma

@njit(cache=True)
def gamma_PSRK(x, T, interactions,
               group_psis, group_mask, qs, rs,
               chem_Qs, chem_Qfractions, index, P=None):
    N_chemicals = index.size
    gamma = np.ones(x.size)
    if N_chemicals > 1:
        interactions = interactions.copy()
        x_sub = np.ones(N_chemicals)
        for i, j in enumerate(index): x_sub[i] = x[j]
        xsum = x_sub.sum()
        if xsum != 0:
            x_sub /= xsum
            psis = psi_abc(T, interactions)
            fill_group_psis(group_psis, psis, group_mask)
            gamma_sub = group_activity_coefficients(x_sub, chem_Qs,
                                        loggammacs_UNIFAC(qs, rs, x_sub),
                                        psis, chem_Qfractions, group_psis)
        for i, j in enumerate(index):
            value = gamma_sub[i]
            if np.isnan(value): continue
            gamma[j] = value
    return gamma

@njit(cache=True)
def gamma_modified_UNIFAC(x, T, interactions,
                          group_psis, group_mask, qs, rs,
                          chem_Qs, chem_Qfractions, index, rs_p, P=None):
    N_chemicals = index.size
    gamma = np.ones(x.size)
    if N_chemicals > 1:
        interactions = interactions.copy()
        x_sub = np.ones(N_chemicals)
        for i, j in enumerate(index): x_sub[i] = x[j]
        xsum = x_sub.sum()
        if xsum:
            x_sub /= xsum
            psis = psi_abc(T, interactions)
            fill_group_psis(group_psis, psis, group_mask)
            gamma_sub = group_activity_coefficients(
                x_sub, chem_Qs,
                loggammacs_modified_UNIFAC(qs, rs, rs_p, x_sub),
                psis, chem_Qfractions, group_psis
            )
        for i, j in enumerate(index):
            value = gamma_sub[i]
            if np.isnan(value): continue
            gamma[j] = value
    return gamma

# %% Analytical derivatives of group activity coefficients

@njit(cache=True)
def psi_UNIFAC_derivatives(T, a):
    # psi = exp(-a/T)
    # dpsi/dT = psi * g, where g = a/T^2
    # d2psi/dT2 = psi * (g^2 - 2a/T^3)
    psis = np.exp(-a / T)
    g = a / (T * T)
    return psis, psis * g, psis * (g * g - 2. * a / (T * T * T))

@njit(cache=True)
def psi_abc_derivatives(T, abc):
    # psi = exp(-a/T - b - c*T)
    # dpsi/dT = psi * g, where g = a/T^2 - c
    # d2psi/dT2 = psi * (g^2 - 2a/T^3)
    a = abc[:, :, 0]
    c = abc[:, :, 2]
    psis = np.exp(-a / T - abc[:, :, 1] - c * T)
    g = a / (T * T) - c
    return psis, psis * g, psis * (g * g - 2. * a / (T * T * T))

@njit(cache=True)
def loggamma_groups_derivatives(thetas, psis, dpsis, d2psis):
    # Return ln Gamma_k / Q_k and its first and second temperature
    # derivatives for each row of group surface area fractions (thetas):
    # ln Gamma_k / Q_k = 1 - ln(S_k) - sum_m(W_m * psi_km)
    # S_k = sum_m(theta_m * psi_mk); in matrix form, S = thetas @ psis
    # W_m = theta_m / S_m
    S = thetas @ psis
    dS = thetas @ dpsis
    d2S = thetas @ d2psis
    empty = S == 0. # Groups absent from pure components
    S = np.where(empty, 1., S)
    dS = np.where(empty, 0., dS)
    d2S = np.where(empty, 0., d2S)
    dlnS = dS / S
    W = thetas / S
    dW = -W * dlnS
    d2W = -W * (d2S / S - 2. * dlnS * dlnS)
    psisT = psis.transpose()
    dpsisT = dpsis.transpose()
    loggammas = 1. - np.log(S) - W @ psisT
    dloggammas = -dlnS - dW @ psisT - W @ dpsisT
    d2loggammas = (
        dlnS * dlnS - d2S / S
        - d2W @ psisT - 2. * dW @ dpsisT - W @ d2psis.transpose()
    )
    return loggammas, dloggammas, d2loggammas

@njit(cache=True)
def group_activity_coefficients_derivatives(
        x, chem_Qs, loggammacs, psis, dpsis, d2psis,
        cQfs, gpsis, dgpsis, d2gpsis
    ):
    # Return activity coefficients and the first and second temperature
    # derivatives of their logarithm. The combinatorial part (loggammacs)
    # is independent of temperature.
    Q_fractions = x @ chem_Qs
    Q_fractions /= Q_fractions.sum()
    thetas = Q_fractions.reshape((1, Q_fractions.size))
    mix = loggamma_groups_derivatives(
        thetas, psis.transpose(), dpsis.transpose(), d2psis.transpose()
    )
    pure = loggamma_groups_derivatives(
        cQfs, gpsis.transpose(), dgpsis.transpose(), d2gpsis.transpose()
    )
    loggammars = ((mix[0] - pure[0]) * chem_Qs).sum(1)
    dloggammars = ((mix[1] - pure[1]) * chem_Qs).sum(1)
    d2loggammars = ((mix[2] - pure[2]) * chem_Qs).sum(1)
    return np.exp(loggammacs + loggammars), dloggammars, d2loggammars

@njit(cache=True)
def group_gamma_derivative(dgamma, x_sub, psis, dpsis, d2psis, group_mask,
                           chem_Qs, chem_Qfractions, index, loggammacs, order):
    # Set first (order=1) or second (order=2) temperature derivative of
    # activity coefficients given the normalized composition of chemicals
    # with groups (x_sub) and the combinatorial part (loggammacs)
    gamma, dlngamma, d2lngamma = group_activity_coefficients_derivatives(
        x_sub, chem_Qs, loggammacs, psis, dpsis, d2psis,
        chem_Qfractions,
        np.where(group_mask, psis, 0.),
        np.where(group_mask, dpsis, 0.),
        np.where(group_mask, d2psis, 0.),
    )
    if order == 1:
        dgamma_sub = gamma * dlngamma
    else:
        dgamma_sub = gamma * (d2lngamma + dlngamma * dlngamma)
    for i, j in enumerate(index):
        value = dgamma_sub[i]
        if not np.isnan(value): dgamma[j] = value

@njit(cache=True)
def gamma_UNIFAC_derivative(x, T, interactions, group_mask, qs, rs,
                            chem_Qs, chem_Qfractions, index, order):
    dgamma = np.zeros(x.size)
    if index.size > 1:
        x_sub = x[index]
        xsum = x_sub.sum()
        if xsum:
            x_sub = x_sub / xsum
            psis, dpsis, d2psis = psi_UNIFAC_derivatives(T, interactions)
            group_gamma_derivative(
                dgamma, x_sub, psis, dpsis, d2psis, group_mask, chem_Qs,
                chem_Qfractions, index, loggammacs_UNIFAC(qs, rs, x_sub), order
            )
    return dgamma

@njit(cache=True)
def gamma_PSRK_derivative(x, T, interactions, group_mask, qs, rs,
                          chem_Qs, chem_Qfractions, index, order):
    dgamma = np.zeros(x.size)
    if index.size > 1:
        x_sub = x[index]
        xsum = x_sub.sum()
        if xsum:
            x_sub = x_sub / xsum
            psis, dpsis, d2psis = psi_abc_derivatives(T, interactions)
            group_gamma_derivative(
                dgamma, x_sub, psis, dpsis, d2psis, group_mask, chem_Qs,
                chem_Qfractions, index, loggammacs_UNIFAC(qs, rs, x_sub), order
            )
    return dgamma

@njit(cache=True)
def gamma_modified_UNIFAC_derivative(x, T, interactions, group_mask, qs, rs,
                                     chem_Qs, chem_Qfractions, index, rs_p, order):
    dgamma = np.zeros(x.size)
    if index.size > 1:
        x_sub = x[index]
        xsum = x_sub.sum()
        if xsum:
            x_sub = x_sub / xsum
            psis, dpsis, d2psis = psi_abc_derivatives(T, interactions)
            group_gamma_derivative(
                dgamma, x_sub, psis, dpsis, d2psis, group_mask, chem_Qs,
                chem_Qfractions, index,
                loggammacs_modified_UNIFAC(qs, rs, rs_p, x_sub), order
            )
    return dgamma

@njit(cache=True)
def gamma_UNIFAC_der_T(x, T, interactions, group_psis, group_mask, qs, rs,
                       chem_Qs, chem_Qfractions, index, P=None):
    return gamma_UNIFAC_derivative(x, T, interactions, group_mask, qs, rs,
                                   chem_Qs, chem_Qfractions, index, 1)

@njit(cache=True)
def gamma_UNIFAC_der2_T(x, T, interactions, group_psis, group_mask, qs, rs,
                        chem_Qs, chem_Qfractions, index, P=None):
    return gamma_UNIFAC_derivative(x, T, interactions, group_mask, qs, rs,
                                   chem_Qs, chem_Qfractions, index, 2)

@njit(cache=True)
def gamma_PSRK_der_T(x, T, interactions, group_psis, group_mask, qs, rs,
                     chem_Qs, chem_Qfractions, index, P=None):
    return gamma_PSRK_derivative(x, T, interactions, group_mask, qs, rs,
                                 chem_Qs, chem_Qfractions, index, 1)

@njit(cache=True)
def gamma_PSRK_der2_T(x, T, interactions, group_psis, group_mask, qs, rs,
                      chem_Qs, chem_Qfractions, index, P=None):
    return gamma_PSRK_derivative(x, T, interactions, group_mask, qs, rs,
                                 chem_Qs, chem_Qfractions, index, 2)

@njit(cache=True)
def gamma_modified_UNIFAC_der_T(x, T, interactions, group_psis, group_mask, qs, rs,
                                chem_Qs, chem_Qfractions, index, rs_p, P=None):
    return gamma_modified_UNIFAC_derivative(x, T, interactions, group_mask, qs, rs,
                                            chem_Qs, chem_Qfractions, index, rs_p, 1)

@njit(cache=True)
def gamma_modified_UNIFAC_der2_T(x, T, interactions, group_psis, group_mask, qs, rs,
                                 chem_Qs, chem_Qfractions, index, rs_p, P=None):
    return gamma_modified_UNIFAC_derivative(x, T, interactions, group_mask, qs, rs,
                                            chem_Qs, chem_Qfractions, index, rs_p, 2)

@njit(cache=True)
def gamma_group_der_P(x, T, interactions, group_psis, group_mask, qs, rs,
                      chem_Qs, chem_Qfractions, index, rs_p=None, P=None):
    # Group contribution models are independent of pressure
    return np.zeros(x.size)

@njit(cache=True)
def ideal_coefficient_der(z=None, T=None, P=None):
    return 0.


# %% Activity Coefficients

class ActivityCoefficients:
    """
    Abstract class for the estimation of activity coefficients. 
    Non-abstract subclasses should implement the following methods:
        
    __init__(self, chemicals: Iterable[:class:`~thermosteam.Chemical`]):
        Should use pure component data from chemicals to setup future 
        calculations of activity coefficients.
    
    __call__(self, x: 1d array, T: float):
        Should accept an array of liquid molar compositions `x`, and
        temperature `T` (in Kelvin), and return an array of activity 
        coefficients. Note that the molar compositions must be in the same 
        order as the chemicals defined when creating the ActivityCoefficients
        object.
    
    """
    __slots__ = ('_chemicals',)
    
    @property
    def chemicals(self):
        """tuple[Chemical] All chemicals involved in the calculation of activity coefficients."""
        return self._chemicals
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"

@ideal
class IdealActivityCoefficients(ActivityCoefficients):
    """
    Create an IdealActivityCoefficients object that estimates all activity 
    coefficients to be 1 when called with a composition and a temperature (K).
    
    Parameters
    ----------
    chemicals : Iterable[:class:`~thermosteam.Chemical`]
    
    """
    __slots__ = ()
    P_dependent = False
    
    def __init__(self, chemicals):
        self._chemicals = tuple(chemicals)
    
    def __call__(self, xs, T, P=101325):
        return np.ones(len(xs))

    @property
    def f_der_T(self):
        return ideal_coefficient_der
    
    @property
    def f_der2_T(self):
        return ideal_coefficient_der

    @property
    def f_der_P(self):
        return ideal_coefficient_der


class GroupActivityCoefficients(ActivityCoefficients):
    """
    Abstract class for the estimation of activity coefficients using
    group contribution methods.
    
    Parameters
    ----------
    
    chemicals : Iterable[:class:`~thermosteam.Chemical`]
    
    """
    P_dependent = False
    __slots__ = ('_rs', '_qs', '_Qs','_chemgroups', '_chem_Qs',
                 '_group_psis',  '_chem_Qfractions',
                 '_group_mask', '_interactions',
                 '_chemicals', '_index', '_args')

    #: Interaction parameters for missing interactions.
    _no_interaction = (0., 0., 0.)
    _cached = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._cached = {} # Each model has its own cache

    def __new__(cls, chemicals):
        chemicals = tuple(chemicals)
        if chemicals in cls._cached:
            return cls._cached[chemicals]
        else:
            self = super().__new__(cls)
        index, chemgroups = get_chemgroups(chemicals, self.group_name)
        if len(index) <= 1: return IdealActivityCoefficients(chemicals)
        self._index = index
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
        self._chem_Qs = chem_Qs = Qs * chemgroups
        self._chem_Qfractions = cQfs = chem_Qs/chem_Qs.sum(1, keepdims=True)
        all_interactions = self.all_interactions
        if isinstance(all_interactions, str): # Load on demand
            all_interactions = getattr(thermo_unifac, all_interactions)
        N_groups = len(all_groups)
        group_shape = (N_groups, N_groups)
        no_interaction = self._no_interaction
        self._interactions = np.array(
            [[get_interaction(all_interactions, i, j, no_interaction)
              for i in main_group_ids]
             for j in main_group_ids], dtype=float)
        # Psis array with only symmetrically available groups
        self._group_psis = np.zeros(group_shape, dtype=float)
        # Make mask for retrieving symmetrically available groups
        rowindex = np.arange(N_groups, dtype=int)
        indices = [rowindex[rowmask] for rowmask in cQfs != 0]
        self._group_mask = group_mask = np.zeros(group_shape, dtype=bool)
        for index in indices:
            for i in index:
                group_mask[i, index] = True
        self._save_args()
        self._cached[chemicals] = self
        self._chemicals = chemicals
        return self
    
    def _save_args(self):
        # Save arguments of activity coefficient functions, which are 
        # independent of temperature, pressure, and composition. Subclasses 
        # may extend this method to save model-specific arguments.
        self._args = (self._interactions, self._group_psis, self._group_mask,
                      self._qs, self._rs, self._chem_Qs,
                      self._chem_Qfractions, self._index)
    
    def __reduce__(self):
        return type(self), (self.chemicals,)
    
    @property
    def args(self):
        return self._args

    @property
    def f_der_P(self):
        return gamma_group_der_P

    def activity_coefficients(self, x, T, P=101325):
        """
        Return activity coefficients of chemicals with defined functional groups.
        
        Parameters
        ----------
        x : array_like
            Molar fractions
        T : float
            Temperature [K]
        
        """
        index = self._index
        x_all = np.zeros(len(self._chemicals))
        x_all[index] = x
        return self.f(x_all, T, *self._args)[index]
    
    def __call__(self, x, T, P=101325):
        """
        Return activity coefficients.
        
        Parameters
        ----------
        x : array_like
            Molar fractions
        T : float
            Temperature [K]
        
        Notes
        -----
        Activity coefficients of chemicals with missing groups are default to 1.
        
        """
        x = np.asarray(x, float)
        gamma = self.f(x, T, *self.args)
        return gamma
    
    
class UNIFACActivityCoefficients(GroupActivityCoefficients):
    """
    Create a UNIFACActivityCoefficients that estimates activity coefficients
    using the original UNIFAC group contribution method when called with a
    composition and a temperature (K).

    Parameters
    ----------

    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    >>> Gamma = tmo.equilibrium.UNIFACActivityCoefficients(chemicals)
    >>> Gamma([0.5, 0.5], 350.)
    array([1.486, 1.231])

    """
    __slots__ = ()
    all_subgroups = UFSG
    all_interactions = 'UFIP'
    group_name = 'UNIFAC'
    _no_interaction = 0. # Only parameter a, where psi = exp(-a/T)

    @property
    def f(self):
        return gamma_UNIFAC

    @property
    def f_der_T(self):
        return gamma_UNIFAC_der_T
    
    @property
    def f_der2_T(self):
        return gamma_UNIFAC_der2_T


class UNIFAC2ActivityCoefficients(UNIFACActivityCoefficients):
    """
    Create a UNIFAC2ActivityCoefficients that estimates activity coefficients
    using the UNIFAC 2.0 group contribution method (original UNIFAC with
    revised interaction parameters) when called with a composition and a
    temperature (K).

    Parameters
    ----------

    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    >>> Gamma = tmo.equilibrium.UNIFAC2ActivityCoefficients(chemicals)
    >>> Gamma([0.5, 0.5], 350.)
    array([1.471, 1.172])

    """
    __slots__ = ()
    all_interactions = 'UF2IP'


class PSRKActivityCoefficients(UNIFACActivityCoefficients):
    """
    Create a PSRKActivityCoefficients that estimates activity coefficients
    using the PSRK (predictive Soave-Redlich-Kwong) UNIFAC group contribution
    method when called with a composition and a temperature (K).

    Parameters
    ----------

    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    >>> Gamma = tmo.equilibrium.PSRKActivityCoefficients(chemicals)
    >>> Gamma([0.5, 0.5], 350.)
    array([1.486, 1.231])

    """
    __slots__ = ()
    all_subgroups = PSRKSG
    all_interactions = 'PSRKIP'
    group_name = 'PSRK'
    _no_interaction = (0., 0., 0.) # Parameters (a, b, c), where psi = exp(-a/T - b - c*T)

    @property
    def f(self):
        return gamma_PSRK

    @property
    def f_der_T(self):
        return gamma_PSRK_der_T

    @property
    def f_der2_T(self):
        return gamma_PSRK_der2_T


class ModifiedUNIFACActivityCoefficients(GroupActivityCoefficients):
    """
    Create a ModifiedUNIFACActivityCoefficients object that estimates activity
    coefficients using the modified (Dortmund) UNIFAC group contribution
    method when called with a composition and a temperature (K).
    Subclasses may use other subgroups and interaction parameters with the
    modified combinatorial term (e.g., NIST-modified UNIFAC).

    Parameters
    ----------

    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    >>> Gamma = tmo.equilibrium.ModifiedUNIFACActivityCoefficients(chemicals)
    >>> composition = [0.5, 0.5]
    >>> T = 350.
    >>> Gamma(composition, T)
    array([1.475, 1.242])

    >>> chemicals = tmo.Chemicals(['Dodecane', 'Tridecane'], cache=True)
    >>> Gamma = tmo.equilibrium.ModifiedUNIFACActivityCoefficients(chemicals)
    >>> # Note how both hydrocarbons have similar lenghts and structure,
    >>> # so activities should be very close
    >>> Gamma([0.5, 0.5], 350.)
    array([1., 1.])

    >>> chemicals = tmo.Chemicals(['Water', 'O2'], cache=True)
    >>> Gamma = tmo.equilibrium.ModifiedUNIFACActivityCoefficients(chemicals)
    >>> # The following warning is issued because O2 does not have Dortmund groups
    >>> # RuntimeWarning: O2 has no defined Dortmund groups;
    >>> # functional group interactions are ignored
    >>> Gamma([0.5, 0.5], 350.)
    array([1., 1.])

    """
    __slots__ = ()
    all_subgroups = DOUFSG
    all_interactions = 'DOUFIP2016'
    group_name = 'Dortmund'

    @property
    def f(self):
        return gamma_modified_UNIFAC

    @property
    def f_der_T(self):
        return gamma_modified_UNIFAC_der_T
    
    @property
    def f_der2_T(self):
        return gamma_modified_UNIFAC_der2_T

    def _save_args(self):
        super()._save_args()
        # Volume fractions of the modified combinatorial term use r_i^0.75
        self._args += (self._rs ** 0.75,)

DortmundActivityCoefficients = ModifiedUNIFACActivityCoefficients


class ModifiedUNIFAC2ActivityCoefficients(ModifiedUNIFACActivityCoefficients):
    """
    Create a ModifiedUNIFAC2ActivityCoefficients that estimates activity
    coefficients using the modified (Dortmund) UNIFAC 2.0 group contribution
    method (Dortmund UNIFAC with revised interaction parameters) when called
    with a composition and a temperature (K).

    Parameters
    ----------

    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    >>> Gamma = tmo.equilibrium.ModifiedUNIFAC2ActivityCoefficients(chemicals)
    >>> Gamma([0.5, 0.5], 350.)
    array([1.417, 1.186])

    """
    __slots__ = ()
    all_interactions = 'DOUF2IP'


class NISTActivityCoefficients(ModifiedUNIFACActivityCoefficients):
    """
    Create a NISTActivityCoefficients that estimates activity coefficients
    using the NIST-modified UNIFAC group contribution method when called with a
    composition and a temperature (K).

    Parameters
    ----------

    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    Examples
    --------
    >>> import thermosteam as tmo
    >>> Water, Ethanol = chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    >>> Ethanol.NIST.set_group_counts_by_name({'CH3':1, 'CH2':1, 'OH prim':1})
    >>> Water.NIST.set_group_counts_by_name({'H2O':1})
    >>> Gamma = tmo.equilibrium.NISTActivityCoefficients(chemicals)
    >>> composition = [0.5, 0.5]
    >>> T = 350.
    >>> Gamma(composition, T)
    array([1.479, 1.238])

    """
    __slots__ = ()
    all_subgroups = NISTUFSG
    all_interactions = 'NISTUFIP'
    group_name = 'NIST'


# %% Activity coefficients from cubic equations of state

def is_missing(value):
    return value is None or value != value

def Cp_dep_partials(T, V, b, delta, epsilon, a_alpha, da_alpha_dT, d2a_alpha_dT2):
    r"""
    Return partial derivatives of the departure isobaric heat capacity of a
    generic cubic equation of state, :math:`P = RT/(V - b) - a_\alpha/D`
    with :math:`D = V^2 + \delta V + \epsilon` (as in thermo), with respect
    to V, b, delta, epsilon, a_alpha, da_alpha_dT, and d2a_alpha_dT2:

    .. math::

        C_{p,dep} = -T \frac{P_T^2}{P_V} + T a_\alpha'' \frac{L}{s} - R

        P_T = \frac{R}{V - b} - \frac{a_\alpha'}{D}

        P_V = \frac{a_\alpha D_V}{D^2} - \frac{RT}{(V - b)^2}

        D_V = 2V + \delta \quad s = \sqrt{\delta^2 - 4\epsilon} \quad
        L = \ln\frac{D_V + s}{D_V - s}

    Partial derivatives were verified with sympy.

    """
    D = V * (V + delta) + epsilon
    D2 = D * D
    D3 = D2 * D
    Dv = V + V + delta
    s2 = delta * delta - 4. * epsilon
    s = s2 ** 0.5
    L_s = np.log((Dv + s) / (Dv - s)) / s
    Vb = 1. / (V - b)
    Vb2 = Vb * Vb
    Vb3 = Vb2 * Vb
    RT = R * T
    P_T = R * Vb - da_alpha_dT / D
    P_V = a_alpha * Dv / D2 - RT * Vb2
    # Partial derivatives of P_T, P_V, and L/s for (V, b, delta, epsilon, a, a', a'')
    dP_T = (
        -R * Vb2 + da_alpha_dT * Dv / D2,
        R * Vb2,
        da_alpha_dT * V / D2,
        da_alpha_dT / D2,
        0.,
        -1. / D,
        0.,
    )
    dP_V = (
        2. * a_alpha * (1. / D2 - Dv * Dv / D3) + 2. * RT * Vb3,
        -2. * RT * Vb3,
        a_alpha * (1. / D2 - 2. * Dv * V / D3),
        -2. * a_alpha * Dv / D3,
        Dv / D2,
        0.,
        0.,
    )
    dL_s = (
        -1. / D,
        0.,
        (-0.5 + 0.5 * Dv * delta / s2) / D - L_s * delta / s2,
        -Dv / (D * s2) + 2. * L_s / s2,
        0.,
        0.,
        0.,
    )
    T_P_V2 = T / (P_V * P_V)
    partials = [
        -T_P_V2 * (2. * P_T * i * P_V - P_T * P_T * j) + T * d2a_alpha_dT2 * k
        for i, j, k in zip(dP_T, dP_V, dL_s)
    ]
    partials[-1] += T * L_s
    return partials

def lnphis_der2_T(eos, phase):
    r"""
    Return the second temperature derivative of the log fugacity coefficients
    of a mixture cubic equation of state (thermo GCEOSMIX object) at constant
    pressure and composition:

    .. math::

        \frac{\partial^2 \ln \phi_i}{\partial T^2} =
        \frac{2 \bar{h}_i}{RT^3} - \frac{\bar{c}_{p,i}}{RT^2}

        \bar{h}_i = -RT^2 \frac{\partial \ln \phi_i}{\partial T}

        \bar{c}_{p,i} = C_{p,dep} + \frac{\partial C_{p,dep}}{\partial z_i}
        - \sum_j z_j \frac{\partial C_{p,dep}}{\partial z_j}

    where :math:`\bar{h}_i` and :math:`\bar{c}_{p,i}` are the partial molar
    departure enthalpy and heat capacity.

    """
    T = eos.T
    Z = getattr(eos, 'Z_' + phase)
    Cp_dep = getattr(eos, 'Cp_dep_' + phase)
    partials = Cp_dep_partials(
        T, getattr(eos, 'V_' + phase), eos.b, eos.delta, eos.epsilon,
        eos.a_alpha, eos.da_alpha_dT, eos.d2a_alpha_dT2
    )
    dzs = (
        eos.dV_dzs(Z), eos.db_dzs, eos.ddelta_dzs, eos.depsilon_dzs,
        eos.da_alpha_dzs, eos.da_alpha_dT_dzs, eos.d2a_alpha_dT2_dzs,
    )
    dCp_dep_dzs = sum([i * np.array(j) for i, j in zip(partials, dzs)])
    Cp_dep_partial_molar = Cp_dep + dCp_dep_dzs - np.dot(eos.zs, dCp_dep_dzs)
    h_dep_partial_molar = -R * T * T * np.array(eos.dlnphis_dT(phase))
    return (2. * h_dep_partial_molar / T - Cp_dep_partial_molar) / (R * T * T)

class EOSActivityCoefficients(ActivityCoefficients):
    r"""
    Abstract class for the estimation of activity coefficients using a cubic
    equation of state (EOS) when called with a composition, a temperature (K),
    and a pressure (Pa). Activity coefficients are computed as the ratio of
    the fugacity coefficient of each chemical in the liquid mixture to its
    fugacity coefficient as a pure liquid at the same temperature and pressure
    (Lewis-Randall standard state):

    .. math::
        \gamma_i(T, P, x) = \frac{\phi_i(T, P, x)}{\phi_i^0(T, P)}

    The liquid root of the EOS is used, or the vapor root if no liquid root
    exists (same as :meth:`thermo.phases.CEOSLiquid.gammas`).

    Non-abstract subclasses must define the `EOS` attribute (a mixture cubic
    EOS class from :mod:`thermo.eos_mix`) and may define the `chemsep_db`
    attribute (name of the ChemSep table of binary interaction parameters).

    Parameters
    ----------
    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    Notes
    -----
    Activity coefficients of chemicals with a missing critical temperature,
    critical pressure, or acentric factor are default to 1.

    """
    __slots__ = ('_index', '_eos', '_eos_parameters', '_TP', '_phis_pure')
    P_dependent = True
    EOS = None #: type[GCEOSMIX] Mixture cubic EOS class.
    chemsep_db = None #: Optional[str] Name of ChemSep table of interaction parameters.
    _cached = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._cached = {} # Each model has its own cache

    def __new__(cls, chemicals):
        chemicals = tuple(chemicals)
        cached = cls._cached
        if chemicals in cached: return cached[chemicals]
        index = []
        for i, chemical in enumerate(chemicals):
            if any([is_missing(getattr(chemical, j)) for j in ('Tc', 'Pc', 'omega')]):
                warn(f"{chemical} has no defined critical temperature, critical "
                      "pressure, or acentric factor; activity coefficient is "
                      "default to 1", RuntimeWarning, stacklevel=2)
            else:
                index.append(i)
        if len(index) <= 1: return IdealActivityCoefficients(chemicals)
        self = super().__new__(cls)
        self._chemicals = chemicals
        self._index = np.array(index, dtype=int)
        self._save_eos_parameters()
        self._eos = None
        self._TP = None
        self._phis_pure = None
        cached[chemicals] = self
        return self

    def _save_eos_parameters(self):
        # Save EOS parameters of chemicals with defined critical properties,
        # which are independent of temperature, pressure, and composition
        chemicals = [self._chemicals[i] for i in self._index]
        Tcs = [i.Tc for i in chemicals]
        Pcs = [i.Pc for i in chemicals]
        omegas = [i.omega for i in chemicals]
        kijs = None
        if self.chemsep_db is not None:
            try:
                kijs = get_kijs(self.chemsep_db, chemicals)
            except: pass
        self._eos_parameters = dict(kijs=kijs, Tcs=Tcs, Pcs=Pcs, omegas=omegas)

    def __reduce__(self):
        return type(self), (self.chemicals,)

    def eos(self, x, T, P):
        """Return mixture EOS object (liquid root) at given composition of
        chemicals with defined critical properties, temperature (K), and
        pressure (Pa)."""
        x = [float(i) for i in x]
        eos = self._eos
        if eos is None:
            eos = self.EOS(T=T, P=P, zs=x, only_l=True, **self._eos_parameters)
        else:
            eos = eos.to_TP_zs_fast(T=T, P=P, zs=x, only_l=True)
        self._eos = eos
        return eos

    def fugacity_coefficients_pure(self, T, P):
        """Return liquid fugacity coefficients of pure chemicals with defined
        critical properties at given temperature (K) and pressure (Pa)."""
        if self._TP == (T, P): return self._phis_pure
        phis_pure = []
        for i in self._eos.pures():
            try: phis_pure.append(i.phi_l)
            except AttributeError: phis_pure.append(i.phi_g)
        self._TP = (T, P)
        self._phis_pure = phis_pure = np.array(phis_pure)
        return phis_pure

    def __call__(self, x, T, P=101325.):
        """
        Return activity coefficients.

        Parameters
        ----------
        x : array_like
            Molar fractions
        T : float
            Temperature [K]
        P : float, optional
            Pressure [Pa]. Defaults to 101325.

        """
        x = np.asarray(x, float)
        gamma = np.ones(x.size)
        index = self._index
        x_sub = x[index]
        xsum = x_sub.sum()
        if xsum == 0: return gamma
        eos = self.eos(x_sub / xsum, T, P)
        try: Z = eos.Z_l
        except AttributeError: Z = eos.Z_g
        lnphis = np.array(eos.fugacity_coefficients(Z))
        gamma[index] = np.exp(lnphis) / self.fugacity_coefficients_pure(T, P)
        return gamma

    def _derivative(self, x, T, P, var, order=1):
        # d gamma_i / dv = gamma_i * d ln gamma_i / dv
        # d2 gamma_i / dv2 = gamma_i * [d2 ln gamma_i / dv2 + (d ln gamma_i / dv)^2]
        # ln gamma_i = ln phi_i - ln phi_i^0
        x = np.asarray(x, float)
        dgamma = np.zeros(x.size)
        index = self._index
        x_sub = x[index]
        xsum = x_sub.sum()
        if xsum == 0: return dgamma
        eos = self.eos(x_sub / xsum, T, P)
        phase = 'l' if hasattr(eos, 'Z_l') else 'g'
        lnphis = np.array(eos.fugacity_coefficients(getattr(eos, 'Z_' + phase)))
        dlnphis = np.array(getattr(eos, 'dlnphis_d' + var)(phase))
        if order == 2: d2lnphis = lnphis_der2_T(eos, phase)
        phis_pure = []
        dlnphis_pure = []
        d2lnphis_pure = []
        for pure in eos.pures():
            phase = 'l' if hasattr(pure, 'phi_l') else 'g'
            phi = getattr(pure, 'phi_' + phase)
            phis_pure.append(phi)
            dlnphis_pure.append(getattr(pure, f'dphi_d{var}_{phase}') / phi)
            if order == 2:
                # d ln phi / dT = -H_dep / (RT^2); d2 ln phi / dT2 = -Cp_dep / (RT^2) + 2 H_dep / (RT^3)
                RT2 = R * T * T
                d2lnphis_pure.append(
                    (2. * getattr(pure, 'H_dep_' + phase) / T
                     - getattr(pure, 'Cp_dep_' + phase)) / RT2
                )
        gamma = np.exp(lnphis) / np.array(phis_pure)
        dlngamma = dlnphis - np.array(dlnphis_pure)
        if order == 1:
            dgamma[index] = gamma * dlngamma
        else:
            d2lngamma = d2lnphis - np.array(d2lnphis_pure)
            dgamma[index] = gamma * (d2lngamma + dlngamma * dlngamma)
        return dgamma

    def derivative_T(self, x, T, P=101325.):
        """Return temperature derivative of activity coefficients [1/K]."""
        return self._derivative(x, T, P, 'T')

    def derivative2_T(self, x, T, P=101325.):
        """Return second temperature derivative of activity coefficients [1/K^2]."""
        return self._derivative(x, T, P, 'T', 2)

    def derivative_P(self, x, T, P=101325.):
        """Return pressure derivative of activity coefficients [1/Pa]."""
        return self._derivative(x, T, P, 'P')

    @property
    def f(self):
        return self

    @property
    def f_der_T(self):
        return self.derivative_T

    @property
    def f_der2_T(self):
        return self.derivative2_T

    @property
    def f_der_P(self):
        return self.derivative_P

    @property
    def args(self):
        return ()


class PRActivityCoefficients(EOSActivityCoefficients):
    """
    Create a PRActivityCoefficients object that estimates activity
    coefficients using the Peng-Robinson equation of state when called with
    a composition, a temperature (K), and a pressure (Pa). Binary interaction
    parameters are taken from ChemSep (missing parameters are zero).

    Parameters
    ----------
    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Hexane', 'Toluene'], cache=True)
    >>> Gamma = tmo.equilibrium.PRActivityCoefficients(chemicals)
    >>> Gamma([0.5, 0.5], 350., 101325.)
    array([1.097, 1.117])

    """
    __slots__ = ()
    EOS = eos_mix.PRMIX
    chemsep_db = 'ChemSep PR'


class PR78ActivityCoefficients(PRActivityCoefficients):
    """
    Create a PR78ActivityCoefficients object that estimates activity
    coefficients using the Peng-Robinson (1978) equation of state when called
    with a composition, a temperature (K), and a pressure (Pa). Binary
    interaction parameters are taken from ChemSep (missing parameters are zero).

    Parameters
    ----------
    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Hexane', 'Toluene'], cache=True)
    >>> Gamma = tmo.equilibrium.PR78ActivityCoefficients(chemicals)
    >>> Gamma([0.5, 0.5], 350., 101325.)
    array([1.097, 1.117])

    """
    __slots__ = ()
    EOS = eos_mix.PR78MIX


class SRKActivityCoefficients(EOSActivityCoefficients):
    """
    Create a SRKActivityCoefficients object that estimates activity
    coefficients using the Soave-Redlich-Kwong equation of state when called
    with a composition, a temperature (K), and a pressure (Pa). Binary
    interaction parameters are taken from ChemSep, or estimated from ChemSep
    PR interaction parameters using the PR2SRK relationship if missing
    (see :func:`~thermosteam.equilibrium.SRK_kijs`).

    Parameters
    ----------
    chemicals : Iterable[:class:`~thermosteam.Chemical`]

    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Hexane', 'Toluene'], cache=True)
    >>> Gamma = tmo.equilibrium.SRKActivityCoefficients(chemicals)
    >>> Gamma([0.5, 0.5], 350., 101325.)
    array([1.097, 1.116])

    """
    __slots__ = ()
    EOS = eos_mix.SRKMIX
    chemsep_db = 'ChemSep SRK'


activity_coefficient_classes = (
    UNIFACActivityCoefficients,
    UNIFAC2ActivityCoefficients,
    PSRKActivityCoefficients,
    ModifiedUNIFACActivityCoefficients,
    ModifiedUNIFAC2ActivityCoefficients,
    NISTActivityCoefficients,
    PRActivityCoefficients,
    PR78ActivityCoefficients,
    SRKActivityCoefficients,
)
