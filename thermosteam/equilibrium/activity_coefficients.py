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
from .unifac import DOUFSG, DOUFIP2016, UFIP, UFSG, NISTUFSG, NISTUFIP
from numba import njit, prange
from .ideal import ideal
import thermosteam as tmo
from thermo.interaction_parameters import IPDB
from fluids.constants import R_inv
from thermo import eos_mix
import numpy as np

__all__ = ('ActivityCoefficients',
           'IdealActivityCoefficients',
           'GroupActivityCoefficients',
           'DortmundActivityCoefficients',
           'UNIFACActivityCoefficients',
           'NISTActivityCoefficients')

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
    return 1. - Vs - np.log(Vs) - 5.*qs*(1. - Vs_over_Fs + np.log(Vs_over_Fs))

@njit(cache=True)
def loggammacs_modified_UNIFAC(qs, rs, x):
    r_net = np.dot(x, rs)
    q_net = np.dot(x, qs)
    rs_p = rs**0.75
    r_pnet = np.dot(rs_p, x)
    Vs = rs/r_net
    Fs = qs/q_net
    Vs_over_Fs = Vs/Fs
    Vs_p = rs_p/r_pnet
    return 1. - Vs_p + np.log(Vs_p) - 5.*qs*(1. - Vs_over_Fs + np.log(Vs_over_Fs))

@njit(cache=True)
def psi_modified_UNIFAC(T, abc):
    abc[:, :, 0] /= T
    abc[:, :, 2] *= T
    return np.exp(-abc.sum(2)) 

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
                 group_psis, group_mask, qs, rs, Qs,
                 chemgroups, chem_Qfractions, index):
    N_chemicals = index.size
    gamma = np.ones(x.size)
    if N_chemicals > 1:
        interactions = interactions.copy()
        x_sub = np.ones(N_chemicals)
        for i, j in enumerate(index): x[j] = x_sub[i]
        xsum = x_sub.sum()
        if xsum != 0: 
            x_sub /= xsum
            psis = psi_UNIFAC(T, interactions)
            fill_group_psis(group_psis, psis, group_mask)
            gamma_sub = group_activity_coefficients(x_sub, chemgroups,
                                        loggammacs_UNIFAC(qs, rs, x_sub),
                                        Qs, psis,
                                        chem_Qfractions,
                                        group_psis)
        for i, j in enumerate(index): 
            value = gamma_sub[i]
            if np.isnan(value): continue
            gamma[j] = value
    return gamma

@njit(cache=True)
def gamma_modified_UNIFAC(x, T, interactions, 
                   group_psis, group_mask, qs, rs, Qs,
                   chemgroups, chem_Qfractions, index):
    N_chemicals = index.size
    gamma = np.ones(x.size)
    if N_chemicals > 1:
        interactions = interactions.copy()
        x_sub = np.ones(N_chemicals)
        for i, j in enumerate(index): x_sub[i] = x[j]
        xsum = x_sub.sum()
        if xsum:
            x_sub /= xsum
            psis = psi_modified_UNIFAC(T, interactions)
            fill_group_psis(group_psis, psis, group_mask)
            gamma_sub = group_activity_coefficients(x_sub, chemgroups,
                                        loggammacs_modified_UNIFAC(qs, rs, x_sub),
                                        Qs, psis,
                                        chem_Qfractions,
                                        group_psis)
        for i, j in enumerate(index): 
            value = gamma_sub[i]
            if np.isnan(value): continue
            gamma[j] = value
    return gamma
    
    
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
    
    def __init__(self, chemicals):
        self._chemicals = tuple(chemicals)
    
    def __call__(self, xs, T):
        return np.ones(xs.size)
    

class GroupActivityCoefficients(ActivityCoefficients):
    """
    Abstract class for the estimation of activity coefficients using
    group contribution methods.
    
    Parameters
    ----------
    
    chemicals : Iterable[:class:`~thermosteam.Chemical`]
    
    """
    __slots__ = ('_rs', '_qs', '_Qs','_chemgroups',
                 '_group_psis',  '_chem_Qfractions',
                 '_group_mask', '_interactions',
                 '_chemicals', '_index')
    
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
    
    @property
    def args(self):
        return (self._interactions, 
                self._group_psis, self._group_mask,
                self._qs, self._rs, self._Qs,
                self._chemgroups, self._chem_Qfractions, 
                self._index)
    
    def activity_coefficients(self, x, T):
        """
        Return activity coefficients of chemicals with defined functional groups.
        
        Parameters
        ----------
        x : array_like
            Molar fractions
        T : float
            Temperature [K]
        
        """
        psis = self.psi(T, self._interactions.copy())
        self._group_psis[self._group_mask] =  psis[self._group_mask]
        return group_activity_coefficients(x, self._chemgroups,
                                           self.loggammacs(self._qs, self._rs, x),
                                           self._Qs, psis,
                                           self._chem_Qfractions,
                                           self._group_psis)
    
    def __call__(self, x, T):
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
        return self.f(x, T, *self.args)
    
    
class UNIFACActivityCoefficients(GroupActivityCoefficients):
    """
    Create a UNIFACActivityCoefficients that estimates activity coefficients 
    using the UNIFAC group contribution method when called with a composition
    and a temperature (K).
    
    Parameters
    ----------
    
    chemicals : Iterable[:class:`~thermosteam.Chemical`]
    
    """
    all_subgroups = UFSG
    all_interactions = UFIP
    group_name = 'UNIFAC'
    _no_interaction = 0.
    _cached = {}
    
    @property
    def f(self):
        return gamma_UNIFAC
    
    @property
    def loggammacs(self):
        return loggammacs_UNIFAC
    
    @property
    def psi(self):
        return psi_UNIFAC


class DortmundActivityCoefficients(GroupActivityCoefficients):
    """
    Create a DortmundActivityCoefficients that estimates activity coefficients
    using the Dortmund UNIFAC group contribution method when called with a 
    composition and a temperature (K).
    
    Parameters
    ----------
    
    chemicals : Iterable[:class:`~thermosteam.Chemical`]
    
    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    >>> Gamma = tmo.equilibrium.DortmundActivityCoefficients(chemicals)
    >>> composition = [0.5, 0.5]
    >>> T = 350.                                
    >>> Gamma(composition, T)
    array([1.475, 1.242])
    
    >>> chemicals = tmo.Chemicals(['Dodecane', 'Tridecane'], cache=True)
    >>> Gamma = tmo.equilibrium.DortmundActivityCoefficients(chemicals)
    >>> # Note how both hydrocarbons have similar lenghts and structure,
    >>> # so activities should be very close
    >>> Gamma([0.5, 0.5], 350.) 
    array([1., 1.])
    
    >>> chemicals = tmo.Chemicals(['Water', 'O2'], cache=True)
    >>> Gamma = tmo.equilibrium.DortmundActivityCoefficients(chemicals)
    >>> # The following warning is issued because O2 does not have Dortmund groups
    >>> # RuntimeWarning: O2 has no defined Dortmund groups; 
    >>> # functional group interactions are ignored
    >>> Gamma([0.5, 0.5], 350.) 
    1.0
    
    """
    __slots__ = ()
    all_subgroups = DOUFSG
    all_interactions = DOUFIP2016
    group_name = 'Dortmund'
    _no_interaction = np.array([0., 0., 0.])
    _cached = {}
    
    @property
    def f(self):
        return gamma_modified_UNIFAC
    
    @property
    def loggammacs(self):
        return loggammacs_modified_UNIFAC
    
    @property
    def psi(self):
        return psi_modified_UNIFAC
    
    
class NISTActivityCoefficients(GroupActivityCoefficients):
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
    all_interactions = NISTUFIP
    group_name = 'NIST'
    _no_interaction = np.array([0., 0., 0.])
    _cached = {}
    
    f = DortmundActivityCoefficients.f
    
    @property
    def loggammacs(self):
        return loggammacs_modified_UNIFAC
    
    @property
    def psi(self):
        return psi_modified_UNIFAC


class GCEOSActivityCoefficients(ActivityCoefficients):
    """
    Abstract class for estimating fugacity coefficients using a generic cubic equation of state 
    when called with composition, temperature (K), and pressure (Pa).
    
    Parameters
    ----------
    chemicals : Iterable[:class:`~thermosteam.Chemical`]
    
    """
    __slots__ = ('_chemicals', '_eos')
    EOS = None # type[GCEOSMIX] Subclasses must implement this attribute.
    chemsep_db = None # Optional[str] Name of chemsep data base for interaction parameters.
    cache = None # [dict] Subclasses must implement this attribute.
    
    def __new__(cls, chemicals):
        chemicals = tuple(chemicals)
        cache = cls.cache
        if chemicals in cache:
            return cache[chemicals]
        else:
            self = super().__new__(cls)
            self._chemicals = chemicals
            cache[chemicals] = self
            return self
    
    @classmethod
    def subclass(cls, EOS, name=None):
        if name is None: name = EOS.__name__[:-3] + 'ActivityCoefficients'
        return type(name, (cls,), dict(EOS=EOS, cache={}))
    
    @property
    def chemicals(self):
        """tuple[Chemical] All chemicals involved in the calculation of fugacity coefficients."""
        return self._chemicals
    
    def eos(self, zs, T, P):
        if zs.__class__ is np.ndarray: zs = [float(i) for i in zs]
        try:
            self._eos = eos = self._eos.to_TP_zs_fast(T=T, P=P, zs=zs, only_l=True, full_alphas=False)
        except:
            data = tmo.ChemicalData(self.chemicals)
            if self.chemsep_db is None:
                kijs = None
            else:
                try:
                    kijs = IPDB.get_ip_asymmetric_matrix(self.chemsep_db, data.CASs, 'kij')
                except:
                    kijs = None
            self._eos = eos = self.EOS(
                zs=zs, T=T, P=P, Tcs=data.Tcs, Pcs=data.Pcs, omegas=data.omegas, kijs=kijs,
                only_l=True
            )
        return eos
    
    def __call__(self, x, T, P=101325):
        eos = self.eos(x, T, P)
        try:
            log_phis = np.array(eos.dlnphi_dns(eos.Z_l)) + eos.G_dep_l * R_inv / T
            return np.exp(log_phis)
        except:
            return np.ones(x.size)
    f = __call__
    args = ()
        
dct = globals()    
clsnames = []
for name in ('PRMIX', 'SRKMIX', 'PR78MIX', 'VDWMIX', 'PRSVMIX',
             'PRSV2MIX', 'TWUPRMIX', 'TWUSRKMIX', 'APISRKMIX', 'IGMIX', 'RKMIX',
             'PRMIXTranslatedConsistent', 'PRMIXTranslatedPPJP', 'PRMIXTranslated',
             'SRKMIXTranslatedConsistent', 'PSRK', 'MSRKMIXTranslated',
             'SRKMIXTranslated'):
    cls = GCEOSActivityCoefficients.subclass(getattr(eos_mix, name))
    clsname = cls.__name__
    clsnames.append(clsname)
    dct[clsname] = cls

dct['PRActivityCoefficients'].chemsep_db = 'ChemSep PR'
__all__ = (*__all__, *clsnames)
del dct, clsnames