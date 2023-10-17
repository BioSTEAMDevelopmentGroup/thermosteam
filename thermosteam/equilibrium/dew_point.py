# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import numpy as np
import flexsolve as flx
from numba import njit
from .. import functional as fn
from ..exceptions import InfeasibleRegion
from ..utils import fill_like, Cache
from .._settings import settings
from .domain import vle_domain

__all__ = ('DewPoint', 'DewPointCache')

# %% Solvers

# @njit(cache=True)
def x_iter(x, x_gamma, T, P, f_gamma, gamma_args):
    # Add back trace amounts for activity coefficients at infinite dilution
    mask = x < 1e-32
    x[mask] = 1e-32
    x = fn.normalize(x)
    gamma = f_gamma(x, T, *gamma_args)
    denominator = gamma
    try:
        x = x_gamma / denominator
    except: 
        raise Exception('liquid phase composition is infeasible')
    if (x < 0).any():
        raise Exception('liquid phase composition is infeasible')
    mask = x > 1e3
    if mask.any():
        x[mask] = 1e3 +  np.log(x[mask] / 1e3) # Avoid crazy numbers
    return x

# @njit(cache=True)
def solve_x(x_guess, x_gamma, T, P, f_gamma, gamma_args):
    args = (x_gamma, T, P, f_gamma, gamma_args)
    x = flx.wegstein(x_iter, x_guess, 1e-10, args=args, checkiter=False,
                     checkconvergence=False, convergenceiter=3)
    return x

# %% Dew point values container

class DewPointValues:
    __slots__ = ('T', 'P', 'IDs', 'z', 'x')
    
    def __init__(self, T, P, IDs, z, x):
        self.T = T
        self.P = P
        self.IDs = IDs
        self.z = z
        self.x = x
        
    @property
    def y(self): return self.z
        
    def __repr__(self):
        return f"{type(self).__name__}(T={self.T:.2f}, P={self.P:.0f}, IDs={self.IDs}, z={self.z}, x={self.x})"


# %% Dew point calculation

class DewPoint:
    """
    Create a DewPoint object that returns dew point values when called with a 
    composition and either a temperture (T) or pressure (P).
    
    Parameters
    ----------
    chemicals=None : Iterable[:class:`~thermosteam.Chemical`], optional
    
    thermo=None : :class:`~thermosteam.Thermo`, optional
    
    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    >>> tmo.settings.set_thermo(chemicals)
    >>> DP = tmo.equilibrium.DewPoint(chemicals)
    >>> # Solve for dew point at constant temperautre
    >>> molar_composition = (0.5, 0.5)
    >>> dp = DP(z=molar_composition, T=355)
    >>> dp
    DewPointValues(T=355.00, P=92008, IDs=('Water', 'Ethanol'), z=[0.5 0.5], x=[0.849 0.151])
    >>> # Note that the result is a DewPointValues object which contain all results as attibutes
    >>> (dp.T, round(dp.P), dp.IDs, dp.z, dp.x)
    (355, 92008, ('Water', 'Ethanol'), array([0.5, 0.5]), array([0.849, 0.151]))
    >>> # Solve for dew point at constant pressure
    >>> DP(z=molar_composition, P=2*101324)
    DewPointValues(T=376.25, P=202648, IDs=('Water', 'Ethanol'), z=[0.5 0.5], x=[0.83 0.17])

    """
    __slots__ = ('chemicals', 'phi', 'gamma', 'IDs', 
                 'pcf', 'Psats', 'Tmin', 'Tmax', 'Pmin', 'Pmax')
    _cached = {}
    T_tol = 1e-9
    P_tol = 1e-3
    def __init__(self, chemicals=(), thermo=None):
        thermo = settings.get_default_thermo(thermo)
        chemicals = tuple(chemicals)
        key = (chemicals, thermo.Gamma, thermo.Phi, thermo.PCF)
        cached = self._cached
        if key in cached:
            other = cached[key]
            fill_like(self, other, self.__slots__)
        else:
            self.IDs = tuple([i.ID for i in chemicals])
            self.gamma = thermo.Gamma(chemicals)
            self.phi = thermo.Phi(chemicals)
            self.pcf = thermo.PCF(chemicals)
            self.Psats = Psats = [i.Psat for i in chemicals]
            Tmin, Tmax = vle_domain(chemicals)
            self.Tmin = Tmin
            self.Tmax = Tmax
            self.Pmin = min([i(Tmin) for i in Psats])
            self.Pmax = max([i(Tmax) for i in Psats])
            self.chemicals = chemicals
            cached[key] = self
    
    def _solve_x(self, x_gamma, T, P, x):
        gamma = self.gamma
        return solve_x(x, x_gamma, T, P, gamma.f, gamma.args)
    
    def _T_error(self, T, P, z_norm, zP, x):
        if T <= 0: raise InfeasibleRegion('negative temperature')
        Psats = np.array([i(T) for i in self.Psats])
        Psats[Psats < 1e-16] = 1e-16 # Prevent floating point error
        phi = self.phi(z_norm, T, P)
        pcf = self.pcf(T, P, Psats)
        x_gamma = phi * zP / Psats / pcf
        x[:] = self._solve_x(x_gamma, T, P, x)
        return 1 - x.sum()
    
    def _T_error_ideal(self, T, zP, x):
        Psats = np.array([i(T) for i in self.Psats])
        Psats[Psats < 1e-16] = 1e-16 # Prevent floating point error
        x[:] = zP / Psats
        return 1 - x.sum()
    
    def _P_error(self, P, T, z_norm, z_over_Psats, Psats, x):
        if P <= 0: raise InfeasibleRegion('negative pressure')
        x_gamma = z_over_Psats * P * self.phi(z_norm, T, P) / self.pcf(T, P, Psats)
        x[:] = self._solve_x(x_gamma, T, P, x)
        return 1 - x.sum()
    
    def _Tx_ideal(self, zP):
        f = self._T_error_ideal
        Tmin = self.Tmin + 10.
        Tmax = self.Tmax - 10.
        x = zP.copy() 
        args = (zP, x)
        fmin = f(Tmin, *args)
        if fmin > 0.: return Tmin, x
        fmax = f(Tmax, *args)
        if fmax < 0.: return Tmax, x
        T = flx.IQ_interpolation(f, Tmin, Tmax, fmin, fmax, 
                                 None, self.T_tol, 5e-12, args,
                                 checkiter=False, checkbounds=False)
        return T, x
    
    def _Px_ideal(self, z_over_Psats):
        P = 1. / z_over_Psats.sum()
        x = z_over_Psats * P
        return P, x
    
    def __call__(self, z, *, T=None, P=None):
        z = np.asarray(z, float)
        if T:
            if P: raise ValueError("may specify either T or P, not both")
            P, x = self.solve_Px(z, T)
        elif P:
            T, x = self.solve_Tx(z, P)
        else:
            raise ValueError("must specify either T or P")
        return DewPointValues(T, P, self.IDs, z, x)
    
    def solve_Tx(self, z, P):
        """
        Dew point given composition and pressure.

        Parameters
        ----------
        z : ndarray
            Molar composition.
        P : float
            Pressure [Pa].

        Returns
        -------
        T : float
            Dew point temperature [K].
        x : numpy.ndarray
            Liquid phase molar composition.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> import numpy as np
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
        >>> tmo.settings.set_thermo(chemicals)
        >>> DP = tmo.equilibrium.DewPoint(chemicals)
        >>> tmo.docround(DP.solve_Tx(z=np.array([0.5, 0.5]), P=101325))
        (357.4419, array([0.847, 0.153]))
        
        """
        positives = z > 0.
        N = positives.sum()
        if N == 0:
            raise ValueError('no positive components present')
        if N == 1:
            chemical = self.chemicals[fn.first_true_index(positives)]
            T = chemical.Tsat(P, check_validity=False) if P <= chemical.Pc else chemical.Tc
            x = z.copy()
        else:
            f = self._T_error
            z_norm = z/z.sum()
            zP = z * P
            T_guess, x = self._Tx_ideal(zP) 
            args = (P, z_norm, zP, x)
            try:
                T = flx.aitken_secant(f, T_guess, T_guess + 1e-3,
                                      self.T_tol, 5e-12, args,
                                      checkiter=False)
            except RuntimeError:
                Tmin = self.Tmin
                Tmax = self.Tmax
                T = flx.IQ_interpolation(f, Tmin, Tmax,
                                         f(Tmin, *args), f(Tmax, *args),
                                         T_guess, 1e-9, 5e-12, args,
                                         checkiter=False, checkbounds=False)
        return T, fn.normalize(x)
    
    def solve_Px(self, z, T):
        """
        Dew point given composition and temperature.

        Parameters
        ----------
        z : ndarray
            Molar composition.
        T : float
            Temperature (K).
        
        Returns
        -------
        P : float
            Dew point pressure (Pa).
        x : ndarray
            Liquid phase molar composition.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> import numpy as np
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
        >>> tmo.settings.set_thermo(chemicals)
        >>> DP = tmo.equilibrium.DewPoint(chemicals)
        >>> tmo.docround(DP.solve_Px(z=np.array([0.5, 0.5]), T=352.28))
        (82480.7363, array([0.851, 0.149]))
 
        """
        positives = z > 0.
        N = positives.sum()
        if N == 0:
            raise ValueError('no positive components present')
        if N == 1:
            chemical = self.chemicals[fn.first_true_index(z)]
            P = chemical.Psat(T) if T <= chemical.Tc else chemical.Pc
            x = z.copy()
        else:
            z_norm = z/z.sum()
            Psats = np.array([i(T) for i in self.Psats], dtype=float)
            z_over_Psats = z/Psats
            P_guess, x = self._Px_ideal(z_over_Psats)
            args = (T, z_norm, z_over_Psats, Psats, x)
            f = self._P_error
            try:
                P = flx.aitken_secant(f, P_guess, P_guess-10, self.P_tol, 5e-12, args,
                                      checkiter=False)
            except RuntimeError:
                Pmin = self.Pmin
                Pmax = self.Pmax
                P = flx.IQ_interpolation(f, Pmin, Pmax, 
                                         f(Pmin, *args), f(Pmax, *args),
                                         P_guess, self.P_tol, 5e-12, args,
                                         checkiter=False, checkbounds=False)
        return P, fn.normalize(x)
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"
    
class DewPointCache(Cache): load = DewPoint
del Cache
