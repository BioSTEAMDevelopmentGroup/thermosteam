# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from numpy import asarray, array
import flexsolve as flx
from .. import functional as fn
from ..exceptions import DomainError, InfeasibleRegion
from .solve_vle_composition import solve_x
from ..utils import fill_like, Cache
from .._settings import settings
from .domain import vle_domain

__all__ = ('DewPoint', 'DewPointCache')

# %% Dew point values container

class DewPointValues:
    __slots__ = ('T', 'P', 'IDs', 'z', 'x')
    
    def __init__(self, T, P, IDs, z, x):
        self.T = T
        self.P = P
        self.IDs = IDs
        self.z = z
        self.x = x
        
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
    DewPointValues(T=355.00, P=91970, IDs=('Water', 'Ethanol'), z=[0.5 0.5], x=[0.851 0.149])
    >>> # Note that the result is a DewPointValues object which contain all results as attibutes
    >>> (dp.T, round(dp.P), dp.IDs, dp.z, dp.x)
    (355, 91970, ('Water', 'Ethanol'), array([0.5, 0.5]), array([0.851, 0.149]))
    >>> # Solve for dew point at constant pressure
    >>> DP(z=molar_composition, P=2*101324)
    DewPointValues(T=376.26, P=202648, IDs=('Water', 'Ethanol'), z=[0.5 0.5], x=[0.832 0.168])

    """
    __slots__ = ('chemicals', 'phi', 'gamma', 'IDs',
                 'pcf', 'Psats', 'P', 'T', 'x',
                 'Tmin', 'Tmax', 'Pmin', 'Pmax')
    Tmin_default = 150.
    _cached = {}
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
            self.P = self.T = self.x = None
            cached[key] = self
    
    def _T_error(self, T, P, z_norm, zP):
        if T <= 0: raise InfeasibleRegion('negative temperature')
        Psats = array([i(T) for i in self.Psats])
        Psats[Psats < 1e-16] = 1e-16 # Prevent floating point error
        phi = self.phi(z_norm, T, P)
        x_gamma_pcf = phi * zP / Psats
        self.x = solve_x(x_gamma_pcf, self.gamma, self.pcf, T, self.x)
        return 1 - self.x.sum()
    
    def _T_error_ideal(self, T, zP):
        Psats = array([i(T) for i in self.Psats])
        Psats[Psats < 1e-16] = 1e-16 # Prevent floating point error
        self.x = zP / Psats
        return 1 - self.x.sum()
    
    def _P_error(self, P, T, z_norm, z_over_Psats):
        if P <= 0: raise InfeasibleRegion('negative pressure')
        x_gamma_pcf = z_over_Psats * P * self.phi(z_norm, T, P)
        self.x = solve_x(x_gamma_pcf, self.gamma, self.pcf, T, self.x)
        return 1 - self.x.sum()
    
    def _T_ideal(self, zP):
        f = self._T_error_ideal
        Tmin = self.Tmin
        Tmax = self.Tmax
        args = (zP,)
        T = flx.IQ_interpolation(f, Tmin, Tmax,
                                 f(Tmin, *args), f(Tmax, *args),
                                 None, 1e-9, 5e-12, args,
                                 checkiter=False, checkbounds=False)
        return T
    
    def _P_ideal(self, z_over_Psats):
        return 1. / z_over_Psats.sum()
    
    def __call__(self, z, *, T=None, P=None):
        z = asarray(z, float)
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
        >>> DP.solve_Tx(z=np.array([0.5, 0.5]), P=101325)
        (357.451847, array([0.849, 0.151]))
        
        """
        if P > self.Pmax: P = self.Pmax
        elif P < self.Pmin: P = self.Pmin
        f = self._T_error
        z_norm = z/z.sum()
        zP = z * P
        args = (P, z_norm, zP)
        self.P = P
        T_guess = self._T_ideal(zP) 
        try:
            T = flx.aitken_secant(f, T_guess, T_guess + 1e-3,
                                  1e-9, 5e-12, args,
                                  checkiter=False)
        except (InfeasibleRegion, DomainError):
            Tmin = self.Tmin
            Tmax = self.Tmax
            T = flx.IQ_interpolation(f, Tmin, Tmax,
                                     f(Tmin, *args), f(Tmax, *args),
                                     T_guess, 1e-9, 5e-12, args,
                                     checkiter=False, checkbounds=False)
        self.x = fn.normalize(self.x)
        return T, self.x.copy()
    
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
        >>> DP.solve_Px(z=np.array([0.5, 0.5]), T=352.28)
        (82444.29876, array([0.853, 0.147]))
 
       """
        if T > self.Tmax: T = self.Tmax
        elif T < self.Tmin: T = self.Tmin
        z_norm = z/z.sum()
        Psats = array([i(T) for i in self.Psats], dtype=float)
        z_over_Psats = z/Psats
        args = (T, z_norm, z_over_Psats)
        self.T = T
        f = self._P_error
        P_guess = self._P_ideal(z_over_Psats)
        try:
            P = flx.aitken_secant(f, P_guess, P_guess-10, 1e-3, 5e-12, args,
                                  checkiter=False)
        except (InfeasibleRegion, DomainError):
            Pmin = self.Pmin
            Pmax = self.Pmax
            P = flx.IQ_interpolation(f, Pmin, Pmax, 
                                     f(Pmin, *args), f(Pmax, *args),
                                     P_guess, 1e-3, 5e-12, args,
                                     checkiter=False, checkbounds=False)
        self.x = fn.normalize(self.x)
        return P, self.x.copy()
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"
    
class DewPointCache(Cache): load = DewPoint
del Cache