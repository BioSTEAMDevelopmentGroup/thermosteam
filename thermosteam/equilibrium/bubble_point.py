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
from .domain import vle_domain
from ..exceptions import InfeasibleRegion, DomainError
from .solve_vle_composition import solve_y
from .. import functional as fn
from ..utils import fill_like, Cache
from .._settings import settings

__all__ = ('BubblePoint', 'BubblePointValues', 'BubblePointCache')

# %% Bubble point values container

class BubblePointValues:
    __slots__ = ('T', 'P', 'IDs', 'z', 'y')
    
    def __init__(self, T, P, IDs, z, y):
        self.T = T
        self.P = P
        self.IDs = IDs
        self.z = z
        self.y = y
        
    def __repr__(self):
        return f"{type(self).__name__}(T={self.T:.2f}, P={self.P:.0f}, IDs={self.IDs}, z={self.z}, y={self.y})"


# %% Bubble point calculation

class BubblePoint:
    """
    Create a BubblePoint object that returns bubble point values when
    called with a composition and either a temperture (T) or pressure (P).
    
    Parameters
    ----------
    chemicals=() : Iterable[:class:`~thermosteam.Chemical`], optional
    thermo=None : :class:`~thermosteam.Thermo`, optional
    
    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    >>> tmo.settings.set_thermo(chemicals)
    >>> BP = tmo.equilibrium.BubblePoint(chemicals)
    >>> molar_composition = (0.5, 0.5)
    >>> # Solve bubble point at constant temperature
    >>> bp = BP(z=molar_composition, T=355)
    >>> bp
    BubblePointValues(T=355.00, P=109755, IDs=('Water', 'Ethanol'), z=[0.5 0.5], y=[0.343 0.657])
    >>> # Note that the result is a BubblePointValues object which contain all results as attibutes
    >>> (bp.T, round(bp.P), bp.IDs, bp.z, bp.y)
    (355, 109755, ('Water', 'Ethanol'), array([0.5, 0.5]), array([0.343, 0.657]))
    >>> # Solve bubble point at constant pressure
    >>> BP(z=molar_composition, P=101325)
    BubblePointValues(T=352.95, P=101325, IDs=('Water', 'Ethanol'), z=[0.5 0.5], y=[0.342 0.658])
    
    """
    __slots__ = ('chemicals', 'IDs', 'gamma', 'phi', 'pcf',
                 'Psats', 'Tmin', 'Tmax', 'Pmin', 'Pmax')
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
            cached[key] = self
    
    def _T_error(self, T, P, z_over_P, z_norm, y):
        if T <= 0: raise InfeasibleRegion('negative temperature')
        y_phi =  (z_over_P
                  * array([i(T) for i in self.Psats])
                  * self.gamma(z_norm, T) 
                  * self.pcf(z_norm, T))
        y[:] = solve_y(y_phi, self.phi, T, P, y)
        return 1. - y.sum()
    
    def _P_error(self, P, T, z_Psat_gamma_pcf, y):
        if P <= 0: raise InfeasibleRegion('negative pressure')
        y_phi = z_Psat_gamma_pcf / P
        y[:] = solve_y(y_phi, self.phi, T, P, y)
        return 1. - y.sum()
        
    def _T_error_ideal(self, T, z_over_P, y):
        y[:] = z_over_P * array([i(T) for i in self.Psats])
        return 1 - y.sum()
    
    def _Ty_ideal(self, z_over_P):
        f = self._T_error_ideal
        y = z_over_P.copy()
        args = (z_over_P, y)
        Tmin = self.Tmin
        Tmax = self.Tmax
        T = flx.IQ_interpolation(f, Tmin, Tmax,
                                 f(Tmin, *args), f(Tmax, *args),
                                 None, 1e-9, 5e-12, args, 
                                 checkiter=False,
                                 checkbounds=False)
        return T, y
    
    def _Py_ideal(self, z_Psat_gamma_pcf):
        P = z_Psat_gamma_pcf.sum()
        y = z_Psat_gamma_pcf / P
        return P, y
    
    def __call__(self, z, *, T=None, P=None):
        z = asarray(z, float)
        if T:
            if P: raise ValueError("may specify either T or P, not both")
            P, y = self.solve_Py(z, T)
        elif P:
            T, y = self.solve_Ty(z, P)
        else:
            raise ValueError("must specify either T or P")
        return BubblePointValues(T, P, self.IDs, z, y)
    
    def solve_Ty(self, z, P):
        """
        Bubble point at given composition and pressure.

        Parameters
        ----------
        z : ndarray
            Molar composotion.
        P : float
            Pressure [Pa].
        
        Returns
        -------
        T : float 
            Bubble point temperature [K].
        y : ndarray
            Vapor phase molar composition.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> import numpy as np
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
        >>> tmo.settings.set_thermo(chemicals)
        >>> BP = tmo.equilibrium.BubblePoint(chemicals)
        >>> BP.solve_Ty(z=np.array([0.6, 0.4]), P=101325)
        (353.7543, array([0.381, 0.619]))
        
        """
        if P > self.Pmax: P = self.Pmax
        elif P < self.Pmin: P = self.Pmin
        f = self._T_error
        z_norm = z / z.sum()
        z_over_P = z/P
        T_guess, y = self._Ty_ideal(z_over_P)
        args = (P, z_over_P, z_norm, y)
        try:
            T = flx.aitken_secant(f, T_guess, T_guess + 1e-3,
                                  1e-9, 5e-12, args,
                                  checkiter=False)
        except (InfeasibleRegion, DomainError):
            Tmin = self.Tmin; Tmax = self.Tmax
            T = flx.IQ_interpolation(f, Tmin, Tmax,
                                     f(Tmin, *args), f(Tmax, *args),
                                     T_guess, 1e-9, 5e-12, args, 
                                     checkiter=False, checkbounds=False)
        return T, fn.normalize(y)
    
    def solve_Py(self, z, T):
        """
        Bubble point at given composition and temperature.

        Parameters
        ----------
        z : ndarray
            Molar composotion.
        T : float
            Temperature [K].
        
        Returns
        -------
        P : float
            Bubble point pressure [Pa].
        y : ndarray
            Vapor phase molar composition.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> import numpy as np
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
        >>> tmo.settings.set_thermo(chemicals)
        >>> BP = tmo.equilibrium.BubblePoint(chemicals)
        >>> BP.solve_Py(z=np.array([0.703, 0.297]), T=352.28)
        (91830.9798, array([0.419, 0.581]))
        
        """
        if T > self.Tmax: T = self.Tmax
        elif T < self.Tmin: T = self.Tmin
        Psat = array([i(T) for i in self.Psats])
        z_norm = z / z.sum()
        z_Psat_gamma_pcf = z * Psat * self.gamma(z_norm, T) * self.pcf(z_norm, T)
        f = self._P_error
        P_guess, y = self._Py_ideal(z_Psat_gamma_pcf)
        args = (T, z_Psat_gamma_pcf, y)
        try:
            P = flx.aitken_secant(f, P_guess, P_guess-1, 1e-3, 1e-9,
                                  args, checkiter=False)
        except (InfeasibleRegion, DomainError):
            Pmin = self.Pmin; Pmax = self.Pmax
            P = flx.IQ_interpolation(f, Pmin, Pmax,
                                     f(Pmin, *args), f(Pmax, *args),
                                     P_guess, 1e-3, 5e-12, args,
                                     checkiter=False, checkbounds=False)
        return P, fn.normalize(y)
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"
    
class BubblePointCache(Cache): load = BubblePoint
del Cache
    