# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import numpy as np
import flexsolve as flx
from .fugacity_coefficients import IdealFugacityCoefficients
from .domain import vle_domain
from ..exceptions import InfeasibleRegion, DomainError
from .. import functional as fn
from ..utils import fill_like, Cache
from .._settings import settings

__all__ = (
    'BubblePoint', 'BubblePointValues', 'BubblePointCache',
    'BubblePointBeta'
)

# %% Solvers

def y_iter(y, y_phi, phi, T, P):
    y = fn.normalize(y)
    return y_phi / phi(y, T, P)

def solve_y(y_phi, phi, T, P, y_guess):
    if isinstance(phi, IdealFugacityCoefficients): return y_phi
    return flx.wegstein(y_iter, y_phi, 1e-9, args=(y_phi, phi, T, P), 
                        checkiter=False,
                        checkconvergence=False, 
                        convergenceiter=3)


# %% Bubble point values container

class BubblePointValues:
    __slots__ = ('T', 'P', 'IDs', 'z', 'y')
    
    def __init__(self, T, P, IDs, z, y):
        self.T = T
        self.P = P
        self.IDs = IDs
        self.z = z
        self.y = y
        
    @property
    def x(self): return self.z
        
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
    BubblePointValues(T=355.00, P=109811, IDs=('Water', 'Ethanol'), z=[0.5 0.5], y=[0.343 0.657])
    >>> # Note that the result is a BubblePointValues object which contain all results as attibutes
    >>> (bp.T, round(bp.P), bp.IDs, bp.z, bp.y)
    (355, 109811, ('Water', 'Ethanol'), array([0.5, 0.5]), array([0.343, 0.657]))
    >>> # Solve bubble point at constant pressure
    >>> BP(z=molar_composition, P=101325)
    BubblePointValues(T=352.95, P=101325, IDs=('Water', 'Ethanol'), z=[0.5 0.5], y=[0.342 0.658])
    
    """
    __slots__ = ('chemicals', 'IDs', 'gamma', 'phi', 'pcf',
                 'Psats', 'Tmin', 'Tmax', 'Pmin', 'Pmax')
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
    
    def _T_error(self, T, P, z_over_P, z_norm, y):
        if T <= 0: raise InfeasibleRegion('negative temperature')
        y_phi =  (z_over_P
                  * np.array([i(T) for i in self.Psats])
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
        y[:] = z_over_P * np.array([i(T) for i in self.Psats])
        return 1 - y.sum()
    
    def _Ty_ideal(self, z_over_P):
        f = self._T_error_ideal
        y = z_over_P.copy()
        args = (z_over_P, y)
        Tmin = self.Tmin
        Tmax = self.Tmax
        fmax = f(Tmin, *args)
        if fmax < 0.: return Tmin, y
        fmin = f(Tmax, *args)
        if fmin > 0.: return Tmax, y
        T = flx.IQ_interpolation(f, Tmin, Tmax, fmax, fmin, 
                                 None, self.T_tol, 5e-12, args, 
                                 checkiter=False,
                                 checkbounds=False)
        return T, y
    
    def _Py_ideal(self, z_Psat_gamma_pcf):
        P = z_Psat_gamma_pcf.sum()
        y = z_Psat_gamma_pcf / P
        return P, y
    
    def __call__(self, z, *, T=None, P=None):
        z = np.asarray(z, float)
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
            Molar composition.
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
        (353.73987267109044, array([0.381, 0.619]))
        
        """
        positives = z > 0.
        N = positives.sum()
        if N == 0:
            raise ValueError('no components present')
        if N == 1:
            T = self.chemicals[fn.first_true_index(positives)].Tsat(P)
            y = z.copy()
        else:
            if P > self.Pmax: P = self.Pmax
            elif P < self.Pmin: P = self.Pmin
            f = self._T_error
            z_norm = z / z.sum()
            z_over_P = z/P
            T_guess, y = self._Ty_ideal(z_over_P)
            args = (P, z_over_P, z_norm, y)
            try:
                T = flx.aitken_secant(f, T_guess, T_guess + 1e-3,
                                      self.T_tol, 5e-12, args,
                                      checkiter=False)
            except (InfeasibleRegion, DomainError):
                Tmin = self.Tmin; Tmax = self.Tmax
                T = flx.IQ_interpolation(f, Tmin, Tmax,
                                         f(Tmin, *args), f(Tmax, *args),
                                         T_guess, self.T_tol, 5e-12, args, 
                                         checkiter=False, checkbounds=False)
        return T, fn.normalize(y)
    
    def solve_Py(self, z, T):
        """
        Bubble point at given composition and temperature.

        Parameters
        ----------
        z : ndarray
            Molar composition.
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
        >>> tmo.docround(BP.solve_Py(z=np.array([0.703, 0.297]), T=352.28))
        (91888.1429, array([0.419, 0.581]))
        
        """
        positives = z > 0.
        N = positives.sum()
        if N == 0:
            raise ValueError('no components present')
        if N == 1:
            P = self.chemicals[fn.first_true_index(positives)].Psat(T)
            y = z.copy()
        else:
            if T > self.Tmax: T = self.Tmax
            elif T < self.Tmin: T = self.Tmin
            Psat = np.array([i(T) for i in self.Psats])
            z_norm = z / z.sum()
            z_Psat_gamma_pcf = z * Psat * self.gamma(z_norm, T) * self.pcf(z_norm, T)
            f = self._P_error
            P_guess, y = self._Py_ideal(z_Psat_gamma_pcf)
            args = (T, z_Psat_gamma_pcf, y)
            try:
                P = flx.aitken_secant(f, P_guess, P_guess-1, self.P_tol, 1e-9,
                                      args, checkiter=False)
            except (InfeasibleRegion, DomainError):
                Pmin = self.Pmin; Pmax = self.Pmax
                P = flx.IQ_interpolation(f, Pmin, Pmax,
                                         f(Pmin, *args), f(Pmax, *args),
                                         P_guess, self.P_tol, 5e-12, args,
                                         checkiter=False, checkbounds=False)
        return P, fn.normalize(y)
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"
    
class BubblePointBeta:
    """
    Create a BubblePointBeta object that returns bubble point values when
    called with a composition and either a temperture (T) or pressure (P).
    
    Parameters
    ----------
    flasher=None : :class:`~thermo.Flash`, optional
    
    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    >>> tmo.settings.set_thermo(chemicals)
    >>> BP = tmo.equilibrium.BubblePointBeta(chemicals)
    >>> molar_composition = (0.5, 0.5)
    >>> # Solve bubble point at constant temperature
    >>> bp = BP(z=molar_composition, T=355)
    >>> bp
    BubblePointValues(T=355.00, P=111889, IDs=['Water', 'Ethanol'], z=[0.5 0.5], y=[0.34 0.66])
    >>> # Note that the result is a BubblePointValues object which contain all results as attibutes
    >>> (bp.T, round(bp.P), bp.IDs, bp.z, bp.y)
    (355, 111889, ['Water', 'Ethanol'], array([0.5, 0.5]), array([0.34, 0.66]))
    >>> # Solve bubble point at constant pressure
    >>> BP(z=molar_composition, P=101325)
    BubblePointValues(T=352.50, P=101325, IDs=['Water', 'Ethanol'], z=[0.5 0.5], y=[0.339 0.661])
    
    """
    __slots__ = ('chemicals', 'IDs', 'flasher')
    _cached = {}
    def __init__(self, chemicals=(), flasher=None):
        self.chemicals = chemicals
        self.IDs = [i.ID for i in chemicals]
        self.flasher = flasher or settings.flasher()
    
    __call__ = BubblePoint.__call__
    
    def solve_Ty(self, z, P):
        """
        Bubble point at given composition and pressure.

        Parameters
        ----------
        z : ndarray
            Molar composition.
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
        >>> BP = tmo.equilibrium.BubblePointBeta(chemicals)
        >>> tmo.docround(BP.solve_Ty(z=np.array([0.6, 0.4]), P=101325))
        (353.3124, array([0., 1.]))
        
        """
        positives = z > 0.
        N = positives.sum()
        if N == 0:
            raise ValueError('no components present')
        if N == 1:
            T = self.chemicals.tuple[fn.first_true_index(positives)].Tsat(P)
            y = z.copy()
        else:
            results = self.flasher.flash(P=P, VF=0., zs=z.tolist())
            y = np.array(results.gas.zs)
            T = results.T
        return T, fn.normalize(y)
    
    def solve_Py(self, z, T):
        """
        Bubble point at given composition and temperature.

        Parameters
        ----------
        z : ndarray
            Molar composition.
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
        >>> tmo.docround(BP.solve_Py(z=np.array([0.703, 0.297]), T=352.28))
        (91888.1429, array([0., 1.]))
        
        """
        positives = z > 0.
        N = positives.sum()
        if N == 0:
            raise ValueError('no components present')
        if N == 1:
            T = self.chemicals.tuple[fn.first_true_index(positives)].Psat(T)
            y = z.copy()
        else:
            results = self.flasher.flash(T=T, VF=0., zs=z.tolist())
            y = np.array(results.gas.zs)
            P = results.P
        return P, fn.normalize(y)
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"
    
class BubblePointCache(Cache): load = BubblePoint
del Cache
    
