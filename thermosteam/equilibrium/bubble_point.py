# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from numpy import asarray, array
import flexsolve as flx
from .solve_vle_composition import solve_y
from ..functional import normalize
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
        return f"{type(self).__name__}(T={self.T}, P={self.P}, IDs={self.IDs}, z={self.z}, y={self.y})"


# %% Bubble point calculation

class BubblePoint:
    """
    Create a BubblePoint object that returns bubble point values when
    called with a composition and either a temperture (T) or pressure (P).
    
    Parameters
    ----------
    chemicals=() : Iterable[Chemical], optional
    thermo=None : Thermo, optional
    
    Examples
    --------
    >>> import thermosteam as tmo
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'])
    >>> tmo.settings.set_thermo(chemicals)
    >>> BP = tmo.equilibrium.BubblePoint(chemicals)
    >>> molar_composition = (0.5, 0.5)
    >>> # Solve bubble point at constant temperature
    >>> bp = BP(z=molar_composition, T=355)
    >>> bp
    BubblePointValues(T=355, P=109755.45319869413, IDs=('Water', 'Ethanol'), z=[0.5 0.5], y=[0.343 0.657])
    >>> # Note that the result is a BubblePointValues object which contain all results as attibutes
    >>> (bp.T, bp.P, bp.IDs, bp.z, bp.y)
    (355, 109755.45319869413, ('Water', 'Ethanol'), array([0.5, 0.5]), array([0.343, 0.657]))
    >>> # Solve bubble point at constant pressure
    >>> BP(z=molar_composition, P=101325)
    BubblePointValues(T=352.95030269946596, P=101325, IDs=('Water', 'Ethanol'), z=[0.5 0.5], y=[0.342 0.658])
    
    """
    __slots__ = ('chemicals', 'IDs', 'gamma', 'phi', 'pcf',
                 'P', 'T', 'y', 'Psats', 'Tbs')
    _cached = {}
    def __init__(self, chemicals=(), thermo=None):
        thermo = settings.get_default_thermo(thermo)
        chemicals = tuple(chemicals)
        key = (chemicals, thermo)
        cached = self._cached
        if key in cached:
            other = cached[key]
            fill_like(self, other, self.__slots__)
        else:
            self.IDs = tuple([i.ID for i in chemicals])
            self.gamma = thermo.Gamma(chemicals)
            self.phi = thermo.Phi(chemicals)
            self.pcf = thermo.PCF(chemicals)
            self.Psats = [i.Psat for i in chemicals]
            self.Tbs = array([s.Tb for s in chemicals])
            self.chemicals = chemicals
            self.P = self.T = self.y = None
            cached[key] = self
    
    def _T_error(self, T, P, z_over_P, z_norm,):
        y_phi =  (z_over_P
                  * array([i(T) for i in self.Psats])
                  * self.gamma(z_norm, T) 
                  * self.pcf(z_norm, T))
        self.y = solve_y(y_phi, self.phi, T, P, self.y)
        return 1. - self.y.sum()
    
    def _P_error(self, P, T, z_Psat_gamma_pcf):
        y_phi = z_Psat_gamma_pcf / P
        self.y = solve_y(y_phi, self.phi, T, P, self.y)
        return 1. - self.y.sum()
    
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
        Bubble point at given composition and pressure

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
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'])
        >>> tmo.settings.set_thermo(chemicals)
        >>> BP = tmo.equilibrium.BubblePoint(chemicals)
        >>> BP.solve_Ty(z=np.array([0.6, 0.4]), P=101325)
        (353.7543445955407, array([0.381, 0.619]))
        
        """
        z_norm = z / z.sum()
        self.P = P
        args = (P, z_norm/P, z_norm)
        T = self.T or (z * self.Tbs).sum()
        try:
            self.T = flx.aitken_secant(self._T_error, T, T+0.01,
                                       1e-6, 5e-9, args)
        except:
            self.y = z.copy()
            T = (z * self.Tbs).sum()
            f = lambda T: self._T_error(T, *args)
            Tmin = max([i.Tmin for i in self.Psats]) + 1e-5
            Tmax = min([i.Tmax for i in self.Psats]) - 1e-5
            if Tmin < 10: Tmin = 10
            self.T = flx.IQ_interpolation(f, Tmin, Tmax,
                                          f(Tmin), f(Tmax),
                                          T, 0., 1e-6, 5e-9)
        self.y = normalize(self.y)
        return self.T, self.y.copy()
    
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
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'])
        >>> tmo.settings.set_thermo(chemicals)
        >>> BP = tmo.equilibrium.BubblePoint(chemicals)
        >>> BP.solve_Py(z=np.array([0.703, 0.297]), T=352.28)
        (91830.9798895787, array([0.419, 0.581]))
        
        """
        Psats = array([i(T) for i in self.Psats])
        z_norm = z / z.sum()
        y_phi = z * Psats * self.gamma(z_norm, T) * self.pcf(z_norm, T)
        self.T = T
        args = (T, y_phi)
        P = self.P or (z * Psats).sum()
        try:
            self.P = flx.aitken_secant(self._P_error, P, P-1,
                                       1e-3, 1e-9, args)
        except:
            self.x = z.copy()
            P = (z * Psats).sum()
            Pmin = min([i(i.Tmin + 1e-5 if i.Tmin > 10 else 10) for i in self.Psats])
            Pmax = max([i(i.Tmax - 1e-5) for i in self.Psats])
            if Pmin < 10: Pmin = 10
            self.P = flx.IQ_interpolation(self._P_error, Pmin, Pmax,
                                          x=P, args=args, xtol=1e-3, ytol=5e-9)
            
        self.y = normalize(self.y)
        return self.P, self.y.copy()
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"
    
class BubblePointCache(Cache): load = BubblePoint
del Cache
    