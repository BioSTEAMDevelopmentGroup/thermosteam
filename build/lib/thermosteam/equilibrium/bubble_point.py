# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:30:33 2019

@author: yoelr
"""
from numpy import asarray, array
from flexsolve import aitken_secant, IQ_interpolation
from .solve_vle_composition import solve_y
from ..utils import fill_like
from .._settings import settings

__all__ = ('BubblePoint', 'BubblePointValues')

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
    >>> import numpy as np
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'])
    >>> tmo.settings.set_thermo(chemicals)
    >>> BP = tmo.equilibrium.BubblePoint(chemicals)
    >>> molar_composition = np.array([0.5, 0.5])
    >>> # Solve bubble point at constant temperature
    >>> bp = BP(z=molar_composition, T=355)
    >>> bp
    BubblePointValues(T=355, P=109755.45319868939, IDs=('Water', 'Ethanol'), z=[0.5 0.5], y=[0.343 0.657])
    >>> # Note that the result is a BubblePointValues object which contain all results as attibutes
    >>> (bp.T, bp.P, bp.IDs, bp.z, bp.y)
    (355, 109755.45319868939, ('Water', 'Ethanol'), array([0.5, 0.5]), array([0.343, 0.657]))
    >>> # Solve bubble point at constant pressure
    >>> BP(z=molar_composition, P=101325)
    BubblePointValues(T=352.95030269946596, P=101325, IDs=('Water', 'Ethanol'), z=[0.5 0.5], y=[0.342 0.658])
    
    """
    __slots__ = ('chemicals', 'IDs', 'gamma', 'phi', 'pcf',
                 'P', 'T', 'y', 'Psats', 'Tbs')
    rootsolver = staticmethod(aitken_secant)
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
        if T:
            if P: raise ValueError("may specify either T or P, not both")
            P, y = self.solve_Py(z, T)
        elif P:
            T, y = self.solve_Ty(z, P)
        else:
            raise ValueError("must specify either T or P")
        return BubblePointValues(T, P, self.IDs, z, y)
    
    def solve_Ty(self, z, P):
        """Bubble point at given composition and pressure

        Parameters
        ----------
        z : array_like
            Molar composotion.
        P : float
            Pressure [Pa].
        
        Returns
        -------
        T : float 
            Bubble point temperature [K].
        y : numpy.ndarray
            Vapor phase molar composition.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'])
        >>> tmo.settings.set_thermo(chemicals)
        >>> BP = tmo.equilibrium.BubblePoint(chemicals)
        >>> BP.solve_Ty(z=(0.6, 0.4), P=101325)
        (353.7543445955407, array([0.381, 0.619]))
        
        """
        
        z = asarray(z)
        z_norm = z / z.sum()
        self.P = P
        args = (P, z/P, z_norm)
        T = self.T or (z * self.Tbs).sum()
        try:
            self.T = self.rootsolver(self._T_error, T, T+0.01,
                                     1e-6, 5e-8, args)
        except:
            self.y = z.copy()
            T = (z * self.Tbs).sum()
            f = lambda T: self._T_error(T, *args)
            Tmin = max([i.Tmin for i in self.Psats]) + 1e-5
            Tmax = min([i.Tmax for i in self.Psats]) - 1e-5
            if Tmin < 10: Tmin = 10
            self.T = IQ_interpolation(f, Tmin, Tmax,
                                      f(Tmin), f(Tmax),
                                      T, 0., 1e-6, 5e-8)
        self.y /= self.y.sum()
        return self.T, self.y.copy()
    
    def solve_Py(self, z, T):
        """Bubble point at given composition and temperature.

        Parameters
        ----------
        z : array_like
            Molar composotion.
        T : float
            Temperature [K].
        
        Returns
        -------
        P : float
            Bubble point pressure [Pa].
        y : numpy.ndarray
            Vapor phase molar composition.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'])
        >>> tmo.settings.set_thermo(chemicals)
        >>> BP = tmo.equilibrium.BubblePoint(chemicals)
        >>> BP.solve_Py(z=(0.703, 0.297), T=352.28)
        (91830.97988957923, array([0.419, 0.581]))
        
        """
        z = asarray(z)
        Psat = array([i(T) for i in self.Psats])
        z_norm = z / z.sum()
        y_phi = z * Psat * self.gamma(z_norm, T) * self.pcf(z_norm, T)
        self.T = T
        args = (T, y_phi)
        P = self.P or (z * Psat).sum()
        try:
            self.P = self.rootsolver(self._P_error, P, P-1,
                                     1e-2, 5e-8, args)
        except:
            P = (z * Psat).sum()
            self.P = self.rootsolver(self._P_error, P, P-1,
                                     1e-2, 5e-8, args)
        self.y /= self.y.sum()
        return self.P, self.y.copy()
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"
    

    