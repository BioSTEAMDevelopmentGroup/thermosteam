# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:15:30 2019

@author: yoelr
"""

from numpy import asarray, array
from flexsolve import aitken_secant, IQ_interpolation
from .solve_composition import solve_x
from ..utils import fill_like
from .._settings import settings

__all__ = ('DewPoint',)

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
        return f"{type(self).__name__}(T={self.T}, P={self.P}, IDs={self.IDs}, z={self.z}, x={self.x})"


# %% Dew point calculation

class DewPoint:
    """
    Create a DewPoint object that returns dew point values when called with a 
    composition and either a temperture (T) or pressure (P).
    
    Parameters
    ----------
    chemicals=None : Iterable[Chemical], optional
    
    thermo=None : Thermo, optional
    
    Examples
    --------
    >>> import thermosteam as tmo
    >>> import numpy as np
    >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'])
    >>> tmo.settings.set_thermo(chemicals)
    >>> DP = tmo.equilibrium.DewPoint(chemicals)
    >>> # Solve for dew point at constant temperautre
    >>> molar_composition = np.array([0.5, 0.5])
    >>> dp = DP(z=molar_composition, T=355)
    >>> dp
    DewPointValues(T=355, P=91970.14968399677, IDs=('Water', 'Ethanol'), z=[0.5 0.5], x=[0.851 0.149])
    >>> # Note that the result is a DewPointValues object which contain all results as attibutes
    >>> (dp.T, dp.P, dp.IDs, dp.z, dp.x)
    (355, 91970.14968399677, ('Water', 'Ethanol'), array([0.5, 0.5]), array([0.851, 0.149]))
    >>> # Solve for dew point at constant pressure
    >>> DP(z=molar_composition, P=2*101324)
    DewPointValues(T=376.261660024686, P=202648, IDs=('Water', 'Ethanol'), z=[0.5 0.5], x=[0.832 0.168])

    """
    __slots__ = ('chemicals', 'phi', 'gamma', 'IDs',
                 'pcf', 'Psats', 'Tbs', 'P', 'T', 'x')
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
            self.P = self.T = self.x = None
            cached[key] = self
    
    def _T_error(self, T, P, z_norm, zP):
        Psats = array([i(T) for i in self.Psats])
        phi = self.phi(z_norm, T, P)
        x_gamma_pcf = phi * zP / Psats
        self.x = solve_x(x_gamma_pcf, self.gamma, self.pcf, T, self.x)
        return 1 - self.x.sum()
    
    def _P_error(self, P, T, z_norm, z_over_Psats):
        x_gamma_pcf = z_over_Psats*P*self.phi(z_norm, T, P)
        self.x = solve_x(x_gamma_pcf, self.gamma, self.pcf, T, self.x)
        return 1 - self.x.sum()
    
    def __call__(self, z, *, T=None, P=None):
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
        z : array_like
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
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'])
        >>> tmo.settings.set_thermo(chemicals)
        >>> DP = tmo.equilibrium.DewPoint(chemicals)
        >>> DP.solve_Tx(z=(0.5, 0.5), P=101325)
        (357.45184743325217, array([0.849, 0.151]))
        
        """
        z = asarray(z)
        z_norm = z/z.sum()
        zP = z * P
        args = (P, z_norm, zP)
        self.P = P
        T = self.T or (z * self.Tbs).sum()
        try:
            self.T = self.rootsolver(self._T_error, T, T-0.01,
                                     1e-6, 5e-8, args)
        except:
            self.x = z.copy()
            T = (z * self.Tbs).sum()
            Tmin = max([i.Tmin for i in self.Psats]) + 1e-5
            Tmax = min([i.Tmax for i in self.Psats]) - 1e-5
            if Tmin < 10: Tmin = 10
            self.T = IQ_interpolation(self._T_error, Tmin, Tmax,
                                      x=T, args=args, xtol=1e-4, ytol=1e-4)
                
        self.x /= self.x.sum()
        return self.T, self.x
    
    def solve_Px(self, z, T):
        """
        Dew point given composition and temperature.

        Parameters
        ----------
        z : array_like
            Molar composition.
        T : float
            Temperature (K).
        
        Returns
        -------
        P : float
            Dew point pressure (Pa).
        x : numpy.ndarray
            Liquid phase molar composition.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> import numpy as np
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'])
        >>> tmo.settings.set_thermo(chemicals)
        >>> DP = tmo.equilibrium.DewPoint(chemicals)
        >>> DP.solve_Px(z=(0.5, 0.5), T=352.28)
        (82444.29876048012, array([0.853, 0.147]))
 
       """
        z = asarray(z)
        z_norm = z/z.sum()
        Psats = array([i(T) for i in self.Psats], dtype=float)
        z_over_Psats = z/Psats
        args = (T, z_norm, z_over_Psats)
        self.T = T
        P = self.P or (z * Psats).sum()
        try:
            self.P = self.rootsolver(self._P_error, P, P+1,
                                     1e-2, 5e-8, args)
        except:
            P = (z * Psats).sum()
            self.x = z.copy()
            self.P = self.rootsolver(self._P_error, P, P+1, 
                                     1e-2, 5e-8, args)
        self.x /= self.x.sum()
        return self.P, self.x
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"