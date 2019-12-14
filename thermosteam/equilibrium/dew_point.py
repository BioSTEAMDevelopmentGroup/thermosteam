# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:15:30 2019

@author: yoelr
"""

from numpy import asarray, array
from flexsolve import aitken_secant
from .solve_composition import solve_x
from ..utils import fill_like
from ..settings import settings

__all__ = ('DewPoint',)

class DewPoint:
    __slots__ = ('chemicals', 'phi', 'gamma',
                 'pcf', 'Psats', 'Tbs', 'P', 'T', 'x')
    rootsolver = staticmethod(aitken_secant)
    _cached = {}
    def __init__(self, chemicals=(), thermo=None, bp=None):
        thermo = settings.get_thermo(thermo)
        chemicals = tuple(chemicals)
        key = (chemicals, thermo)
        cached = self._cached
        if key in cached:
            other = cached[key]
            fill_like(self, other, self.__slots__)
        else:
            if bp:
                fill_like(self, bp, self.__slots__[:-3])
            else:
                self.gamma = thermo.Gamma(chemicals)
                self.phi = thermo.Phi(chemicals)
                self.pcf = thermo.PCF(chemicals)
                self.Psats = [i.Psat for i in chemicals]
                self.Tbs = array([s.Tb for s in chemicals])
                self.chemicals = chemicals
            self.P = self.T = self.x = None
            cached[key] = self
    
    def _T_error(self, T, P, z, zP):
        x_gamma_pcf = zP/array([i(T) for i in self.Psats]) * self.phi(z, T, P)
        self.x = solve_x(x_gamma_pcf, self.gamma, self.pcf, T, self.x)
        return 1 - self.x.sum()
    
    def _P_error(self, P, T, z, z_over_Psats):
        x_gamma_pcf = z_over_Psats*P*self.phi(z, T, P)
        self.x = solve_x(x_gamma_pcf, self.gamma, self.pcf, T, self.x)
        return 1 - self.x.sum()
    
    def __call__(self, z, *, T=None, P=None):
        if T:
            if P: raise ValueError("may specify either T or P, not both")
            return self.solve_Px(z, T)
        elif P:
            return self.solve_Tx(z, P)
        else:
            raise ValueError("must specify either T or P")
    
    def solve_Tx(self, z, P):
        """Dew point given composition and pressure.

        Parameters
        ----------
        z : array_like
            Molar composition.

        P : float
            Pressure (Pa).

        Returns
        -------
        T : float
            Dew point temperature (K).
        x : numpy.ndarray
            Liquid phase composition.

        Examples
        --------
        >>> from thermotree import Chemicals, DewPoint
        >>> dp = DewPoint(Chemicals('Ethanol', 'Water'))
        >>> dp.solve_Tx(z=(0.5, 0.5), P=101325)
        (357.45184742263075, array([0.151, 0.849]))
        """
        z = asarray(z)
        zP = z * P
        args = (P, z, zP)
        self.P = P
        T = self.T or (z * self.Tbs).sum()
        try:
            self.T = self.rootsolver(self._T_error, T, T-0.01,
                                     1e-6, 5e-8, args)
        except:
            self.x = z.copy()
            T = (z * self.Tbs).sum()
            self.T = self.rootsolver(self._T_error, T, T-0.01,
                                     1e-6, 5e-8, args)
                
        self.x /= self.x.sum()
        return self.T, self.x
    
    def solve_Px(self, z, T):
        """Dew point given composition and temperature.

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
            Liquid phase composition.

        Examples
        --------
        >>> from thermotree import Chemicals, DewPoint
        >>> dp = DewPoint(Chemicals('Ethanol', 'Water'))
        >>> dp.solve_Px(z=(0.703, 0.297), T=352.28)
        (111366.15384513882, array([0.6, 0.4]))
 
       """
        z = asarray(z)
        Psats = array([i(T) for i in self.Psats])
        z_over_Psats = z/Psats
        args = (T, z, z_over_Psats)
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