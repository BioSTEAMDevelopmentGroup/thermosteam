# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:15:30 2019

@author: yoelr
"""

from numpy import asarray, array
from flexsolve import aitken_secant
from .activity_coefficients import DortmundActivityCoefficients
from .fugacity_coefficients import IdealFugacityCoefficients

__all__ = ('DewPoint',)

class DewPoint:
    __slots__ = ('gamma', 'P', 'T', 'x', 'Psats', 'phi_V', 'Tbs')
    rootsolver = staticmethod(aitken_secant)
    def __init__(self, chemicals, gamma=None, phi_V=None):
        self.gamma = gamma or DortmundActivityCoefficients(chemicals)
        self.phi_V = phi_V or IdealFugacityCoefficients(chemicals)
        self.Psats = [i.Psat for i in chemicals]
        self.Tbs = array([s.Tb for s in chemicals])
        self.P = self.T = self.x = None
    
    def _T_error(self, T, P, z, zP):
        x_gamma = zP/array([i(T) for i in self.Psats]) * self.phi_V(z, T, P)
        self.x = self.gamma.solve_x(x_gamma, T, self.x)
        return 1 - self.x.sum()
    
    def _P_error(self, P, T, z, z_over_Psats):
        x_gamma = z_over_Psats*P*self.phi_V(z, T, P)
        self.x = self.gamma.solve_x(x_gamma, T, self.x)
        return 1 - self.x.sum()
    
    def solve_Tx(self, z, P):
        """Dew point given composition and pressure.

        Parameters
        ----------
        y : array_like
            Vapor phase composition.

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
        >>> from biosteam import Species, DewPoint, Dortmund
        >>> gamma = Dortmund(*Species('Ethanol', 'Water'))
        >>> dp = DewPoint(gamma)
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
        y : array_like
            Vapor phase composition.
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
        >>> from biosteam import Species, DewPoint, Dortmund
        >>> gamma = Dortmund(*Species('Ethanol', 'Water'))
        >>> dp = DewPoint(gamma)
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
        chemicals = ", ".join([i.ID for i in self.gamma.chemicals])
        return f"<{type(self).__name__}([{chemicals}])>"