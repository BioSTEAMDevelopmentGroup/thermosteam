# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:30:33 2019

@author: yoelr
"""
from numpy import asarray, array
from flexsolve import aitken_secant
from .solve_composition import solve_y
from ..utils import fill_like
from ..settings import settings

__all__ = ('BubblePoint',)

class BubblePoint:
    __slots__ = ('chemicals', 'gamma', 'phi', 'pcf',
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
            self.gamma = thermo.Gamma(chemicals)
            self.phi = thermo.Phi(chemicals)
            self.pcf = thermo.PCF(chemicals)
            self.Psats = [i.Psat for i in chemicals]
            self.Tbs = array([s.Tb for s in chemicals])
            self.chemicals = chemicals
            self.P = self.T = self.y = None
            cached[key] = self
    
    def _T_error(self, T, z, P, z_over_P):
        y_phi =  (z_over_P
                  * array([i(T) for i in self.Psats])
                  * self.gamma(z, T) 
                  * self.pcf(z, T))
        self.y = solve_y(y_phi, self.phi, T, P, self.y)
        return 1. - self.y.sum()
    
    def _P_error(self, P, T, z, Psat_gamma_pcf):
        y_phi = z * Psat_gamma_pcf / P
        self.y = solve_y(y_phi, self.phi, T, P, self.y)
        return 1. - self.y.sum()
    
    def solve_Ty(self, z, P):
        """Bubble point at given composition and pressure

        Parameters
        ----------
        z : array_like
            Molar composotion.
        P : float
            Pressure (Pa).
        
        Returns
        -------
        T : float 
            Bubble point temperature (K)
        y : numpy.ndarray
            Composition of the vapor phase.

        Examples
        --------
        >>> from thermotree import Chemicals, BubblePoint
        >>> bp = BubblePoint(*Chemicals('Ethanol', 'Water'))
        >>> bp.solve_Ty(z=(0.6, 0.4), P=101325)
        (352.2820850833474, array([0.703, 0.297]))
        
        """
        
        z = asarray(z)
        self.P = P
        args = (z, P, z/P)
        T = self.T or (z * self.Tbs).sum()
        try:
            self.T = self.rootsolver(self._T_error, T, T+0.01,
                                     1e-6, 5e-8, args)
        except:
            self.x = z.copy()
            T = (z * self.Tbs).sum()
            self.T = self.rootsolver(self._T_error, T, T+0.01,
                                     1e-6, 5e-8, args)
        self.y /= self.y.sum()
        return self.T, self.y
    
    def solve_Py(self, z, T):
        """Bubble point at given composition and temperature.

        Parameters
        ----------
        z : array_like
            Molar composotion.
        T : float
            Temperature (K).
        
        Returns
        -------
        P : float
            Bubble point pressure (Pa).
        y : numpy.ndarray
            Vapor phase composition.

        Examples
        --------
        >>> from thermotree import Chemicals, BubblePoint
        >>> bp = BubblePoint(*Chemicals('Ethanol', 'Water'))
        >>> bp.solve_Py(z=(0.703, 0.297), T=352.28)
        (103494.17209657285, array([0.757, 0.243]))
        
        """
        z = asarray(z)
        Psat = array([i(T) for i in self.Psats])
        Psat_gamma_pcf =  Psat * self.gamma(z, T) * self.pcf(z, T)
        self.T = T
        args = (T, z, Psat_gamma_pcf)
        P = self.P or (z * Psat).sum()
        try:
            self.P = self.rootsolver(self._P_error, P, P-1,
                                     1e-2, 5e-8, args)
        except:
            P = (z * Psat).sum()
            self.P = self.rootsolver(self._P_error, P, P-1,
                                     1e-2, 5e-8, args)
        self.y /= self.y.sum()
        return self.P, self.y
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"
    

    