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
from .fugacity_coefficients import IdealFugacityCoefficients
from .domain import vle_domain
from ..exceptions import InfeasibleRegion
from .. import functional as fn
from .._settings import settings

__all__ = (
    'BubblePoint', 'BubblePointValues',
    # 'BubblePointBeta'
)

# %% Solvers

def y_iter(y, y_phi, phi, T, P):
    y = fn.normalize(y)
    return y_phi / phi(y, T, P)

def solve_y(y_phi, phi, T, P, y_guess):
    if isinstance(phi, IdealFugacityCoefficients): return y_phi
    return flx.wegstein(y_iter, y_phi, 1e-12, args=(y_phi, phi, T, P), 
                        checkiter=False,
                        checkconvergence=False, 
                        convergenceiter=5,
                        maxiter=BubblePoint.maxiter)


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
        
    @property
    def K(self):
        return self.y / self.x
    
    def __repr__(self):
        return f"{type(self).__name__}(T={self.T:.2f}, P={self.P:.0f}, IDs={self.IDs}, z={self.z}, y={self.y})"


class ReactiveBubblePointValues:
    __slots__ = ('T', 'P', 'IDs', 'z0', 'dz', 'y', 'x')
    
    def __init__(self, T, P, IDs, z0, dz, y, x):
        self.T = T
        self.P = P
        self.IDs = IDs
        self.z0 = z0
        self.dz = dz
        self.y = y
        self.x = x
        
    @property
    def K(self):
        return self.y / self.x
        
    def __repr__(self):
        return f"{type(self).__name__}(T={self.T:.2f}, P={self.P:.0f}, IDs={self.IDs}, z0={self.z0}, dz={self.dz}, y={self.y}, x={self.x})"


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
    BubblePointValues(T=355.00, P=109407, IDs=('Water', 'Ethanol'), z=[0.5 0.5], y=[0.344 0.656])
    >>> # Note that the result is a BubblePointValues object which contain all results as attibutes
    >>> (bp.T, round(bp.P), bp.IDs, bp.z, bp.y)
    (355, 109407, ('Water', 'Ethanol'), array([0.5, 0.5]), array([0.344, 0.656]))
    >>> # Solve bubble point at constant pressure
    >>> BP(z=molar_composition, P=101325)
    BubblePointValues(T=353.03, P=101325, IDs=('Water', 'Ethanol'), z=[0.5 0.5], y=[0.343 0.657])
    
    """
    __slots__ = ('chemicals', 'IDs', 'gamma', 'phi', 'pcf',
                 'Psats', 'Tmin', 'Tmax', 'Pmin', 'Pmax')
    _cached = {}
    maxiter = 100
    T_tol = 1e-12
    P_tol = 1e-6
    def __new__(cls, chemicals=None, thermo=None):
        thermo = settings.get_default_thermo(thermo)
        if chemicals is None: 
            chemicals = thermo.chemicals.tuple
        else:
            chemicals = tuple(chemicals)
        key = (chemicals, thermo.Gamma, thermo.Phi, thermo.PCF)
        cached = cls._cached
        if key in cached:
            return cached[key]
        else:
            self = super().__new__(cls)
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
            return self
    
    def _T_error(self, T, P, z_over_P, z_norm, y):
        if T <= 0: raise InfeasibleRegion('negative temperature')
        Psats = np.array([i(T) for i in self.Psats], dtype=float)
        y_phi =  (z_over_P
                  * Psats
                  * self.gamma(z_norm, T, P) 
                  * self.pcf(T, P, Psats))
        y[:] = solve_y(y_phi, self.phi, T, P, y)
        return 1. - y.sum()
    
    def _P_error(self, P, T, z_Psat_gamma, Psats, y):
        if P <= 0: raise InfeasibleRegion('negative pressure')
        y_phi = z_Psat_gamma * self.pcf(T, P, Psats) / P
        y[:] = solve_y(y_phi, self.phi, T, P, y)
        return 1. - y.sum()
        
    def _P_error_dep(self, P, T, z, Psats, z_Psats, y):
        if P <= 0: raise InfeasibleRegion('negative pressure')
        y_phi = z_Psats * self.gamma(z, T, P) * self.pcf(T, P, Psats) / P
        y[:] = solve_y(y_phi, self.phi, T, P, y)
        return 1. - y.sum()
    
    def _T_error_reactive(self, T, P, z, dz, y, x, liquid_conversion):
        if T <= 0: raise InfeasibleRegion('negative temperature')
        dz[:] = liquid_conversion(z, T, P, 'l')
        x[:] = z + dz
        x /= x.sum()
        Psats = np.array([i(T) for i in self.Psats], dtype=float)
        y_phi =  (x / P
                  * Psats
                  * self.gamma(x, T, P) 
                  * self.pcf(T, P, Psats))
        y[:] = solve_y(y_phi, self.phi, T, P, y)
        return 1. - y.sum()
    
    def _P_error_reactive(self, P, T, Psats, z, dz, y, x, liquid_conversion):
        if P <= 0: raise InfeasibleRegion('negative pressure')
        dz[:] = liquid_conversion(z, T, P, 'l')
        x[:] = z + dz
        x /= x.sum()
        z_Psat_gamma = x * Psats * self.gamma(x, T, P)
        y_phi = z_Psat_gamma * self.pcf(T, P, Psats) / P
        y[:] = solve_y(y_phi, self.phi, T, P, y)
        return 1. - y.sum()
    
    def _T_error_ideal(self, T, z_over_P, y):
        y[:] = z_over_P * np.array([i(T) for i in self.Psats], dtype=float)
        return 1 - y.sum()
    
    def _Ty_ideal(self, z_over_P):
        f = self._T_error_ideal
        y = z_over_P.copy()
        args = (z_over_P, y)
        Tmin = self.Tmin + 10
        Tmax = self.Tmax - 10
        fmax = f(Tmin, *args)
        if fmax < 0.: return Tmin, y
        fmin = f(Tmax, *args)
        if fmin > 0.: return Tmax, y
        T = flx.IQ_interpolation(f, Tmin, Tmax, fmax, fmin, 
                                 None, self.T_tol, 5e-12, args, 
                                 checkiter=False,
                                 checkbounds=False, 
                                 maxiter=self.maxiter)
        return T, y
    
    def _Py_ideal(self, z_Psat_gamma_pcf):
        P = z_Psat_gamma_pcf.sum()
        y = z_Psat_gamma_pcf / P
        return P, y
    
    def __call__(self, z, *, T=None, P=None, liquid_conversion=None):
        z = np.asarray(z, float)
        if T:
            if P: raise ValueError("may specify either T or P, not both")
            P, *args = self.solve_Py(z, T, liquid_conversion)
        elif P:
            T, *args = self.solve_Ty(z, P, liquid_conversion)
        else:
            raise ValueError("must specify either T or P")
        if liquid_conversion:
            return ReactiveBubblePointValues(T, P, self.IDs, z, *args)
        else:
            return BubblePointValues(T, P, self.IDs, z, *args)
    
    def solve_Ty(self, z, P, liquid_conversion=None, guess=None):
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
        >>> tmo.docround(BP.solve_Ty(z=np.array([0.6, 0.4]), P=101325))
        (353.8284, array([0.383, 0.617]))
        
        """
        positives = z > 0.
        N = positives.sum()
        if N == 0:
            raise ValueError('no components present')
        if N == 1 and liquid_conversion is None:
            chemical = self.chemicals[fn.first_true_index(positives)]
            T = chemical.Tsat(P, check_validity=False) if P <= chemical.Pc else chemical.Tc
            y = z.copy()
            return T, fn.normalize(y)
        elif liquid_conversion is None:
            f = self._T_error
            z_norm = z / z.sum()
            z_over_P = z/P
            T_guess, y = self._Ty_ideal(z_over_P)
            args = (P, z_over_P, z_norm, y)
            try:
                T = flx.aitken_secant(f, T_guess, T_guess + 1e-3,
                                      self.T_tol, 5e-12, args,
                                      checkiter=False, 
                                      maxiter=self.maxiter)
            except RuntimeError:
                Tmin = self.Tmin; Tmax = self.Tmax
                T = flx.IQ_interpolation(f, Tmin, Tmax,
                                         f(Tmin, *args), f(Tmax, *args),
                                         T_guess, self.T_tol, 5e-12, args, 
                                         checkiter=False, checkbounds=False, 
                                         maxiter=self.maxiter)
            return T, fn.normalize(y)
        else:
            f = self._T_error_reactive
            z_norm = z / z.sum()
            x = z_norm.copy()
            dz = z_norm.copy()
            z_over_P = z / P
            T_guess, y = self._Ty_ideal(z_over_P)
            args = (P, z_norm, dz, y, x, liquid_conversion)
            try:
                T = flx.aitken_secant(f, T_guess, T_guess + 1e-3,
                                      self.T_tol, 5e-12, args,
                                      checkiter=False, maxiter=self.maxiter)
            except RuntimeError:
                Tmin = self.Tmin; Tmax = self.Tmax
                T = flx.IQ_interpolation(f, Tmin, Tmax,
                                         f(Tmin, *args), f(Tmax, *args),
                                         T_guess, self.T_tol, 5e-12, args, 
                                         checkiter=False, checkbounds=False,
                                         maxiter=self.maxiter)
            return T, dz, fn.normalize(y), x
    
    def solve_Py(self, z, T, liquid_conversion=None):
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
        (91592.781, array([0.42, 0.58]))
        
        """
        positives = z > 0.
        N = positives.sum()
        if N == 0:
            raise ValueError('no components present')
        if N == 1 and liquid_conversion is None:
            chemical = self.chemicals[fn.first_true_index(positives)]
            P = chemical.Psat(T) if T <= chemical.Tc else chemical.Pc
            y = z.copy()
            return P, fn.normalize(y)
        elif liquid_conversion is None:
            if T > self.Tmax: T = self.Tmax
            elif T < self.Tmin: T = self.Tmin
            Psats = np.array([i(T) for i in self.Psats])
            z_norm = z / z.sum()
            if self.gamma.P_dependent:
                f = self._P_error_dep
                z_Psats = z_norm * Psats
                P_guess, y = self._Py_ideal(z_Psats)
                args = (T, z_norm, Psats, z_Psats, y)
            else:
                z_Psat_gamma = z_norm * Psats * self.gamma(z_norm, T, 101325)
                P_guess, y = self._Py_ideal(z_Psat_gamma)
                f = self._P_error
                args = (T, z_Psat_gamma, Psats, y)
            try:
                P = flx.aitken_secant(f, P_guess, P_guess-1, self.P_tol, 1e-9,
                                      args, checkiter=False, maxiter=self.maxiter)
            except RuntimeError:
                Pmin = self.Pmin; Pmax = self.Pmax
                P = flx.IQ_interpolation(f, Pmin, Pmax,
                                         f(Pmin, *args), f(Pmax, *args),
                                         P_guess, self.P_tol, 5e-12, args,
                                         checkiter=False, checkbounds=False, 
                                         maxiter=self.maxiter)
            return P, fn.normalize(y)
        else:
            f = self._P_error_reactive
            z_norm = z / z.sum()
            Psats = np.array([i(T) for i in self.Psats])
            x = z_norm.copy()
            dz = z_norm.copy()
            z_Psat_gamma = z * Psats * self.gamma(z_norm, T, 101325)
            P_guess, y = self._Py_ideal(z_Psat_gamma)
            args = (T, Psats, z_norm, dz, y, x, liquid_conversion)
            try:
                P = flx.aitken_secant(f, P_guess, P_guess-1, self.P_tol, 1e-9,
                                      args, checkiter=False, 
                                      maxiter=self.maxiter)
            except RuntimeError:
                Pmin = self.Pmin; Pmax = self.Pmax
                P = flx.IQ_interpolation(f, Pmin, Pmax,
                                         f(Pmin, *args), f(Pmax, *args),
                                         P_guess, self.P_tol, 5e-12, args,
                                         checkiter=False, checkbounds=False,
                                         maxiter=self.maxiter)
            return P, dz, fn.normalize(y), x
    
    def __repr__(self):
        chemicals = ", ".join([i.ID for i in self.chemicals])
        return f"{type(self).__name__}([{chemicals}])"


# class BubblePointBeta:
#     """
#     Create a BubblePointBeta object that returns bubble point values when
#     called with a composition and either a temperture (T) or pressure (P).
    
#     Parameters
#     ----------
#     flasher=None : :class:`~thermo.Flash`, optional
    
#     Examples
#     --------
#     >>> import thermosteam as tmo
#     >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
#     >>> tmo.settings.set_thermo(chemicals)
#     >>> BP = tmo.equilibrium.BubblePointBeta(chemicals)
#     >>> molar_composition = (0.5, 0.5)
#     >>> # Solve bubble point at constant temperature
#     >>> bp = BP(z=molar_composition, T=355)
#     >>> # bp
#     >>> # BubblePointValues(T=355.00, P=111447, IDs=['Water', 'Ethanol'], z=[0.5 0.5], y=[0.341 0.659])
#     >>> # Note that the result is a BubblePointValues object which contain all results as attibutes
#     >>> # (bp.T, round(bp.P), bp.IDs, bp.z, bp.y)
#     >>> # (355, 111447, ['Water', 'Ethanol'], array([0.5, 0.5]), array([0.341, 0.659]))
#     >>> # Solve bubble point at constant pressure
#     >>> # BP(z=molar_composition, P=101325)
#     >>> # BubblePointValues(T=352.59, P=101325, IDs=['Water', 'Ethanol'], z=[0.5 0.5], y=[0.34 0.66])
    
#     """
#     __slots__ = ('chemicals', 'IDs', 'flasher')
#     _cached = {}
#     def __init__(self, chemicals=(), flasher=None):
#         self.chemicals = chemicals
#         self.IDs = [i.ID for i in chemicals]
#         self.flasher = flasher or settings.flasher()
    
#     __call__ = BubblePoint.__call__
    
#     def solve_Ty(self, z, P, liquid_conversion=None):
#         """
#         Bubble point at given composition and pressure.

#         Parameters
#         ----------
#         z : ndarray
#             Molar composition.
#         P : float
#             Pressure [Pa].
        
#         Returns
#         -------
#         T : float 
#             Bubble point temperature [K].
#         y : ndarray
#             Vapor phase molar composition.

#         Examples
#         --------
#         >>> import thermosteam as tmo
#         >>> import numpy as np
#         >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
#         >>> tmo.settings.set_thermo(chemicals)
#         >>> BP = tmo.equilibrium.BubblePointBeta(chemicals)
#         >>> # tmo.docround(BP.solve_Ty(z=np.array([0.6, 0.4]), P=101325))
#         >>> # (353.4052, array([0.38, 0.62]))
        
#         """
#         positives = z > 0.
#         N = positives.sum()
#         if N == 0:
#             raise ValueError('no components present')
#         if N == 1:
#             T = self.chemicals.tuple[fn.first_true_index(positives)].Tsat(P, check_validity=False)
#             y = z.copy()
#         else:
#             results = self.flasher.flash(P=P, VF=0., zs=z.tolist())
#             y = np.array(results.gas.zs)
#             T = results.T
#         return T, fn.normalize(y)
    
#     def solve_Py(self, z, T, liquid_conversion=None):
#         """
#         Bubble point at given composition and temperature.

#         Parameters
#         ----------
#         z : ndarray
#             Molar composition.
#         T : float
#             Temperature [K].
        
#         Returns
#         -------
#         P : float
#             Bubble point pressure [Pa].
#         y : ndarray
#             Vapor phase molar composition.

#         Examples
#         --------
#         >>> import thermosteam as tmo
#         >>> import numpy as np
#         >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
#         >>> tmo.settings.set_thermo(chemicals)
#         >>> BP = tmo.equilibrium.BubblePoint(chemicals)
#         >>> # tmo.docround(BP.solve_Py(z=np.array([0.703, 0.297]), T=352.28))
#         >>> # (92966.9114, array([0.418, 0.582]))
        
#         """
#         positives = z > 0.
#         N = positives.sum()
#         if N == 0:
#             raise ValueError('no components present')
#         if N == 1:
#             T = self.chemicals.tuple[fn.first_true_index(positives)].Psat(T)
#             y = z.copy()
#         else:
#             results = self.flasher.flash(T=T, VF=0., zs=z.tolist())
#             y = np.array(results.gas.zs)
#             P = results.P
#         return P, fn.normalize(y)
    
#     def __repr__(self):
#         chemicals = ", ".join([i.ID for i in self.chemicals])
#         return f"{type(self).__name__}([{chemicals}])"
    
