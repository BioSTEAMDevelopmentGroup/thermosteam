# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2025, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import flexsolve as flx
from ..exceptions import NoEquilibrium
from .bubble_point import BubblePoint
from .dew_point import DewPoint
from .vle import VLE
from .lle import LLE
from .equilibrium import Equilibrium
from ..utils import Cache
import thermosteam as tmo
import numpy as np
from scipy.optimize import root
from numba import njit
from typing import NamedTuple
from chemicals import Rachford_Rice_solution2

__all__ = ('VLLE', 'VLLECache')

@njit(cache=True)
def RashfordRice_VLLE_residuals(VL1, z, Ka, Kb):
    V, L1 = VL1
    L2 = 1.0 - V - L1
    denom = V + L1 / Ka + L2 / Kb
    y = z / denom
    x_alpha = y / Ka
    residuals = np.zeros(2)
    residuals[0] = y.sum() - 1.0
    residuals[1] = x_alpha.sum() - 1.0
    return residuals

class BubblePointValuesVLLE(NamedTuple):
    K_L: np.ndarray
    phi_L: float
    K_bubble: np.ndarray
    T_bubble: float
    P_bubble: float

class DewPointValuesVLLE(NamedTuple):
    K_dew: np.ndarray
    T_dew: float
    P_dew: float

class VLLEInterpolator:
    __slots__ = (
        'K_L', 'phi_L', 'K_bubble', 'K_dew',
        'bubble_value', 'dew_value', 'variable',
    )
    
    def __init__(self, 
                 K_L, phi_L, 
                 K_bubble, K_dew, 
                 P_bubble, P_dew,
                 T_bubble, T_dew):
        self.K_L = K_L
        self.phi_L = phi_L
        self.K_bubble = K_bubble
        self.K_dew = K_dew
        if P_bubble == P_dew:
            self.variable = 'T'
            self.bubble_value = T_bubble
            self.dew_value = T_dew
        elif T_bubble == T_dew:
            self.variable = 'P' 
            self.bubble_value = P_bubble
            self.dew_value = P_dew
        else:
            raise ValueError(
                'either temperature or pressure must be the same '
                'in bubble and dew points'
            )
        
    def __call__(self, **kwargs):
        value = kwargs[self.variable]
        bubble = self.bubble_value
        dew = self.dew_value
        f = (value - bubble) / (dew - bubble)
        Ka = f * self.K_dew + (1 - f) * self.K_bubble
        n = Ka.size
        V_La_Lb_Ka_KL = np.zeros(n * 2 + 3)
        V_La_Lb_Ka_KL[0] = f
        L = (1 - f)
        V_La_Lb_Ka_KL[1] = La = self.phi_L * L
        V_La_Lb_Ka_KL[2] = L - La
        V_La_Lb_Ka_KL[3:n+3] = Ka
        V_La_Lb_Ka_KL[n+3:] = self.K_L
        return V_La_Lb_Ka_KL

# def f_logKV(y, x, T, P, pcf_Psat_over_P, f_gamma, gamma_args, f_phi):
#     return np.log(pcf_Psat_over_P * f_gamma(x, T, *gamma_args) / f_phi(y, T, P))

# def f_logKL_gammaxa(xb, gamma_xa, T, f_gamma, gamma_args):
#     gamma_xa[gamma_xa < 0] = 1e-64
#     gamma_xb = f_gamma(xb, T, *gamma_args)
#     K = gamma_xb / gamma_xa 
#     xa = K * xb
#     xa /= xa.sum()
#     gamma_xa = f_gamma(xa, T, *gamma_args)
#     K = gamma_xb / gamma_xa
#     n = xb.size
#     logKL_gammaxa = np.zeros(2 * n)
#     logKL_gammaxa[:n] = np.log(K)
#     logKL_gammaxa[n:] = gamma_xa
#     return logKL_gammaxa

# def inner_loop_TP_iter(logKL_gammaxa, La, V, Lb, T, P, z, Ka, KL_args):
#     n = z.size
#     KL = np.exp(logKL_gammaxa[:n])
#     gammaxa = logKL_gammaxa[n:]
#     Kb = Ka * KL
#     denom = V + La / Ka + Lb / Kb
#     y = z / denom
#     y /= y.sum()
#     xa  = y / Ka
#     xa /= xa.sum()
#     xb  = y / Kb
#     xb /= xb.sum()
#     print('Inner loop')
#     print('----------')
#     print('y', y)
#     print('xa', xa)
#     print('xb', xb)
#     print()
#     return f_logKL_gammaxa(xb, gammaxa, T, *KL_args)

# def print_guess_middle(V_La_Lb_logKL_gammaxa):
#     n = (V_La_Lb_logKL_gammaxa.size - 3) // 2
#     V, La, Lb = V_La_Lb_logKL_gammaxa[:3]
#     print('Phase fractions')
#     print('---------------')
#     print('V', round(V, 3))
#     print('La', round(La, 3))
#     print('Lb', round(Lb, 3))
#     KL = np.exp(V_La_Lb_logKL_gammaxa[3:n+3])
#     print()
#     print('Partition coefficients')
#     print('----------------------')
#     print('KL', KL)
#     print()

# def print_guess_outer(V_La_Lb_logKa_logKL_gammaxa):
#     n = (V_La_Lb_logKa_logKL_gammaxa.size - 3) // 3
#     V, La, Lb = V_La_Lb_logKa_logKL_gammaxa[:3]
#     print('Phase fractions')
#     print('---------------')
#     print('V', round(V, 3))
#     print('La', round(La, 3))
#     print('Lb', round(Lb, 3))
#     Ka_KL = np.exp(V_La_Lb_logKa_logKL_gammaxa[3:n*2+3])
#     print()
#     print('Partition coefficients')
#     print('----------------------')
#     print('Ka', Ka_KL[:n])
#     print('KL', Ka_KL[n:])
#     print()
    
# def middle_loop_TP_iter(V_La_Lb_logKL_gammaxa, T, P, z, Ka, KL_args):
#     n = z.size
#     logKL_gammaxa = V_La_Lb_logKL_gammaxa[3:]
#     V_La_Lb = V_La_Lb_logKL_gammaxa[:3]
#     logKL_gammaxa[:] = flx.fixed_point(
#         inner_loop_TP_iter, logKL_gammaxa, 
#         args=(*V_La_Lb, T, P, z, Ka, KL_args),
#         checkiter=False, convergenceiter=5, checkconvergence=False,
#         xtol=1e-9, rtol=1e-12,
#     )
#     logKL = logKL_gammaxa[:n]
#     KL = np.exp(logKL)
#     La, Lb, *_ = Rachford_Rice_solution2(z, Ka, 1 / KL)
#     V = 1 - La - Lb
#     V_La_Lb[0] = V
#     V_La_Lb[1] = La
#     V_La_Lb[2] = Lb
#     V_La_Lb[V_La_Lb < 1e-64] = 1e-64
#     V_La_Lb /= V_La_Lb.sum()
#     print_guess_middle(V_La_Lb_logKL_gammaxa)
#     breakpoint()
#     return V_La_Lb_logKL_gammaxa

# def outer_loop_TP_iter(V_La_Lb_logKa_logKL_gammaxa, T, P, z, KV_args, KL_args):
#     V_La_Lb_logKa_logKL_gammaxa = V_La_Lb_logKa_logKL_gammaxa.copy()
#     V, La, Lb = V_La_Lb_logKa_logKL_gammaxa[:3]
#     n = z.size
#     logKa_logKL = V_La_Lb_logKa_logKL_gammaxa[3:2*n+3]
#     logKa = logKa_logKL[:n]
#     logKL = logKa_logKL[n:]
#     logKb = logKa + logKL
#     Ka = np.exp(logKa)
#     Kb = np.exp(logKb)
#     denom = V + La / Ka + Lb / Kb
#     y = z / denom
#     y /= y.sum()
#     # xa  = y / Ka
#     # xa /= xa.sum()
#     xb  = y / Kb
#     xb /= xb.sum()
#     # logKa1 = f_logKV(y, xa, T, P, *KV_args)
#     logKb = f_logKV(y, xb, T, P, *KV_args)
#     logKa[:] = logKL - logKb
#     V_La_Lb_logKL_gammaxa = np.zeros(3 + 2 * n)
#     V_La_Lb_logKL_gammaxa[:3] = V_La_Lb_logKa_logKL_gammaxa[:3]
#     V_La_Lb_logKL_gammaxa[3:] = V_La_Lb_logKa_logKL_gammaxa[3 + n:]
#     V_La_Lb_logKL_gammaxa = flx.fixed_point(
#         middle_loop_TP_iter, V_La_Lb_logKL_gammaxa, 
#         args=(T, P, z, Ka, KL_args),
#         checkiter=False, convergenceiter=5, checkconvergence=False,
#         xtol=1e-9, rtol=1e-12,
#     ) 
#     V_La_Lb_logKa_logKL_gammaxa[:3] = V_La_Lb_logKL_gammaxa[:3]
#     V_La_Lb_logKa_logKL_gammaxa[3 + n:] = V_La_Lb_logKL_gammaxa[3:]
#     return V_La_Lb_logKa_logKL_gammaxa

class VLLE(Equilibrium, phases='Llg'):
    """
    Create a VLLE object that performs vapor-liquid-liquid equilibrium when called.
        
    Parameters
    ----------
    imol=None : :class:`~thermosteam.indexer.MaterialIndexer`, optional
        Molar chemical phase data is stored here.
    thermal_condition=None : :class:`~thermosteam.ThermalCondition`, optional
        Temperature and pressure results are stored here.
    thermo=None : :class:`~thermosteam.Thermo`, optional
        Themodynamic property package for equilibrium calculations.
        Defaults to `thermosteam.settings.get_thermo()`.
    
    """
    __slots__ = (
        'top_chemical',
        'vle_l',
        'vle_L',
        'lle',
        'iter',
        '_bubble_point',
        '_dew_point',
        '_bubble_point_values',
        '_dew_point_values',
        '_data',
        '_indices',
        '_total',
        '_z',
        '_yx_bubble',
        '_x_dew',
        '_vlle_interpolator',
    )
    
    def __init__(self, 
            imol=None,
            thermal_condition=None,
            thermo=None,
        ):
        super().__init__(imol, thermal_condition, thermo)
        imol = self._imol
        thermo = self._thermo
        thermal_condition = self._thermal_condition
        imol_vle_l = tmo.indexer.MaterialIndexer.from_data(
            imol['gl'], phases=('g', 'l')
        )
        imol_vle_L = tmo.indexer.MaterialIndexer.from_data(
            imol['gL'], phases=('g', 'l')
        )
        imol_lle = tmo.indexer.MaterialIndexer.from_data(
            imol['Ll'], phases=('L', 'l')
        )
        self.vle_l = VLE(imol_vle_l, thermal_condition, thermo)
        self.vle_L = VLE(imol_vle_L, thermal_condition, thermo)
        self.lle = LLE(imol_lle, thermal_condition, thermo)
    
    def __call__(self, *, T=None, P=None, V=None, H=None, S=None, Q=None, B=None, top_chemical=None):
        """
        Perform vapor-liquid-liquid equilibrium.

        Parameters
        ----------
        T : float, optional
            Operating temperature [K].
        P : float, optional
            Operating pressure [Pa].
        V : float, optional
            Molar vapor fraction [by mol].
        H : float, optional
            Enthalpy [kJ/hr].
        S : float, optional
            Entropy [kJ/hr/K]
        Q : float, optional
            Duty [kJ/hr].
        B : float, optional
            Boil up ratio [by mol]
        
        Notes
        -----
        You may only specify two of the following parameters: P, H, T, V, S.
        
        """
        if Q is not None:
            H = Q + self.mixture.xH(self.imol, *self.thermal_condition)
        if B is not None:
            if B == np.inf:
                V = 1.
            else:
                V = B / (1 + B)
        
        ### Decide what kind of equilibrium to run ###
        T_spec = T is not None
        P_spec = P is not None
        V_spec = V is not None
        H_spec = H is not None
        S_spec = S is not None
        N_specs = (T_spec + P_spec + V_spec + H_spec + S_spec)
        assert N_specs == 2, ("must pass two and only two of the following "
                              "specifications: T, P, V, H, S")
        self._setup(top_chemical)
        # Run equilibrium
        if T_spec:
            if P_spec:
                try:
                    self._set_TP(T, P)
                except NoEquilibrium:
                    thermal_condition = self._thermal_condition
                    thermal_condition.T = T
                    thermal_condition.P = P
            elif V_spec:
                try:
                    self._set_TV(T, V)
                except NoEquilibrium:
                    thermal_condition = self._thermal_condition
                    thermal_condition.T = T
            elif H_spec:
                self._set_TH(T, H)
            elif S_spec:
                self._set_TS(T, S)
        elif P_spec:
            if V_spec:
                try:
                    self._set_PV(P, V)
                except NoEquilibrium:
                    thermal_condition = self._thermal_condition
                    thermal_condition.P = P
            elif H_spec:
                try:
                    self._set_PH(P, H)
                except NoEquilibrium:
                    thermal_condition = self._thermal_condition
                    thermal_condition.P = P
            elif S_spec:
                try:
                    self._set_PS(P, S)
                except:
                    try:
                        self._set_PS(P, S)
                    except NoEquilibrium:
                        thermal_condition = self._thermal_condition
                        thermal_condition.P = P
        elif S_spec: # pragma: no cover
            if H_spec:
                raise NotImplementedError('specification H and S is invalid')
            else: # V_spec
                raise NotImplementedError('specification V and S not implemented')
        elif H_spec: # pragma: no cover
            raise NotImplementedError('specification V and H not implemented')
    
    @property
    def vapor_fraction(self):
        imol = self.imol
        g = imol['g'].sum()
        return g / (g + imol['lL'].sum())
    
    def _setup(self, top_chemical):
        chemicals = self.chemicals
        imol = self.imol
        self._data = data = imol['Lgl']
        self._indices = indices = chemicals.get_vlle_indices(data.nonzero_keys())
        subdata = data[:, indices].sum(axis=0)
        self._total = total = subdata.sum()
        self.top_chemical = self.chemicals.IDs[subdata.argmax()] if top_chemical is None else top_chemical
        self._z = subdata / total
        self._vlle_interpolator = None
        chemicals = chemicals.tuple
        chemicals = tuple([chemicals[i] for i in indices])
        thermo = self.thermo
        self._bubble_point = BubblePoint(chemicals, thermo)
        self._dew_point = DewPoint(chemicals, thermo)
        self._bubble_point_values = None
        self._dew_point_values = None
    
    def _hot_start(self, T, P):
        if self._vlle_interpolator is None:
            K_L, phi_L, K_bubble, T_bubble, P_bubble = self._bubble_point_values
            K_dew, T_dew, P_dew = self._dew_point_values
            self._vlle_interpolator = vlle_interpolator = VLLEInterpolator(
                K_L, phi_L, K_bubble, K_dew, 
                P_bubble, P_dew, T_bubble, T_dew
            )
            return vlle_interpolator(T=T, P=P)
        else:
            return self._vlle_interpolator(T=T, P=P)
    
    def _save_bubble_point(self):
        T_bubble, P_bubble = self.thermal_condition
        lle = self.lle
        phi_L = lle._phi
        K_L = lle._K
        y, x = self._yx_bubble 
        K_bubble = y / x
        self._bubble_point_values = BubblePointValuesVLLE(K_L, phi_L, K_bubble, T_bubble, P_bubble)
    
    def _save_dew_point(self):
        T_dew, P_dew = self.thermal_condition
        K_dew = self._z / self._x_dew
        self._dew_point_values = DewPointValuesVLLE(K_dew, T_dew, P_dew)
    
    def _iter_flows_at_TP(self, data, T, P, indices, z):
        imol = self.imol
        L_mol, g_mol, l_mol = imol['Lgl']
        data *= z / data.sum(axis=0)
        L_mol[indices] = data[0]
        g_mol[indices] = data[1]
        l_mol[indices] = data[2]
        self.iter += 1
            
        # LLE
        self.lle(T=T, P=P, top_chemical=self.top_chemical, use_cache=True)
        
        # VLE with extract
        try:
            self.vle_L._setup()
            self.vle_L._solve_TP(T=T, P=P)
        except:
            self.vle_L(T=T, P=P)
        y = g_mol[indices]
        V0 = y.sum()
        xa = L_mol[indices]
        La = xa.sum()
        if La and V0: 
            y[y < 1e-64] = 1e-64
            xa[xa < 1e-64] = 1e-64
            y /= V0
            xa /= La
            Ka = y / xa
        else:
            Ka = None
        
        # VLE with raffinate
        try:
            self.vle_l._setup()
            self.vle_l._solve_TP(T=T, P=P)
        except:
            self.vle_l(T=T, P=P)
        y = g_mol[indices]
        V = y.sum() 
        xb = l_mol[indices]
        Lb = xb.sum()
        if V and Lb:
            y[y < 1e-64] = 1e-64
            xb[xb < 1e-64] = 1e-64
            y /= V
            xb /= Lb
            Kb = y / xb
        else:
            Kb = None
        
        if Ka is None or Kb is None:
            # At most 2 phases
            values = np.array([
                L_mol[indices],
                g_mol[indices],
                l_mol[indices],
            ])
        else:
            # Potentially 3 phases
            guess = np.array([V, La, Lb])
            guess /= guess.sum()
            sol = root(
                RashfordRice_VLLE_residuals, 
                guess[:2], 
                args=(z, Ka, Kb), 
                method="hybr", 
                tol=1e-12,
            )
            V, La = sol.x
            Lb = 1 - V - La
            if V < 1e-16 or La < 1e-16 or Lb < 1e-16:
                # Actually 2 phases
                values = np.array([
                    L_mol[indices],
                    g_mol[indices],
                    l_mol[indices],
                ])
            else:
                denom = V + La / Ka + Lb / Kb
                y = z / denom
                y /= y.sum()
                xa = y / Ka
                xa /= xa.sum()
                xb  = y / Kb
                xb /= xb.sum()
                values = np.array([
                    xa * La, y * V, xb * Lb
                ])
        values[values < 1e-24] = 0
        return values
    
    def _set_TP(self, T, P):
        self.iter = 0
        indices = self._indices
        thermal_condition = self._thermal_condition
        thermal_condition.T = T
        thermal_condition.P = P
        L_mol, g_mol, l_mol = self._data
        
        # Check dew point
        if self._dew_point_values is None: self.dew_point_at_P(P)
        T_dew = self._dew_point_values.T_dew
        P_dew = self._dew_point_values.P_dew
        if T >= T_dew - 1e-64 and P_dew <= P_dew + 1e64: 
            thermal_condition.T = T
            thermal_condition.P = P
            g_mol[:] += L_mol + l_mol
            L_mol.clear()
            l_mol.clear()
            return
        
        # Check bubble point
        if self._bubble_point_values is None: self.bubble_point_at_P(P)
        T_bubble = self._bubble_point_values.T_bubble
        P_bubble = self._bubble_point_values.P_bubble
        if T <= T_bubble + 1e-64 and P_bubble >= P_bubble - 1e64: 
            l_mol += g_mol
            g_mol[:] = 0
            self.lle(T=T, P=P)
            return
        else:
            thermal_condition.T = T
            thermal_condition.P = P
        
        z = self._z
        if (L_mol[indices].any() and g_mol[indices].any() and l_mol[indices].any()):
            subdata = np.array([
                L_mol[indices], g_mol[indices], l_mol[indices],
            ])
            subdata *= z / subdata.sum(axis=0)
        else:
            V_La_Lb_Ka_KL = self._hot_start(T, P)
            n = z.size
            V, La, Lb = V_La_Lb_Ka_KL[:3]
            Ka_KL = V_La_Lb_Ka_KL[3:]
            Ka = Ka_KL[:n]
            Kb = Ka * Ka_KL[n:]
            denom = V + La / Ka + Lb / Kb
            y = z / denom
            y /= y.sum()
            xa = y / Ka
            xa /= xa.sum()
            xb  = y / Kb
            xb /= xb.sum()
            subdata = np.array([
                xa * La, y * V, xb * Lb,
            ])
        new_subdata = flx.fixed_point(
            self._iter_flows_at_TP, subdata, xtol=1e-6, 
            args=(T, P, indices, z),
            convergenceiter=10, 
            checkconvergence=False,
            checkiter=False,
            maxiter=100,
        )
        self._data[:, indices] = self._total * new_subdata * z / new_subdata.sum(axis=0)
    
    def _T_bubble(self):
        thermal_condition = self._thermal_condition
        P = thermal_condition.P
        Ta, ya = self._bubble_point.solve_Ty(self._z, P, lle=True)
        self.lle(
            T=Ta, P=P,
            top_chemical=self.top_chemical, 
        )
        z = self._data[0, self._indices]
        La = z.sum()
        xa = z / La
        self._yx_bubble = (ya, xa)
        return Ta
     
    def _P_bubble(self):
        thermal_condition = self._thermal_condition
        T = thermal_condition.T
        Pa, ya = self._bubble_point.solve_Py(self._z, T, lle=True)
        self.lle(
            T=T, P=Pa,
            top_chemical=self.top_chemical,
        )
        z = self._data[0, self._indices]
        La = z.sum()
        xa = z / La
        self._yx_bubble = (ya, xa)
        return Pa
       
    def bubble_point_at_T(self, T=None):
        thermal_condition = self._thermal_condition
        if T is None: 
            T = thermal_condition.T
        else:
            thermal_condition.T = T
        imol = self.imol
        l_mol = imol['l']
        g_mol = imol['g']
        l_mol += g_mol
        g_mol[:] = 0
        self._P_bubble()
        self._save_bubble_point()
        return thermal_condition.P
    
    def bubble_point_at_P(self, P=None):
        thermal_condition = self._thermal_condition
        if P is None: 
            P = thermal_condition.P
        else:
            thermal_condition.P = P
        imol = self.imol
        l_mol = imol['l']
        g_mol = imol['g']
        l_mol += g_mol
        g_mol[:] = 0
        self._T_bubble()
        self._save_bubble_point()
        return thermal_condition.T
    
    def dew_point_at_T(self, T=None):
        thermal_condition = self.thermal_condition
        if T is None: 
            T = thermal_condition.T
        else:
            thermal_condition.T = T
        dew_point = self._dew_point
        P, x = dew_point.solve_Px(self._z, T)
        data_copy = self._data.copy()
        L_mol, g_mol, l_mol = self._data
        L_mol[self._indices] = x
        l_mol.clear()
        g_mol.clear()
        self.lle(
            T=T, P=thermal_condition.P,
            top_chemical=self.top_chemical,
        )
        if l_mol.any():
            xa = L_mol[self._indices]
            xa = xa / xa.sum()
            Pa, xa = dew_point.solve_Px(self._z, T, guess=(P, xa, P+10))
            xb = l_mol[self._indices]
            xb = xb / xb.sum()
            Pb, xb = dew_point.solve_Px(self._z, T, guess=(Pa, xb, Pa+10))
            if Pa < Pb:
                P = Pa
                x = xa
            else:
                P = Pb
                x = xb
        self._x_dew = x
        thermal_condition.P = P
        self._save_dew_point()
        self._data[:] = data_copy
        g_mol[:] += L_mol + l_mol
        L_mol.clear()
        l_mol.clear()
        return P
    
    def dew_point_at_P(self, P=None):
        thermal_condition = self.thermal_condition
        if P is None: 
            P = thermal_condition.P
        else:
            thermal_condition.P = P
        dew_point = self._dew_point
        T, x = dew_point.solve_Tx(self._z, P)
        data_copy = self._data.copy()
        L_mol, g_mol, l_mol = self._data
        L_mol[self._indices] = x
        l_mol.clear()
        g_mol.clear()
        self.lle(
            P=P, T=thermal_condition.T,
            top_chemical=self.top_chemical,
        )
        if l_mol.any():
            xa = L_mol[self._indices]
            xa = xa / xa.sum()
            Ta, xa = dew_point.solve_Tx(self._z, P, guess=(T, xa, T+10))
            xb = l_mol[self._indices]
            xb = xb / xb.sum()
            Tb, xb = dew_point.solve_Tx(self._z, P, guess=(Ta, xb, Ta+10))
            if Ta > Tb:
                T = Ta
                x = xa
            else:
                T = Tb
                x = xb
        self._x_dew = x
        thermal_condition.T = T
        self._save_dew_point()
        self._data[:] = data_copy
        g_mol[:] += L_mol + l_mol
        L_mol.clear()
        l_mol.clear()
        return T
    
    def _H_hat_err_at_T(self, T, H_hat, F_mass, phase_data):
        P = self.thermal_condition.P
        self._set_TP(T, P)
        return self.mixture.xH(phase_data, T, P) / F_mass - H_hat
    
    def _H_hat_err_at_P(self, P, H_hat, F_mass, phase_data):
        T = self.thermal_condition.T
        self._set_TP(T, P)
        return self.mixture.xH(phase_data, T, P) / F_mass - H_hat
    
    def _V_hat_err_at_T(self, T, V):
        P = self.thermal_condition.P
        self._set_TP(T, P)
        return self.vapor_fraction - V
    
    def _V_hat_err_at_P(self, P, V):
        T = self.thermal_condition.T
        self._set_TP(T, P)
        return self.vapor_fraction - V
    
    def _S_hat_err_at_T(self, T, H_hat, F_mass, phase_data):
        P = self.thermal_condition.P
        self._set_TP(T, P)
        return self.mixture.xS(phase_data, T, P) / F_mass - H_hat
    
    def _S_hat_err_at_P(self, P, H_hat, F_mass, phase_data):
        T = self.thermal_condition.T
        self._set_TP(T, P)
        return self.mixture.xS(phase_data, T, P) / F_mass - H_hat
    
    def _set_PH(self, P, H):
        thermal_condition = self.thermal_condition
        thermal_condition.P = P
        T_guess = thermal_condition.T
        imol = self.imol
        phase_data = tuple(imol)
        
        # Check if super heated vapor
        T_dew = self.dew_point_at_P(P)
        H_dew = self.mixture.xH(phase_data, T_dew, P)
        dH_dew = H - H_dew
        if dH_dew >= 0:
            thermal_condition.T = self.mixture.xsolve_T_at_HP(phase_data, H, T_dew, P)
            return
        
        # Check if subcooled liquid
        T_bubble = self.bubble_point_at_P(P)
        H_bubble = self.mixture.xH(phase_data, T_bubble, P)
        dH_bubble = H - H_bubble
        if dH_bubble <= 0:
            thermal_condition.T = self.mixture.xsolve_T_at_HP(phase_data, H, T_bubble, P)
            return
        
        F_mass = self.mixture.MW(imol.data.sum(axis=0))
        H_hat = H / F_mass
        H_hat_bubble = H_bubble / F_mass
        H_hat_dew = H_dew / F_mass
        T0 = flx.IQ_interpolation(
            self._H_hat_err_at_T, T_bubble, T_dew, 
            H_hat_bubble - H_hat, H_hat_dew - H_hat,
            T_guess, 1e-3, 1e-3,
            (H_hat, F_mass, phase_data),
            checkiter=False, checkbounds=False,
            maxiter=10,
        )
        T1 = self.mixture.xsolve_T_at_HP(phase_data, H, thermal_condition.T, P)
        T = flx.IQ_interpolation(
            self._H_hat_err_at_T, 2 * T0 - T1, T1, 
            xtol=1e-6, ytol=1e-6,
            args=(H_hat, F_mass, phase_data),
            checkiter=False, checkbounds=False,
            maxiter=100,
        )
        self.thermal_condition.T = self.mixture.xsolve_T_at_HP(phase_data, H, T, P)

    def _set_TH(self, T, H):
        thermal_condition = self.thermal_condition
        thermal_condition.T = T
        P_guess = thermal_condition.P
        imol = self.imol
        phase_data = tuple(imol)
        
        # Check if super heated vapor
        P_dew = self.dew_point_at_T(T)
        H_dew = self.mixture.xH(phase_data, T, P_dew)
        dH_dew = H - H_dew
        if dH_dew >= 0:
            raise NotImplementedError('cannot solve for pressure yet')
        
        # Check if subcooled liquid
        P_bubble = self.bubble_point_at_T(T)
        H_bubble = self.mixture.xH(phase_data, T, P_bubble)
        dH_bubble = H - H_bubble
        if dH_bubble <= 0:
            raise NotImplementedError('cannot solve for pressure yet')
        
        F_mass = self.mixture.MW(imol.data.sum(axis=0))
        H_hat = H / F_mass
        H_hat_bubble = H_bubble / F_mass
        H_hat_dew = H_dew / F_mass
        flx.IQ_interpolation(
            self._H_hat_err_at_P, P_bubble, P_dew, 
            H_hat_bubble - H_hat, H_hat_dew - H_hat,
            P_guess, 1, 1e-6,
            (H_hat, F_mass, phase_data),
            checkiter=False, checkbounds=False,
            maxiter=100,
        )

    def _set_PS(self, P, S):
        thermal_condition = self.thermal_condition
        thermal_condition.P = P
        T_guess = thermal_condition.T
        imol = self.imol
        phase_data = tuple(imol)
        
        # Check if super heated vapor
        T_dew = self.dew_point_at_P(P)
        S_dew = self.mixture.xS(phase_data, T_dew, P)
        dS_dew = S - S_dew
        if dS_dew >= 0:
            thermal_condition.T = self.mixture.xsolve_T_at_SP(phase_data, S, T_dew, P)
            return
        
        # Check if subcooled liquid
        T_bubble = self.bubble_point_at_P(P)
        S_bubble = self.mixture.xS(phase_data, T_bubble, P)
        dS_bubble = S - S_bubble
        if dS_bubble <= 0:
            thermal_condition.T = self.mixture.xsolve_T_at_SP(phase_data, S, T_bubble, P)
            return
        
        F_mass = self.mixture.MW(imol.data.sum(axis=0))
        S_hat = S / F_mass
        S_hat_bubble = S_bubble / F_mass
        S_hat_dew = S_dew / F_mass
        T0 = flx.IQ_interpolation(
            self._S_hat_err_at_T, T_bubble, T_dew, 
            S_hat_bubble - S_hat, S_hat_dew - S_hat,
            T_guess, 1e-3, 1e-3,
            (S_hat, F_mass, phase_data),
            checkiter=False, checkbounds=False,
            maxiter=10,
        )
        T1 = self.mixture.xsolve_T_at_SP(phase_data, S, thermal_condition.T, P)
        T = flx.IQ_interpolation(
            self._S_hat_err_at_T, 2 * T0 - T1, T1, 
            xtol=1e-6, ytol=1e-6,
            args=(S_hat, F_mass, phase_data),
            checkiter=False, checkbounds=False,
            maxiter=100,
        )
        self.thermal_condition.T = self.mixture.xsolve_T_at_SP(phase_data, S, T, P)
        
    def _set_TS(self, T, S):
        thermal_condition = self.thermal_condition
        thermal_condition.T = T
        P_guess = thermal_condition.P
        imol = self.imol
        phase_data = tuple(imol)
        
        # Check if super heated vapor
        P_dew = self.dew_point_at_T(T)
        S_dew = self.mixture.xS(phase_data, P_dew, T)
        dS_dew = S - S_dew
        if dS_dew >= 0:
            thermal_condition.P = self.mixture.xsolve_P_at_ST(phase_data, S, P_dew, T)
            return
        
        # Check if subcooled liquid
        P_bubble = self.bubble_point_at_T(T)
        S_bubble = self.mixture.xS(phase_data, T, P_bubble)
        dS_bubble = S - S_bubble
        if dS_bubble <= 0:
            thermal_condition.P = self.mixture.xsolve_P_at_ST(phase_data, S, P_bubble, T)
            return
        
        F_mass = self.mixture.MW(imol.data.sum(axis=0))
        S_hat = S / F_mass
        S_hat_bubble = S_bubble / F_mass
        S_hat_dew = S_dew / F_mass
        flx.IQ_interpolation(
            self._S_hat_err_at_P, P_bubble, P_dew, 
            S_hat_bubble - S_hat, S_hat_dew - S_hat,
            P_guess, 1, 1e-6,
            (S_hat, F_mass, phase_data),
            checkiter=False, checkbounds=False,
            maxiter=100,
        )
        
    def _set_PV(self, P, V):
        thermal_condition = self.thermal_condition
        thermal_condition.P = P
        if V == 1:
            self.dew_point_at_P(P)
        elif V == 0:
            self.bubble_point_at_P(P)
        else:
            T_bubble = self.bubble_point_at_P(P)
            T_dew = self.dew_point_at_P(P)
            flx.IQ_interpolation(
                self._V_hat_err_at_T, T_bubble, T_dew, 
                thermal_condition.T, xtol=1e-6, ytol=1e-6, args=(V,),
                checkiter=False, checkbounds=False,
                maxiter=100,
            )

    def _set_TV(self, T, V):
        thermal_condition = self.thermal_condition
        thermal_condition.T = T
        if V == 1:
            self.dew_point_at_T(T)
        elif V == 0:
            self.bubble_point_at_T(T)
        else:
            P_bubble = self.bubble_point_at_T(T)
            P_dew = self.dew_point_at_T(T)
            flx.IQ_interpolation(
                self._V_hat_err_at_P, P_bubble, P_dew, 
                x=thermal_condition.P, xtol=1, ytol=1e-6, args=(V,),
                checkiter=False, checkbounds=False,
                maxiter=100,
            )
    
class VLLECache(Cache): load = VLLE
del Cache, Equilibrium

