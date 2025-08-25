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
from .vle import VLE
from .lle import LLE
from .equilibrium import Equilibrium
from ..utils import Cache
import thermosteam as tmo
import numpy as np
from scipy.optimize import root
from numba import njit

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
    
    def __call__(self, *, T=None, P=None, V=None, H=None, S=None, top_chemical=None):
        """
        Perform vapor-liquid-liquid equilibrium.

        Parameters
        ----------
        T : float, optional
            Operating temperature [K].
        P : float, optional
            Operating pressure [Pa].
        V : float, optional
            Molar vapor fraction.
        H : float, optional
            Enthalpy [kJ/hr].
        S : float, optional
            Entropy [kJ/hr/K]
        
        Notes
        -----
        You may only specify two of the following parameters: P, H, T, V, S.
        
        """
        ### Decide what kind of equilibrium to run ###
        T_spec = T is not None
        P_spec = P is not None
        V_spec = V is not None
        H_spec = H is not None
        S_spec = S is not None
        N_specs = (T_spec + P_spec + V_spec + H_spec + S_spec)
        assert N_specs == 2, ("must pass two and only two of the following "
                              "specifications: T, P, V, H, S")
        self.top_chemical = self.chemicals.IDs[self.imol.data.sum(axis=0).argmax()] if top_chemical is None else top_chemical
        # Run equilibrium
        if T_spec:
            if P_spec:
                try:
                    self.set_TP(T, P)
                except NoEquilibrium:
                    thermal_condition = self._thermal_condition
                    thermal_condition.T = T
                    thermal_condition.P = P
            elif V_spec:
                try:
                    self.set_TV(T, V)
                except NoEquilibrium:
                    thermal_condition = self._thermal_condition
                    thermal_condition.T = T
            elif H_spec:
                self.set_TH(T, H)
            elif S_spec:
                self.set_TS(T, S)
        elif P_spec:
            if V_spec:
                try:
                    self.set_PV(P, V)
                except NoEquilibrium:
                    thermal_condition = self._thermal_condition
                    thermal_condition.P = P
            elif H_spec:
                try:
                    self.set_PH(P, H)
                except NoEquilibrium:
                    thermal_condition = self._thermal_condition
                    thermal_condition.P = P
            elif S_spec:
                try:
                    self.set_PS(P, S)
                except:
                    try:
                        self.set_PS(P, S)
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
    
    def set_TP(self, T, P):
        thermal_condition = self._thermal_condition
        thermal_condition.T = T
        thermal_condition.P = P
        chemicals = self.chemicals
        imol = self.imol
        data = imol['Lgl']
        indices = chemicals.get_vlle_indices(data.nonzero_keys())
        subdata = data[:, indices]
        total = subdata.sum()
        subdata = subdata / total
        z = subdata.sum(axis=0)
        self.iter = 0
        new_subdata = flx.fixed_point(
            self._iter_flows_at_TP, subdata, xtol=1e-6, 
            args=(T, P, indices, z),
            convergenceiter=10, 
            checkconvergence=False,
            checkiter=False,
            maxiter=100,
        )
        data[:, indices] = total * new_subdata * z / new_subdata.sum(axis=0)
    
    def _iter_flows_at_TP(self, data, T, P, indices, z):
        imol = self.imol
        L_mol, g_mol, l_mol = imol['Lgl']
        data *= z / data.sum(axis=0)
        L_mol[indices] = data[0]
        g_mol[indices] = data[1]
        l_mol[indices] = data[2]
        self.iter += 1
        
        # LLE
        try:
            self.lle(T=T, P=P, top_chemical=self.top_chemical, use_cache=True)
        except:
            self.lle(T=T, P=P, top_chemical=self.top_chemical)
        
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
            y[y < 1e-16] = 1e-16
            xa[xa < 1e-16] = 1e-16
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
        V1 = y.sum() 
        xb = l_mol[indices]
        Lb = xb.sum()
        if V1 and Lb:
            y[y < 1e-16] = 1e-16
            xb[xb < 1e-16] = 1e-16
            y /= V1
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
            guess = np.array([V1, La, Lb])
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
            if V < 1e-12 or La < 1e-12 or Lb < 1e-12:
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
        values[values < 1e-12] = 0
        return values
    
    # def set_TP(self, T, P):
    #     thermal_condition = self._thermal_condition
    #     thermal_condition.T = T
    #     thermal_condition.P = P
    #     chemicals = self.chemicals
    #     imol = self.imol
    #     data = imol['Lgl']
    #     indices = chemicals.get_vlle_indices(data.nonzero_keys())
    #     subdata = data[:, indices]
    #     total = subdata.sum()
    #     subdata = subdata / total
    #     z = subdata.sum(axis=0)
        
    #     Ka, Kb = self._evaluate_Ks(z, T, P, indices, z)
    #     if Ka is None or Kb is None:
    #         new_values = data[:, indices]
    #         data[:, indices] *= total * new_values * z / new_values.sum(axis=0)
    #     else:
    #         self.iter = 0
    #         logKs = np.log([Ka, Kb])
    #         new_subdata = flx.fixed_point(
    #             self._iter_flows_at_TP, logKs, xtol=1e-6, 
    #             args=(T, P, indices, z),
    #             convergenceiter=10, 
    #             checkconvergence=False,
    #             checkiter=False,
    #             maxiter=100,
    #         )
    #         data[:, indices] = total * new_subdata * z / new_subdata.sum(axis=0)
    
    # def _evaluate_Ks(self, data, T, P, indices, z):
    #     imol = self.imol
    #     L_mol, g_mol, l_mol = imol['Lgl']
    #     data *= z / data.sum(axis=0)
    #     L_mol[indices] = data[0]
    #     g_mol[indices] = data[1]
    #     l_mol[indices] = data[2]
    #     self.iter += 1
        
    #     # LLE
    #     try:
    #         self.lle(T=T, P=P, top_chemical=self.top_chemical, use_cache=True)
    #     except:
    #         self.lle(T=T, P=P, top_chemical=self.top_chemical)
        
    #     # VLE with extract
    #     try:
    #         self.vle_L._setup()
    #         self.vle_L._solve_TP(T=T, P=P)
    #     except:
    #         self.vle_L(T=T, P=P)
    #     y = g_mol[indices]
    #     V0 = y.sum()
    #     xa = L_mol[indices]
    #     La = xa.sum()
    #     if La and V0: 
    #         y[y < 1e-16] = 1e-16
    #         xa[xa < 1e-16] = 1e-16
    #         y /= V0
    #         xa /= La
    #         Ka = y / xa
    #     else:
    #         Ka = None
        
    #     # VLE with raffinate
    #     try:
    #         self.vle_l._setup()
    #         self.vle_l._solve_TP(T=T, P=P)
    #     except:
    #         self.vle_l(T=T, P=P)
    #     y = g_mol[indices]
    #     V1 = y.sum() 
    #     xb = l_mol[indices]
    #     Lb = xb.sum()
    #     if V1 and Lb:
    #         y[y < 1e-16] = 1e-16
    #         xb[xb < 1e-16] = 1e-16
    #         y /= V1
    #         xb /= Lb
    #         Kb = y / xb
    #     else:
    #         Kb = None
        
    #     return Ka, Kb
    
    def _T_bubble_iter(self, T):
        thermal_condition = self._thermal_condition
        self.lle(
            T=T,
            P=thermal_condition.P,
            top_chemical=self.top_chemical
        )
        self.vle_L(V=0, P=thermal_condition.P)
        return thermal_condition.T
     
    def _P_bubble_iter(self, P):
        thermal_condition = self._thermal_condition
        self.lle(
            T=thermal_condition.T,
            P=P,
            top_chemical=self.top_chemical
        )
        self.vle_L(V=0, P=P)
        return thermal_condition.P
       
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
        return flx.wegstein(
            self._P_bubble_iter, 
            thermal_condition.P,
            xtol=1e-6, 
            convergenceiter=10, 
            checkconvergence=False,
            checkiter=False,
            maxiter=100,
        )
    
    def dew_point_at_T(self, T=None):
        thermal_condition = self.thermal_condition
        imol = self.imol
        imol['g'] += imol['L'] + imol['l']
        imol['l'].clear()
        imol['L'].clear()
        if T is None: 
            T = thermal_condition.T
        else:
            thermal_condition.T = T
        self.vle_l(V=1, T=T)
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
        return flx.wegstein(
            self._T_bubble_iter, 
            thermal_condition.T,
            xtol=1e-6, 
            convergenceiter=10, 
            checkconvergence=False,
            checkiter=False,
            maxiter=100,
        )
    
    def dew_point_at_P(self, P=None):
        thermal_condition = self.thermal_condition
        imol = self.imol
        imol['g'] += imol['L'] + imol['l']
        imol['l'].clear()
        imol['L'].clear()
        if P is None: 
            P = thermal_condition.P
        else:
            thermal_condition.P = P
        self.vle_l(V=1, P=P)
        return thermal_condition.T
    
    def _H_hat_err_at_T(self, T, H_hat, F_mass, phase_data):
        P = self.thermal_condition.P
        self.set_TP(T, P)
        return self.mixture.xH(phase_data, T, P) / F_mass - H_hat
    
    def _H_hat_err_at_P(self, P, H_hat, F_mass, phase_data):
        T = self.thermal_condition.T
        self.set_TP(T, P)
        return self.mixture.xH(phase_data, T, P) / F_mass - H_hat
    
    @property
    def vapor_fraction(self):
        imol = self.imol
        g = imol['g'].sum()
        return g / (g + imol['lL'].sum())
    
    def _V_hat_err_at_T(self, T, V):
        P = self.thermal_condition.P
        self.set_TP(T, P)
        return self.vapor_fraction - V
    
    def _V_hat_err_at_P(self, P, V):
        T = self.thermal_condition.T
        self.set_TP(T, P)
        return self.vapor_fraction - V
    
    def _S_hat_err_at_T(self, T, H_hat, F_mass, phase_data):
        P = self.thermal_condition.P
        self.set_TP(T, P)
        return self.mixture.xS(phase_data, T, P) / F_mass - H_hat
    
    def _S_hat_err_at_P(self, P, H_hat, F_mass, phase_data):
        T = self.thermal_condition.T
        self.set_TP(T, P)
        return self.mixture.xS(phase_data, T, P) / F_mass - H_hat
    
    def set_PH(self, P, H):
        thermal_condition = self.thermal_condition
        thermal_condition.P = P
        T_guess = thermal_condition.T
        imol = self.imol
        phase_data = tuple(imol)
        
        # Check if subcooled liquid
        T_bubble = self.bubble_point_at_P(P)
        H_bubble = self.mixture.xH(phase_data, T_bubble, P)
        dH_bubble = H - H_bubble
        if dH_bubble <= 0:
            thermal_condition.T = self.mixture.xsolve_T_at_HP(phase_data, H, T_bubble, P)
            return
        
        # Check if super heated vapor
        T_dew = self.dew_point_at_P(P)
        if T_dew <= T_bubble: 
            T_dew, T_bubble = T_bubble, T_dew
            T_dew += 0.5
            T_bubble -= 0.5
        H_dew = self.mixture.xH(phase_data, T_dew, P)
        dH_dew = H - H_dew
        if dH_dew >= 0:
            thermal_condition.T = self.mixture.xsolve_T_at_HP(phase_data, H, T_dew, P)
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

    def set_TH(self, T, H):
        thermal_condition = self.thermal_condition
        thermal_condition.T = T
        P_guess = thermal_condition.P
        imol = self.imol
        phase_data = tuple(imol)
        
        # Check if subcooled liquid
        P_bubble = self.bubble_point_at_T(T)
        H_bubble = self.mixture.xH(phase_data, T, P_bubble)
        dH_bubble = H - H_bubble
        if dH_bubble <= 0:
            raise NotImplementedError('cannot solve for pressure yet')
        
        # Check if super heated vapor
        P_dew = self.dew_point_at_T(T)
        if P_bubble <= P_dew: 
            P_dew, P_bubble = P_bubble, P_dew
            P_dew += 0.5
            P_bubble -= 0.5
        H_dew = self.mixture.xH(phase_data, T, P_dew)
        dH_dew = H - H_dew
        if dH_dew >= 0:
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

    def set_PS(self, P, S):
        thermal_condition = self.thermal_condition
        thermal_condition.P = P
        T_guess = thermal_condition.T
        imol = self.imol
        phase_data = tuple(imol)
        
        # Check if subcooled liquid
        T_bubble = self.bubble_point_at_P(P)
        S_bubble = self.mixture.xS(phase_data, T_bubble, P)
        dS_bubble = S - S_bubble
        if dS_bubble <= 0:
            thermal_condition.T = self.mixture.xsolve_T_at_SP(phase_data, S, T_bubble, P)
            return
        
        # Check if super heated vapor
        T_dew = self.dew_point_at_P(P)
        if T_dew <= T_bubble: 
            T_dew, T_bubble = T_bubble, T_dew
            T_dew += 0.5
            T_bubble -= 0.5
        S_dew = self.mixture.xS(phase_data, T_dew, P)
        dS_dew = S - S_dew
        if dS_dew >= 0:
            thermal_condition.T = self.mixture.xsolve_T_at_SP(phase_data, S, T_dew, P)
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
        
    def set_TS(self, T, S):
        thermal_condition = self.thermal_condition
        thermal_condition.T = T
        P_guess = thermal_condition.P
        imol = self.imol
        phase_data = tuple(imol)
        
        # Check if subcooled liquid
        P_bubble = self.bubble_point_at_T(T)
        S_bubble = self.mixture.xS(phase_data, T, P_bubble)
        dS_bubble = S - S_bubble
        if dS_bubble <= 0:
            thermal_condition.P = self.mixture.xsolve_P_at_ST(phase_data, S, P_bubble, T)
            return
        
        # Check if super heated vapor
        P_dew = self.dew_point_at_T(T)
        if P_bubble <= P_dew: 
            P_dew, P_bubble = P_bubble, P_dew
            P_dew += 0.5
            P_bubble -= 0.5
        S_dew = self.mixture.xS(phase_data, P_dew, T)
        dS_dew = S - S_dew
        if dS_dew >= 0:
            thermal_condition.P = self.mixture.xsolve_P_at_ST(phase_data, S, P_dew, T)
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
        
    def set_PV(self, P, V):
        thermal_condition = self.thermal_condition
        thermal_condition.P = P
        if V == 0:
            self.bubble_point_at_P(P)
        elif V == 1:
            self.dew_point_at_P(P)
        else:
            T_bubble = self.bubble_point_at_P(P)
            T_dew = self.dew_point_at_P(P)
            flx.IQ_interpolation(
                self._V_hat_err_at_T, T_bubble, T_dew, 
                thermal_condition.T, xtol=1e-6, ytol=1e-6, args=(V,),
                checkiter=False, checkbounds=False,
                maxiter=100,
            )

    def set_TV(self, T, V):
        thermal_condition = self.thermal_condition
        thermal_condition.T = T
        if V == 0:
            self.bubble_point_at_T(T)
        elif V == 1:
            self.dew_point_at_T(T)
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

