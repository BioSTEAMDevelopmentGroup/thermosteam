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

__all__ = ('VLLE', 'VLLECache')

EXTRACT_INDEX = 0
GAS_INDEX = 1
RAFFINATE_INDEX = 2

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
            tmo.base.SparseArray.from_rows([imol['g'], imol['l']]),
            phases=('g', 'l')
        )
        imol_vle_L = tmo.indexer.MaterialIndexer.from_data(
            tmo.base.SparseArray.from_rows([imol['g'], imol['L']]),
            phases=('g', 'l')
        )
        imol_lle = tmo.indexer.MaterialIndexer.from_data(
            tmo.base.SparseArray.from_rows([imol['L'], imol['l']]),
            phases=('L', 'l')
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
                    self.set_thermal_condition(T, P)
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
       
    def set_thermal_condition(self, T, P):
        thermal_condition = self._thermal_condition
        thermal_condition.T = T
        thermal_condition.P = P
        data = self.imol.data
        total = data.sum()
        new_data = flx.fixed_point(
            self._iter_flows_at_TP, data.to_array() / data.sum(), xtol=1e-6, 
            args=(T, P),
            convergenceiter=10, 
            checkconvergence=False,
            checkiter=False,
            maxiter=100,
        )
        data[:] = total * new_data
    
    def _iter_flows_at_TP(self, data, T, P):
        self.imol.data[:] = data / data.sum()
        self.lle(T=T, P=P, top_chemical=self.top_chemical)
        self.vle_L(T=T, P=P)
        self.vle_l(T=T, P=P)
        return self.imol.data.to_array()
      
    def _iter_flows_T_at_HP(self, dataT, H, P):
        thermal_condition = self._thermal_condition
        data = dataT[:-1]
        thermal_condition.T = T = dataT[-1]
        self.imol.data.flat[:] = data / data.sum()
        self.lle(T=T, P=P, top_chemical=self.top_chemical)
        self.vle_L(T=T, P=P)
        self.vle_l(T=T, P=P)
        dataT_new = np.array([*self.imol.data.flat, self.mixture.xsolve_T_at_HP(tuple(self.imol), H, T, P)])
        return dataT_new  
      
    def _iter_flows_T_at_bubble_point_P(self, dataT, P):
        thermal_condition = self._thermal_condition
        data = dataT[:-1]
        thermal_condition.T = dataT[-1]
        mixture = self.mixture
        data = data / data.sum()
        N = self.chemicals.size
        imol = self.imol
        L_mol = imol['L']
        l_mol = imol['l']
        L_mol = data[:N]
        l_mol = data[N:]
        self.lle(T=thermal_condition.T, P=P, top_chemical=self.top_chemical)
        self.vle_L(V=0, P=P)
        H_L = mixture.H('l', L_mol, thermal_condition.T, P)
        self.vle_l(V=0, P=P)
        phase_data = [(i, j) for (i, j) in imol if i != 'L']
        H_other = mixture.xH(phase_data, thermal_condition.T, P)
        T = mixture.xsolve_T_at_HP(
            tuple(imol), H_other + H_L, thermal_condition.T, P
        )
        return np.array([*L_mol, *l_mol, T])
    
    def bubble_point_at_P(self, P=None):
        thermal_condition = self._thermal_condition
        if P is None: 
            P = thermal_condition.P
        else:
            thermal_condition.P = P
        imol = self.imol
        L_mol = imol['L']
        l_mol = imol['l']
        g_mol = imol['g']
        l_mol += g_mol
        g_mol[:] = 0
        self.lle(T=thermal_condition.T, P=P, top_chemical=self.top_chemical)
        if not (imol['l'].any()):
            self.vle_L(V=0, P=P)
            return
        
        total = (L_mol + l_mol).sum()
        L_mol /= total
        l_mol /= total
        dataT = np.array([*L_mol, *l_mol, thermal_condition.T])
        flx.fixed_point(
            self._iter_flows_T_at_bubble_point_P, 
            dataT, args=(P,), xtol=1e-6, 
            convergenceiter=10, 
            checkconvergence=False,
            checkiter=False,
            maxiter=100,
        )
        L_mol *= total
        l_mol *= total
        return thermal_condition.T
    
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
        self.set_thermal_condition(T, P)
        return self.mixture.xH(phase_data, T, P) / F_mass - H_hat
    
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
            thermal_condition.T = self.mixture.solve_T_at_HP('l', self.imol['l'] + self.imol['L'], H, T_bubble, P)
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
        # data = self.imol.data
        # total = data.sum()
        # data /= total
        # dataT = np.array([*data.flat, thermal_condition.T])
        # flx.fixed_point(
        #     self._iter_flows_T_at_HP, 
        #     dataT, args=(H, P), xtol=1e-6, 
        #     convergenceiter=10, 
        #     checkconvergence=False,
        #     checkiter=False
        # )
        # data *= total
        
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

class VLLECache(Cache): load = VLLE
del Cache, Equilibrium

