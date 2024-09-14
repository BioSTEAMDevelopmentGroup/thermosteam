# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2025, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from scipy.optimize import shgo
import flexsolve as flx
from numba import njit
from warnings import warn
from .vle import VLE
from .lle import LLE
from ..base import SparseVector
from ..exceptions import InfeasibleRegion, NoEquilibrium
from . import binary_phase_fraction as binary
from .equilibrium import Equilibrium
from .dew_point import DewPoint
from .bubble_point import BubblePoint
from .fugacity_coefficients import IdealFugacityCoefficients
from .poyinting_correction_factors import MockPoyintingCorrectionFactors
from . import activity_coefficients as ac
from .. import functional as fn
from .lle import pseudo_equilibrium_outer_loop
from ..utils import Cache
import thermosteam as tmo
import numpy as np

__all__ = ('VLLE', 'VLLECache')


def psuedo_equilibrium_inner_loop(logKgammay, z, T, n, f_gamma, gamma_args, phi):
    logKgammay_new = logKgammay.copy()
    K = np.exp(logKgammay[:n])
    x = z/(1. + phi * (K - 1.))
    x = x / x.sum()
    gammay = logKgammay[n:]
    gammax = f_gamma(x, T, *gamma_args)
    K = gammax / gammay 
    y = K * x
    y /= y.sum()
    gammay = f_gamma(y, T, *gamma_args)
    K = gammax / gammay
    logKgammay_new[n:] = np.log(K)
    logKgammay_new[n:] = gammay
    return logKgammay_new

def pseudo_equilibrium_outer_loop(logKgammayphi, z, T, n, f_gamma, gamma_args, inner_loop_options):
    logKgammayphi_new = logKgammayphi.copy()
    # breakpoint()
    logKgammay = logKgammayphi[:-1]
    phi = logKgammayphi[-1]
    args=(z, T, n, f_gamma, gamma_args)
    logKgammay = flx.aitken(
        psuedo_equilibrium_inner_loop, logKgammay, 
        args=(*args, phi), **inner_loop_options,
    )
    K = np.exp(logKgammay[:n])
    try:
        phi = phase_fraction(z, K, phi)
    except (ZeroDivisionError, FloatingPointError):
        raise NoEquilibrium
    if np.isnan(phi): raise NoEquilibrium
    if phi > 1: phi = 1 - 1e-16
    if phi < 0: phi = 1e-16
    logKgammayphi_new[:-1] = logKgammay
    logKgammayphi_new[-1] = phi
    return logKgammayphi_new

def pseudo_equilibrium(K, phi, z, T, n, f_gamma, gamma_args, inner_loop_options, outer_loop_options):
    phi = phase_fraction(z, K, phi)
    try:
        x = z/(1. + phi * (K - 1.))
    except:
        x = np.ones(n)
    x /= x.sum()
    y = K * x
    logKgammayphi = np.zeros(2*n + 1)
    logKgammayphi[:n] = np.log(K)
    logKgammayphi[n:-1] = f_gamma(y, T, *gamma_args)
    logKgammayphi[-1] = phi
    try:
        logKgammayphi = flx.aitken(
            pseudo_equilibrium_outer_loop, logKgammayphi,
            args=(z, T, n, f_gamma, gamma_args, inner_loop_options),
            **outer_loop_options,
        )
    except NoEquilibrium:
        return z
    K = np.exp(logKgammayphi[:n])
    phi = logKgammayphi[-1]
    return z/(1. + phi * (K - 1.)) * (1 - phi)

def _zliq_V_logKg0_logK0_gamma1_phi_iter(
        zliq_V_logKg0_logK0_gamma1_phi, 
        pcf_Psat_over_P, T, P,
        z, z_light, z_heavy, f_gamma, gamma_args,
        f_phi, n,
    ):
    new_values = zliq_V_logKg0_logK0_gamma1_phi.copy()
    n2 = 2 * n
    # LLE outer loop (liq-LIQ equilibrium)
    zliq = zliq_V_logKg0_logK0_gamma1_phi[:n]
    logK0_gamma1_phi_new = pseudo_equilibrium_outer_loop(
        zliq_V_logKg0_logK0_gamma1_phi[n2+1:],
        z, T, n, f_gamma, gamma_args, inner_loop_options=LLE.pseudo_equilibrium_inner_loop_options
    )
    phi = logK0_gamma1_phi_new[-1]
    K0 = np.exp(logK0_gamma1_phi_new[:n])
    x = zliq / zliq.sum()
    x0 = x / (1. + phi * (K0 - 1.))
    x0 = x0 / x0.sum()
    x1 = K0 * x0
    x1 = x1 / x1.sum()
    
    # Update Kg0 through gas-liq equilibrium
    V = zliq_V_logKg0_logK0_gamma1_phi[n]
    logKg0 = zliq_V_logKg0_logK0_gamma1_phi[n:n2]
    Kg0 = np.exp(logKg0)
    y = z - zliq
    # y = x0 * Kg0
    y /= y.sum()
    Kg0 = f_gamma(x0, T, *gamma_args) * pcf_Psat_over_P / f_phi(y, T, P)
    Kg0[Kg0 < 1e-16] = 1e-16
    if V < 0.: V = 1e-16
    elif V > 1.: V = 1. - 1e-16
    
    # Update zliq through gas-liq equilibrium and Rashford-Rice
    z0 = x0 * zliq.sum() * phi + (z - zliq)
    V0 = binary.solve_phase_fraction_Rashford_Rice(z0, Kg0, V, z_light, z_heavy)
    x0 = z0 / (1. + V0 * (Kg0 - 1.))
    zliq = (1 - V) * (1 - phi) * x1 + (1 - V0) * x0 * z0.sum()
    V = 1. - zliq.sum()
    # Update x through mass balance
    print('----')
    print(zliq)
    print(y)
    print(V)
    print(phi)
    print(Kg0)
    breakpoint()
    new_values[:n] = zliq
    new_values[n] = V
    new_values[n+1:n2+1] = np.log(Kg0)
    new_values[n2+1:] = logK0_gamma1_phi_new
    return new_values

def set_flows(vapor_mol, liquid_mol, index, vapor_data, total_data):
    vapor_mol[index] = vapor_data
    liquid_mol[index] = total_data - vapor_data

class VLLE(Equilibrium, phases='lLg'):
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
    
    Examples
    --------
    First create a VLLE object:
    
    >>> from thermosteam import indexer, equilibrium, settings
    >>> settings.set_thermo(['Water', 'Ethanol', 'Methanol', 'Propanol'], cache=True)
    >>> imol = indexer.MolarFlowIndexer(
    ...             l=[('Water', 304), ('Ethanol', 30)],
    ...             g=[('Methanol', 40), ('Propanol', 1)])
    >>> vlle = equilibrium.VLLE(imol)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Methanol', 40), ('Propanol', 1)],
            l=[('Water', 304), ('Ethanol', 30)]),
        thermal_condition=ThermalCondition(T=298.15, P=101325))
    
    Equilibrium given vapor fraction and pressure:
    
    >>> vlle(V=0.5, P=101325)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 126.7), ('Ethanol', 26.38), ('Methanol', 33.49), ('Propanol', 0.8958)],
            l=[('Water', 177.3), ('Ethanol', 3.622), ('Methanol', 6.509), ('Propanol', 0.1042)]),
        thermal_condition=ThermalCondition(T=363.85, P=101325))
    
    Equilibrium given temperature and pressure:
    
    >>> vlle(T=363.88, P=101325)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 127.4), ('Ethanol', 26.41), ('Methanol', 33.54), ('Propanol', 0.8968)],
            l=[('Water', 176.6), ('Ethanol', 3.59), ('Methanol', 6.456), ('Propanol', 0.1032)]),
        thermal_condition=ThermalCondition(T=363.88, P=101325))
    
    Equilibrium given enthalpy and pressure:
    
    >>> H = vlle.thermo.mixture.xH(vlle.imol, T=363.88, P=101325)
    >>> vlle(H=H, P=101325)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 127.4), ('Ethanol', 26.41), ('Methanol', 33.54), ('Propanol', 0.8968)],
            l=[('Water', 176.6), ('Ethanol', 3.59), ('Methanol', 6.456), ('Propanol', 0.1032)]),
        thermal_condition=ThermalCondition(T=363.88, P=101325))
    
    Equilibrium given entropy and pressure:
    
    >>> S = vlle.thermo.mixture.xS(vlle.imol, T=363.88, P=101325)
    >>> vlle(S=S, P=101325)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 127.4), ('Ethanol', 26.41), ('Methanol', 33.54), ('Propanol', 0.8968)],
            l=[('Water', 176.6), ('Ethanol', 3.59), ('Methanol', 6.456), ('Propanol', 0.1032)]),
        thermal_condition=ThermalCondition(T=363.88, P=101325))
    
    Equilibrium given vapor fraction and temperature:
    
    >>> vlle(V=0.5, T=363.88)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 126.7), ('Ethanol', 26.38), ('Methanol', 33.49), ('Propanol', 0.8958)],
            l=[('Water', 177.3), ('Ethanol', 3.622), ('Methanol', 6.509), ('Propanol', 0.1042)]),
        thermal_condition=ThermalCondition(T=363.88, P=101431))
    
    Equilibrium given enthalpy and temperature:
    
    >>> vlle(H=H, T=363.88)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 127.4), ('Ethanol', 26.41), ('Methanol', 33.54), ('Propanol', 0.8968)],
            l=[('Water', 176.6), ('Ethanol', 3.59), ('Methanol', 6.456), ('Propanol', 0.1032)]),
        thermal_condition=ThermalCondition(T=363.88, P=101325))
    
    Non-partitioning heavy and gaseous chemicals also affect VLLE. Calculation 
    are repeated with non-partitioning chemicals:
        
    >>> from thermosteam import indexer, equilibrium, settings, Chemical
    >>> O2 = Chemical('O2', phase='g')
    >>> Glucose = Chemical('Glucose', phase='l', default=True)
    >>> settings.set_thermo(['Water', 'Ethanol', 'Methanol', 'Propanol', O2, Glucose], cache=True)
    >>> imol = indexer.MolarFlowIndexer(
    ...             l=[('Water', 304), ('Ethanol', 30), ('Glucose', 5)],
    ...             g=[('Methanol', 40), ('Propanol', 1), ('O2', 10)])
    >>> vlle = equilibrium.VLLE(imol)
    >>> vlle(T=363.88, P=101325)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 159.5), ('Ethanol', 27.65), ('Methanol', 35.63), ('Propanol', 0.9337), ('O2', 10)],
            l=[('Water', 144.5), ('Ethanol', 2.352), ('Methanol', 4.369), ('Propanol', 0.0663), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=363.88, P=101325))
    
    >>> vlle(V=0.5, P=101325)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 126.7), ('Ethanol', 26.38), ('Methanol', 33.52), ('Propanol', 0.8957), ('O2', 10)],
            l=[('Water', 177.3), ('Ethanol', 3.618), ('Methanol', 6.478), ('Propanol', 0.1043), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=362.47, P=101325))
    
    >>> H = vlle.thermo.mixture.xH(vlle.imol, T=363.88, P=101325)
    >>> vlle(H=H, P=101325)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 127.4), ('Ethanol', 26.42), ('Methanol', 33.58), ('Propanol', 0.8968), ('O2', 10)],
            l=[('Water', 176.6), ('Ethanol', 3.583), ('Methanol', 6.421), ('Propanol', 0.1032), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=362.51, P=101325))
    
    >>> S = vlle.thermo.mixture.xS(vlle.imol, T=363.88, P=101325)
    >>> vlle(S=S, P=101325)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 128.2), ('Ethanol', 26.45), ('Methanol', 33.63), ('Propanol', 0.8979), ('O2', 10)],
            l=[('Water', 175.8), ('Ethanol', 3.548), ('Methanol', 6.365), ('Propanol', 0.1021), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=362.54, P=101325))
    
    >>> vlle(V=0.5, T=363.88)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 126.7), ('Ethanol', 26.38), ('Methanol', 33.49), ('Propanol', 0.8958), ('O2', 10)],
            l=[('Water', 177.3), ('Ethanol', 3.622), ('Methanol', 6.509), ('Propanol', 0.1042), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=363.88, P=106841))
    
    >>> vlle(H=H, T=363.88)
    >>> vlle 
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 126.7), ('Ethanol', 26.38), ('Methanol', 33.49), ('Propanol', 0.8958), ('O2', 10)],
            l=[('Water', 177.3), ('Ethanol', 3.622), ('Methanol', 6.51), ('Propanol', 0.1042), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=363.88, P=106842))

    >>> vlle(S=S, T=363.88)
    >>> vlle 
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 128.1), ('Ethanol', 26.45), ('Methanol', 33.6), ('Propanol', 0.8978), ('O2', 10)],
            l=[('Water', 175.9), ('Ethanol', 3.555), ('Methanol', 6.399), ('Propanol', 0.1022), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=363.88, P=106562))

    The presence of a non-partitioning gaseous chemical will result in some
    evaporation, even if the tempeture is below the saturated bubble point:
        
    >>> from thermosteam import indexer, equilibrium, settings, Chemical
    >>> O2 = Chemical('O2', phase='g')
    >>> settings.set_thermo(['Water', O2], cache=True)
    >>> imol = indexer.MolarFlowIndexer(
    ...             l=[('Water', 30)],
    ...             g=[('O2', 10)])
    >>> vlle = equilibrium.VLLE(imol)
    >>> vlle(T=300., P=101325)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            g=[('Water', 0.3617), ('O2', 10)],
            l=[('Water', 29.64)]),
        thermal_condition=ThermalCondition(T=300.00, P=101325))

    """
    __slots__ = (
        'vle', 
        'lle',
        'method', # [str] Method for solving equilibrium.
        '_T', # [float] Temperature [K].
        '_P', # [float] Pressure [Pa].
        '_H_hat', # [float] Specific enthalpy [kJ/kg].
        '_S_hat', # [float] Specific entropy [kJ/K/kg].
        '_V', # [float] Molar vapor fraction.
        '_dew_point', # [DewPoint] Solves for dew point.
        '_bubble_point', # [BubblePoint] Solves for bubble point.
        '_x', # [1d array] Vapor composition.
        '_y', # [1d array] Liquid composition.
        '_z_last', # tuple[1d array] Last bulk composition.
        '_gamma',
        '_phi', # [FugacityCoefficients] Estimates fugacity coefficients of gas.
        '_pcf', # [PoyintingCorrectionFactors] Estimates the PCF of a liquid.
        '_zliq_V_logKg0_logK0_gamma1_phi',
        '_LIQUID_mol', # [1d array] LIQUID molar data.
        '_liquid_mol', # [1d array] Liquid molar data.
        '_vapor_mol', # [1d array] Vapor molar data.
        '_phase_data', # tuple[str, 1d array] Phase-data pairs.
        '_v',  # [1d array] Molar vapor data in equilibrium.
        '_index', # [1d array] Index of chemicals in equilibrium.
        '_F_mass', # [float] Total mass data.
        '_chemical', # [Chemical] Single chemical in equilibrium.
        '_mol_vlle', # [1d array] Moles of chemicals in VLLE calculations.
        '_N', # [int] Number of chemicals in equilibrium.
        '_z', # [1d array] Molar composition of chemicals in equilibrium
        '_z_norm', # [1d array] Normalized molar composition of chemicals in equilibrium
        '_z_light', # [1d array] Molar composition of light chemicals not included in equilibrium calculation.
        '_z_heavy', # [1d array] Molar composition of heavy chemicals not included in equilibrium calculation.
        '_nonzero', # [1d array(bool)] Chemicals present in the mixture
        '_F_mol', # [float] Total moles of chemicals (accounting for dissociation).
        '_F_mol_vlle', # [float] Total moles in equilibrium.
        '_F_mol_light', # [float] Total moles of gas chemicals not included in equilibrium calculation.
        '_F_mol_heavy', # [float] Total moles of heavy chemicals not included in equilibrium calculation.
        '_dmol_vlle',
        '_dF_mol',
    )
    maxiter = 20
    T_tol = 5e-8
    P_tol = 1.
    H_hat_tol = 1e-6
    S_hat_tol = 1e-6
    V_tol = 1e-6
    x_tol = 1e-8
    y_tol = 1e-8
    default_method = 'fixed-point'
    
    def __init__(self, imol=None, thermal_condition=None,
                 thermo=None):
        self.method = self.default_method
        self._T = self._P = self._H_hat = self._V = 0
        super().__init__(imol, thermal_condition, thermo)
        imol = self._imol
        thermo = self._thermo
        thermal_condition = self._thermal_condition
        self.vle = VLE(imol, thermal_condition, thermo)
        self.lle = LLE(imol, thermal_condition, thermo)
        self._x = None
        self._z_last = None
        self._nonzero = None
        self._index = ()
    
    def __call__(self, *, T=None, P=None, V=None, H=None, S=None, x=None, y=None):
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
        You may only specify two of the following parameters: P, H, T, V.
        
        """
        data = self._imol.data
        LIQ, gas, liq = data
        liq += LIQ # All flows must be in the 'l' phase for VLE
        LIQ[:] = 0.
        self.vle(T=T, P=P, V=V, H=H, S=S)
        if not gas.any() or not liq.any(): return
        self.lle(T, P)
        if not (LIQ.any() and liq.any()): return
        
        ### Decide what kind of equilibrium to run ###
        T_spec = T is not None
        P_spec = P is not None
        V_spec = V is not None
        H_spec = H is not None
        S_spec = S is not None
        x_spec = x is not None
        y_spec = y is not None
        N_specs = (T_spec + P_spec + V_spec + H_spec + S_spec + x_spec + y_spec)
        assert N_specs == 2, ("must pass two and only two of the following "
                              "specifications: T, P, V, H, S, x, y")
        
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
            elif x_spec:
                self.set_Tx(T, np.asarray(x))
            else: # y_spec
                self.set_Ty(T, np.asarray(y))
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
            elif x_spec:
                self.set_Px(P, np.asarray(x))
            else: # y_spec
                self.set_Py(P, np.asarray(y))
        elif S_spec: # pragma: no cover
            if y_spec:
                raise NotImplementedError('specification S and y is invalid')
            elif x_spec:
                raise NotImplementedError('specification S and x is invalid')
            elif H_spec:
                raise NotImplementedError('specification H and S is invalid')
            else: # V_spec
                raise NotImplementedError('specification V and S not implemented')
        elif H_spec: # pragma: no cover
            if y_spec:
                raise NotImplementedError('specification H and y is invalid')
            elif x_spec:
                raise NotImplementedError('specification H and x is invalid')
            else: # V_spec
                raise NotImplementedError('specification V and H not implemented')
        elif V_spec: # pragma: no cover
            if y_spec:
                raise ValueError("specification V and y is invalid")
            else: # x_spec
                raise ValueError("specification V and x is invalid")
        else: # pragma: no cover
            raise ValueError("can only pass either 'x' or 'y' arguments, not both")
       
    def _load_guess(self):
        z = self._z
        n = z.size
        self._zliq_V_logKg0_logK0_gamma1_phi = guess = np.zeros(4*n + 2)
        liquid_mol = self._liquid_mol
        LIQUID_mol = self._LIQUID_mol
        vapor_mol = self._vapor_mol
        F_liq = liquid_mol.sum()
        F_LIQ = LIQUID_mol.sum()
        F_vap = vapor_mol.sum()
        liquid_mol_total = liquid_mol + LIQUID_mol
        F_liq_total = F_liq + F_LIQ
        zliq = liquid_mol_total / self._F_mol_vlle
        x0 = liquid_mol / F_liq
        x1 = LIQUID_mol / F_LIQ
        y = vapor_mol / F_vap
        Kg0 = y / x0
        K0 = x0 / x1
        guess[:n] = zliq
        guess[n] = F_vap / self._F_mol_vlle
        guess[n+1:n*2+1] = np.log(Kg0)
        guess[n*2+1:n*3+1] = np.log(K0)
        guess[n*3+1:-1] = self._gamma(x1, self._T)
        guess[-1] = F_liq / F_liq_total
    
    def _solve(self, T, P):
        Psats = np.array([i(T) for i in self._bubble_point.Psats])
        pcf_Psat_over_P = self._pcf(T, P, Psats) * Psats / P
        gamma = self._gamma
        z = self._z
        n = z.size
        args = (pcf_Psat_over_P, T, P, z, 
                self._z_light, self._z_heavy, 
                gamma.f, gamma.args, self._phi, n)
        self._zliq_V_logKg0_logK0_gamma1_phi = results = flx.aitken(
            _zliq_V_logKg0_logK0_gamma1_phi_iter,
            self._zliq_V_logKg0_logK0_gamma1_phi,
            args=args, checkiter=False, 
            checkconvergence=False, 
            convergenceiter=5,
            maxiter=self.maxiter
        )
        zliq = results[:n]
        x = zliq / zliq.sum()
        V = results[n]
        Kg0 = np.exp(results[n+1:2*n+1])
        K0 = np.exp(results[2*n+1:3*n+1])
        phi = results[-1]
        x0 = x / (1. + phi * (K0 - 1.))
        x0 = x0 / x0.sum()
        x1 = K0 * x0
        x1 = x1 / x1.sum()
        y = x0 * Kg0
        v = self._F_mol_vlle * V * y
        l0 = x0 * self._F_mol_vlle * (1 - V) * phi
        l1 = self._mol_vlle - v - l0
        l1[l1 < 0] - 0  
        index = self._index
        self._vapor_mol[index] = v 
        self._liquid_mol[index] = l0
        self._LIQUID_mol[index] = l1
    
    def set_thermal_condition(self, T, P):
        self._setup()
        thermal_condition = self._thermal_condition
        self._T = thermal_condition.T = T
        self._P = thermal_condition.P = P
        self._load_guess()
        self._solve(T, P)
        
    def set_TV(self, T, V, gas_conversion=None, liquid_conversion=None):
        self._setup(gas_conversion, liquid_conversion)
        thermal_condition = self._thermal_condition
        thermal_condition.T = self._T = T
        if self._N == 0: raise RuntimeError('no chemicals present to perform VLLE')
        if self._N == 1: return self._set_TV_chemical(T, V)
        if self._F_mol_heavy and V == 1.: V = 1. - 1e-3
        if self._F_mol_light and V == 0.: V = 1e-3
        if V == 1:
            if gas_conversion:
                P_dew, dz, y, x_dew = self._dew_point.solve_Px(self._z, T, gas_conversion)
                self._vapor_mol[self._index] = self._mol_vlle + dz * self._F_mol_vlle
            else:
                P_dew, x_dew = self._dew_point.solve_Px(self._z, T)
                self._vapor_mol[self._index] = self._mol_vlle
            self._liquid_mol[self._index] = 0
            thermal_condition.P = P_dew
        elif V == 0:
            if liquid_conversion:
                P_bubble, dz, y_bubble, x = self._bubble_point.solve_Py(self._z, T)
                self._liquid_mol[self._index] = self._mol_vlle + dz * self._F_mol_vlle
            else:
                P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
                self._liquid_mol[self._index] = self._mol_vlle
            self._vapor_mol[self._index] = 0
            thermal_condition.P = P_bubble
        else:
            if liquid_conversion:
                P_bubble, dz_bubble, y_bubble, _ = self._bubble_point.solve_Py(self._z, T, liquid_conversion)
            else:
                P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
                dz_bubble = None
            if gas_conversion:
                P_dew, dz_dew, y, x_dew = self._dew_point.solve_Px(self._z, T, gas_conversion)
            else:
                P_dew, x_dew = self._dew_point.solve_Px(self._z, T, gas_conversion)
                dz_dew = None
            self._refresh_v(V, y_bubble, x_dew, dz_bubble, dz_dew)
            if self._F_mol_light: P_bubble = 0.1 * self._bubble_point.Pmax + 0.9 * P_bubble
            if self._F_mol_heavy: P_dew = 0.1 * self._bubble_point.Pmin + 0.9 * P_dew
            
            V_bubble = self._V_err_at_P(P_bubble, 0., gas_conversion, liquid_conversion)
            if V_bubble > V:
                F_mol = self._F_mol
                mol = self._mol_vlle
                if liquid_conversion:
                    mol = mol + self._dmol_vlle
                    F_mol = F_mol + self._dF_mol
                F_mol_vapor = F_mol * V
                v = y_bubble * F_mol_vapor
                mask = v > mol
                v[mask] = mol[mask]
                P = P_bubble
            else:
                V_dew = self._V_err_at_P(P_dew, 0., gas_conversion, liquid_conversion)
                if V_dew < V:
                    F_mol = self._F_mol
                    mol = self._mol_vlle
                    if gas_conversion:
                        mol = mol + self._dmol_vlle
                        F_mol = F_mol + self._dF_mol
                    l = x_dew * F_mol * (1. - V)
                    mask = l > mol 
                    l[mask] = mol[mask]
                    v = mol - l
                    P = P_dew
                else:
                    P = flx.IQ_interpolation(
                        self._V_err_at_P,
                        P_bubble, P_dew, V_bubble - V, V_dew - V,
                        self._P, self.P_tol, self.V_tol,
                        (V, gas_conversion, liquid_conversion),
                        checkiter=False, checkbounds=False,
                        maxiter=self.maxiter,
                    )
                    v = self._v
                    mol = self._mol_vlle
                    if gas_conversion or liquid_conversion:
                        mol = mol + self._dmol_vlle
            
            self._P = thermal_condition.P = P
            set_flows(self._vapor_mol, self._liquid_mol, self._index, v, mol)
            if liquid_conversion or gas_conversion:
                try: self._H_hat = (
                        self.mixture.xH(self._phase_data, T, P) + (self.chemicals.Hf * mol).sum()
                    ) / self._F_mass
                except: pass
            else:
                try: self._H_hat = self.mixture.xH(self._phase_data, T, P) / self._F_mass
                except: pass

    def set_TH(self, T, H, gas_conversion=None, liquid_conversion=None):
        self._setup(gas_conversion, liquid_conversion)
        if self._N == 0: raise RuntimeError('no chemicals present to perform VLLE')
        if self._N == 1: return self._set_TH_chemical(T, H)
        self._T = T
        index = self._index
        mol = self._mol_vlle
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        phase_data = self._phase_data
        
        # Check if super heated vapor
        P_dew, x_dew = self._dew_point.solve_Px(self._z, T)
        if self._F_mol_heavy: P_dew = 0.5 * P_dew + 0.5 * self._bubble_point.Pmin
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = self.mixture.xH(phase_data, T, P_dew)
        dH_dew = (H - H_dew)
        if dH_dew >= 0:
            raise NotImplementedError('cannot solve for pressure yet')

        # Check if subcooled liquid
        P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
        if self._F_mol_light: P_bubble = 2 * P_bubble
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = self.mixture.xH(phase_data, T, P_bubble)
        dH_bubble = (H - H_bubble)
        if dH_bubble <= 0:
            raise NotImplementedError('cannot solve for pressure yet')

        # Guess overall vapor fraction, and vapor flow rates
        V = dH_bubble/(H_dew - H_bubble)
        
        # Guess composition in the vapor is a weighted average of boiling points
        self._refresh_v(V, y_bubble, x_dew)
        F_mass = self._F_mass
        H_hat = H/F_mass
        P = flx.IQ_interpolation(
            self._H_hat_err_at_P,
            P_bubble, P_dew,
            H_bubble/F_mass - H_hat, H_dew/F_mass - H_hat,
            self._P, self.P_tol, self.H_hat_tol,
            (H_hat,), checkiter=False, checkbounds=False,
            maxiter=self.maxiter,
        )
        self._P = self._thermal_condition.P = P   
        self._thermal_condition.T = T
    
    def set_TS(self, T, S, gas_conversion=None, liquid_conversion=None):
        self._setup(gas_conversion, liquid_conversion)
        if self._N == 0: raise RuntimeError('no chemicals present to perform VLLE')
        if self._N == 1: return self._set_TS_chemical(T, S)
        self._T = T
        index = self._index
        mol = self._mol_vlle
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        phase_data = self._phase_data
        
        # Check if super heated vapor
        P_dew, x_dew = self._dew_point.solve_Px(self._z, T)
        if self._F_mol_heavy: P_dew = 0.5 * P_dew + 0.5 * self._bubble_point.Pmin
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        S_dew = self.mixture.xS(phase_data, T, P_dew)
        dS_dew = (S - S_dew)
        if dS_dew >= 0:
            raise NotImplementedError('cannot solve for pressure yet')

        # Check if subcooled liquid
        P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
        if self._F_mol_light: P_bubble = 2 * P_bubble
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        S_bubble = self.mixture.xS(phase_data, T, P_bubble)
        dS_bubble = (S - S_bubble)
        if dS_bubble <= 0:
            raise NotImplementedError('cannot solve for pressure yet')

        # Guess overall vapor fraction, and vapor flow rates
        V = dS_bubble/(S_dew - S_bubble)
        
        # Guess composition in the vapor is a weighted average of boiling points
        self._refresh_v(V, y_bubble, x_dew)
        F_mass = self._F_mass
        S_hat = S/F_mass
        P = flx.IQ_interpolation(
            self._S_hat_err_at_P,
            P_bubble, P_dew,
            S_bubble/F_mass - S_hat, S_dew/F_mass - S_hat,
            self._P, self.P_tol, self.S_hat_tol,
            (S_hat,), checkiter=False, checkbounds=False,
            maxiter=self.maxiter,
        )
        self._P = self._thermal_condition.P = P   
        self._thermal_condition.T = T
    
    def set_PV(self, P, V, gas_conversion=None, liquid_conversion=None):
        self._setup(gas_conversion, liquid_conversion)
        self._thermal_condition.P = self._P = P
        if self._N == 0: raise RuntimeError('no chemicals present to perform VLLE')
        if self._N == 1: return self._set_PV_chemical(P, V)
        
        # Setup bounderies
        thermal_condition = self._thermal_condition
        index = self._index
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        if self._F_mol_heavy and V == 1.: V = 1. - 1e-3
        if self._F_mol_light and V == 0.: V = 1e-3
        if V == 1 and not self._F_mol_heavy:
            if gas_conversion:
                T_dew, dz, y, x_dew = self._dew_point.solve_Tx(self._z, P, gas_conversion)
                self._vapor_mol[self._index] = self._mol_vlle + dz * self._F_mol_vlle
            else:
                T_dew, x_dew = self._dew_point.solve_Tx(self._z, P)
                self._vapor_mol[self._index] = self._mol_vlle
            self._liquid_mol[self._index] = 0
            thermal_condition.T = T_dew
        elif V == 0 and not self._F_mol_light:
            if liquid_conversion:
                T_bubble, dz, y_bubble, x = self._bubble_point.solve_Ty(self._z, P, liquid_conversion)
                self._liquid_mol[self._index] = self._mol_vlle + dz * self._F_mol_vlle
            else:
                T_bubble, y_bubble = self._bubble_point.solve_Ty(self._z, P)
                self._liquid_mol[self._index] = self._mol_vlle
            self._vapor_mol[self._index] = 0
            thermal_condition.T = T_bubble
        else:
            if liquid_conversion:
                T_bubble, dz_bubble, y_bubble, _ = self._bubble_point.solve_Ty(self._z, P, liquid_conversion)
            else:
                T_bubble, y_bubble = self._bubble_point.solve_Ty(self._z, P)
                dz_bubble = None
            if gas_conversion:
                T_dew, dz_dew, y, x_dew = self._dew_point.solve_Tx(self._z, P, gas_conversion)
            else:
                T_dew, x_dew = self._dew_point.solve_Tx(self._z, P)
                dz_dew = None
            self._refresh_v(V, y_bubble, x_dew, dz_bubble, dz_dew)
            if self._F_mol_light: T_bubble = 0.1 * self._bubble_point.Tmin + 0.9 * T_bubble
            if self._F_mol_heavy: T_dew = 0.1 * self._bubble_point.Tmax + 0.9 * T_dew
            V_bubble = self._V_err_at_T(T_bubble, 0., gas_conversion, liquid_conversion)
            if V_bubble > V:
                F_mol = self._F_mol
                mol = self._mol_vlle
                if liquid_conversion:
                    mol = mol + self._dmol_vlle
                    F_mol = F_mol + self._dF_mol
                F_mol_vapor = F_mol * V
                v = y_bubble * F_mol_vapor
                mask = v > mol
                v[mask] = mol[mask]
                T = T_bubble
            else:
                V_dew = self._V_err_at_T(T_dew, 0., gas_conversion, liquid_conversion)
                if V_dew < V:
                    F_mol = self._F_mol
                    mol = self._mol_vlle
                    if gas_conversion:
                        mol = mol + self._dmol_vlle
                        F_mol = F_mol + self._dF_mol
                    l = x_dew * F_mol * (1. - V)
                    mask = l > mol 
                    l[mask] = mol[mask]
                    v = mol - l
                    T = T_dew
                else:
                    T = flx.IQ_interpolation(
                        self._V_err_at_T,
                        T_bubble, T_dew, V_bubble - V, V_dew - V,
                        self._T, self.T_tol, self.V_tol,
                        (V, gas_conversion, liquid_conversion),
                        checkiter=False, checkbounds=False,
                        maxiter=self.maxiter,
                    )
                    v = self._v
                    mol = self._mol_vlle
                    if gas_conversion or liquid_conversion:
                        mol = mol + self._dmol_vlle
            self._T = thermal_condition.T = T
            set_flows(vapor_mol, liquid_mol, index, v, mol)
            if liquid_conversion or gas_conversion:
                try: self._H_hat = (
                        self.mixture.xH(self._phase_data, T, P) + (self.chemicals.Hf * mol).sum(0)
                    ) / self._F_mass
                except: pass
            else:
                try: self._H_hat = self.mixture.xH(self._phase_data, T, P) / self._F_mass
                except: pass
    
    def set_PS(self, P, S, gas_conversion=None, liquid_conversion=None):
        self._setup(gas_conversion, liquid_conversion)
        thermal_condition = self._thermal_condition
        thermal_condition.P = self._P = P
        if self._N == 0: 
            thermal_condition.T = self.mixture.xsolve_T_at_SP(
                self._phase_data, S, thermal_condition.T, P
            )
            return
        if self._N == 1: return self._set_PS_chemical(P, S)
        
        # Setup bounderies
        index = self._index
        mol = self._mol_vlle
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        
        # Check if subcooled liquid
        T_bubble, y_bubble = self._bubble_point.solve_Ty(self._z, P)
        if self._F_mol_light: T_bubble = 0.9 * T_bubble + 0.1 * self._bubble_point.Tmin
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        S_bubble = self.mixture.xS(self._phase_data, T_bubble, P)
        dS_bubble = S - S_bubble
        if dS_bubble <= 0:
            thermal_condition.T = self.mixture.xsolve_T_at_SP(self._phase_data, S, T_bubble, P)
            return
        
        # Check if super heated vapor
        T_dew, x_dew = self._dew_point.solve_Tx(self._z, P)
        if T_dew <= T_bubble: 
            T_dew, T_bubble = T_bubble, T_dew
            T_dew += 0.5
            T_bubble -= 0.5
        if self._F_mol_heavy: T_dew = 0.9 * T_dew + 0.1 * self._dew_point.Tmax
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        S_dew = self.mixture.xS(self._phase_data, T_dew, P)
        dS_dew = S - S_dew
        if dS_dew >= 0:
            thermal_condition.T = self.mixture.xsolve_T_at_SP(self._phase_data, S, T_dew, P)
            return
        
        # Guess T, overall vapor fraction, and vapor flow rates
        V = dS_bubble/(S_dew - S_bubble)
        self._refresh_v(V, y_bubble, x_dew)
        
        F_mass = self._F_mass
        S_hat = S/F_mass
        S_hat_bubble = self._S_hat_err_at_T(T_bubble, 0.)
        if (S_hat_bubble:=self._S_hat_err_at_T(T_bubble, 0.)) > S_hat:
            T = T_bubble
        elif (S_hat_dew:=self._S_hat_err_at_T(T_dew, 0.)) < S_hat:
            T = T_dew
        else:
            T = flx.IQ_interpolation(
                self._S_hat_err_at_T, T_bubble, T_dew, 
                S_hat_bubble - S_hat, S_hat_dew - S_hat,
                self._T, self.T_tol, self.S_hat_tol,
                (S_hat,), checkiter=False, checkbounds=False,
                maxiter=self.maxiter
            )
        # Make sure energy balance is correct by vaporizing a fraction
        # of the liquid or condensing a fraction of the vapor
        self._T = thermal_condition.T = T
        mol_liq = liquid_mol.copy()
        mol_liq[:] = 0.
        mol_liq[index] = liquid_mol[index]
        mol_gas= vapor_mol.copy()
        mol_gas[:] = 0.
        mol_gas[index] = vapor_mol[index]
        S_gas = self.mixture.S('g', mol_gas, T, P)
        S_liq = self.mixture.S('l', mol_liq, T, P)
        S_current = self.mixture.xS(self._phase_data, T, P)
        if S_current > S: # Condense a fraction of the vapor
            # Energy balance: S = f * S_condense + S_current
            S_condense = self.mixture.S('l', mol_gas, T, P) - S_gas
            try:
                f = (S - S_current) / S_condense
            except: # Floating point errors
                f = 0
            else:
                if f < 0.:
                    f = 0.
                elif f > 0.:
                    if f > 1.: f = 1.
                    condensed = f * mol_gas[index]
                    liquid_mol[index] += condensed
                    vapor_mol[index] -= condensed 
                else:
                    f = 0.
        else: # Vaporize a fraction of the liquid
            # Energy balance: S = f * S_vaporise + S_current
            S_vaporise = self.mixture.S('g', mol_liq, T, P) - S_liq
            try:
                f = (S - S_current) / S_vaporise
            except: # Floating point errors
                f = 0
            else:
                if f < 0.:
                    f = 0.
                elif f > 0.:
                    if f > 1.: f = 1.
                    vaporised = f * mol_liq[index]
                    vapor_mol[index] += vaporised
                    liquid_mol[index] -= vaporised
                else:
                    f = 0.
        if f == 0. or f == 1.:
            self._T = thermal_condition.T = self.mixture.xsolve_T_at_SP(
                self._phase_data, S, T, P
            )
        self._S_hat = S_hat
    
    def set_PH(self, P, H, gas_conversion=None, liquid_conversion=None):
        self._setup(gas_conversion, liquid_conversion)
        thermal_condition = self._thermal_condition
        thermal_condition.P = self._P = P
        if self._N == 0: 
            thermal_condition.T = self.mixture.xsolve_T_at_HP(
                self._phase_data, H, thermal_condition.T, P
            )
            return
        if self._N == 1: return self._set_PH_chemical(P, H)
        
        # Setup bounderies
        index = self._index
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        
        # Check if subcooled liquid
        if liquid_conversion:
            T_bubble, dz_bubble, y_bubble, x = self._bubble_point.solve_Ty(self._z, P, liquid_conversion)
            mol = self._mol_vlle + dz_bubble * self._F_mol_vlle
        else:
            T_bubble, y_bubble = self._bubble_point.solve_Ty(self._z, P)
            mol = self._mol_vlle
            dz_bubble = None
        if self._F_mol_light: T_bubble = 0.9 * T_bubble + 0.1 * self._bubble_point.Tmin
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = self.mixture.xH(self._phase_data, T_bubble, P)
        if liquid_conversion: 
            Hf = (self.chemicals.Hf * mol).sum()
            H_bubble += Hf
        dH_bubble = H - H_bubble
        if dH_bubble <= 0:
            if liquid_conversion: H -= Hf
            thermal_condition.T = self.mixture.xsolve_T_at_HP(self._phase_data, H, T_bubble, P)
            return
        
        # Check if super heated vapor
        if gas_conversion:
            T_dew, dz_dew, y, x_dew = self.dew_point.solve_Tx(self._z, P, gas_conversion)
            mol = self._mol_vlle + dz_dew
        else:
            T_dew, x_dew = self._dew_point.solve_Tx(self._z, P)
            mol = self._mol_vlle
            dz_dew = None
        if T_dew <= T_bubble: 
            T_dew, T_bubble = T_bubble, T_dew
            T_dew += 0.5
            T_bubble -= 0.5
        if self._F_mol_heavy: T_dew = 0.9 * T_dew + 0.1 * self._dew_point.Tmax
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = self.mixture.xH(self._phase_data, T_dew, P)
        if gas_conversion: 
            Hf = (self.chemicals.Hf * mol).sum()
            H_dew += Hf
        dH_dew = H - H_dew
        if dH_dew >= 0:
            if gas_conversion: H -= Hf
            thermal_condition.T = self.mixture.xsolve_T_at_HP(self._phase_data, H, T_dew, P)
            return
        
        # Guess T, overall vapor fraction, and vapor flow rates
        V = abs(dH_bubble / (H_dew - H_bubble))
        self._refresh_v(V, y_bubble, x_dew, dz_bubble, dz_dew)
        
        F_mass = self._F_mass
        H_hat = H/F_mass
        
        if (H_hat_bubble:=self._H_hat_err_at_T(T_bubble, 0., gas_conversion, liquid_conversion)) > H_hat:
            T = T_bubble
        elif (H_hat_dew:=self._H_hat_err_at_T(T_dew, 0., gas_conversion, liquid_conversion)) < H_hat:
            T = T_dew
        else:
            T = flx.IQ_interpolation(
                self._H_hat_err_at_T, T_bubble, T_dew, 
                H_hat_bubble - H_hat, H_hat_dew - H_hat,
                self._T, self.T_tol, self.H_hat_tol,
                (H_hat, gas_conversion, liquid_conversion),
                checkiter=False, checkbounds=False,
                maxiter=self.maxiter
            )
         
        if gas_conversion or liquid_conversion: 
            H -= (self.chemicals.Hf * (self._mol_vlle + self._dmol_vlle)).sum()
            
        # Make sure energy balance is correct by vaporizing a fraction
        # of the liquid or condensing a fraction of the vapor
        self._T = thermal_condition.T = T
        mol_liq = liquid_mol.copy()
        mol_liq[:] = 0.
        mol_liq[index] = liquid_mol[index]
        mol_gas= vapor_mol.copy()
        mol_gas[:] = 0.
        mol_gas[index] = vapor_mol[index]
        H_gas = self.mixture.H('g', mol_gas, T, P)
        H_liq = self.mixture.H('l', mol_liq, T, P)
        H_current = self.mixture.xH(self._phase_data, T, P)
        if H_current > H: # Condense a fraction of the vapor
            # Energy balance: H = f * H_condense + H_current
            H_condense = self.mixture.H('l', mol_gas, T, P) - H_gas
            try:
                f = (H - H_current) / H_condense
            except: # Floating point errors
                f = 0
            else:
                if f < 0.:
                    f = 0.
                elif f > 0.:
                    if f > 1.: f = 1.
                    condensed = f * mol_gas[index]
                    liquid_mol[index] += condensed
                    vapor_mol[index] -= condensed 
                else:
                    f = 0
        else: # Vaporize a fraction of the liquid
            # Energy balance: H = f * H_vaporise + H_current
            H_vaporise = self.mixture.H('g', mol_liq, T, P) - H_liq
            try:
                f = (H - H_current) / H_vaporise
            except: # Floating point errors
                f = 0
            else:
                if f < 0.:
                    f = 0.
                elif f > 0.:
                    if f > 1.: f = 1.
                    vaporised = f * mol_liq[index]
                    vapor_mol[index] += vaporised
                    liquid_mol[index] -= vaporised
                else:
                    f = 0
        if f == 0. or f == 1.:
            self._T = thermal_condition.T = self.mixture.xsolve_T_at_HP(
                self._phase_data, H, T, P
            )
        self._H_hat = H_hat
    
    def _estimate_v(self, V, y_bubble, x_dew, dz_bubble, dz_dew):
        F_mol_vlle = self._F_mol_vlle
        z = self._z_norm
        L = (1 - V)
        if dz_bubble is not None:
            F_mol_vlle = F_mol_vlle - dz_bubble * L * F_mol_vlle
        if dz_dew is not None:
            F_mol_vlle = F_mol_vlle - dz_dew * V * F_mol_vlle
        v_bubble = (V * z + L * y_bubble) * V * F_mol_vlle
        v_dew = (z - L * (L * z + V * x_dew)) * F_mol_vlle
        v = L * v_bubble + V * v_dew
        return v
    
    def _refresh_v(self, V, y_bubble, x_dew, dz_bubble=None, dz_dew=None): # TODO: use reaction data here for better estimate
        self._v = v = self._estimate_v(V, y_bubble, x_dew, dz_bubble, dz_dew)
        self._V = V
        self._y = fn.normalize(v, v.sum() + self._F_mol_light)
        l = self._mol_vlle - v
        l[l < 0] = 1e-12
        self._x = fn.normalize(l, l.sum() + self._F_mol_heavy)
    
    def _H_hat_err_at_T(self, T, H_hat, gas_conversion, liquid_conversion):
        v = self._solve_v(T, self._P, gas_conversion, liquid_conversion)
        if gas_conversion or liquid_conversion: 
            mol_vlle = self._mol_vlle + self._dmol_vlle
            set_flows(self._vapor_mol, self._liquid_mol, self._index, v, mol_vlle)
            H = self.mixture.xH(self._phase_data, T, self._P) + (self.chemicals.Hf * mol_vlle).sum()
        else:
            mol_vlle = self._mol_vlle 
            set_flows(self._vapor_mol, self._liquid_mol, self._index, v, mol_vlle)
            H = self.mixture.xH(self._phase_data, T, self._P)
        self._H_hat = H / self._F_mass
        return self._H_hat - H_hat
    
    def _H_hat_err_at_P(self, P, H_hat, gas_conversion=None, liquid_conversion=None):
        v = self._solve_v(self._T, P, gas_conversion, liquid_conversion)
        if gas_conversion or liquid_conversion: 
            mol_vlle = self._mol_vlle + self._dmol_vlle
            set_flows(self._vapor_mol, self._liquid_mol, self._index, v, mol_vlle)
            H = self.mixture.xH(self._phase_data, self._T, P) + (self.chemicals.Hf * mol_vlle).sum()
        else:
            mol_vlle = self._mol_vlle 
            set_flows(self._vapor_mol, self._liquid_mol, self._index, v, mol_vlle)
            H = self.mixture.xH(self._phase_data, self._T, P)
        self._H_hat = H / self._F_mass
        return self._H_hat - H_hat
    
    def _S_hat_err_at_T(self, T, S_hat):
        set_flows(self._vapor_mol, self._liquid_mol, self._index,
                  self._solve_v(T, self._P), self._mol_vlle)
        self._S_hat = self.mixture.xS(self._phase_data, T, self._P)/self._F_mass
        return self._S_hat - S_hat
    
    def _S_hat_err_at_P(self, P, S_hat):
        set_flows(self._vapor_mol, self._liquid_mol, self._index,
                  self._solve_v(self._T, P), self._mol_vlle)
        self._S_hat = self.mixture.xS(self._phase_data, self._T, P)/self._F_mass
        return self._S_hat - S_hat
    
    def _V_err_at_P(self, P, V, gas_conversion, liquid_conversion):
        v = self._solve_v(self._T , P, gas_conversion, liquid_conversion).sum()
        if gas_conversion or liquid_conversion:
            F_mol_vlle = self._F_mol_vlle + self._dF_mol
        else:
            F_mol_vlle = self._F_mol_vlle
        return v / F_mol_vlle - V
        
    def _V_err_at_T(self, T, V, gas_conversion, liquid_conversion):
        v = self._solve_v(T, self._P, gas_conversion, liquid_conversion).sum()
        if gas_conversion or liquid_conversion:
            F_mol_vlle = self._F_mol_vlle + self._dF_mol
        else:
            F_mol_vlle = self._F_mol_vlle
        return v / F_mol_vlle - V
    
    def _solve_v(self, T, P, gas_conversion=None, liquid_conversion=None):
        """Solve for vapor mol"""
        method = self.method
        if method == 'fixed-point':
            Psats = np.array([i(T) for i in
                              self._bubble_point.Psats])
            pcf_Psats_over_P = self._pcf(T, P, Psats) * Psats / P
            self._T = T
            y = self._solve_y(self._y, pcf_Psats_over_P, T, P, gas_conversion, liquid_conversion)
            if gas_conversion or liquid_conversion:
                self._v = v = (self._F_mol + self._dF_mol) * self._V * y
                mol_vlle = self._mol_vlle + self._dmol_vlle
            else:
                self._v = v = self._F_mol * self._V * y
                mol_vlle = self._mol_vlle
            mask = v > mol_vlle
            v[mask] = mol_vlle[mask]
            v[v < 0.] = 0.
        else:
            raise RuntimeError(f"invalid method '{method}'")
        return v
    
    def _setup(self, gas_conversion=None, liquid_conversion=None):
        imol = self._imol
        self._phase_data = tuple(imol)
        self._liquid_mol = imol['l']
        self._LIQUID_mol = imol['L']
        self._vapor_mol = imol['g']
        
        # Get overall composition
        vle = self.vle
        self._F_mass = vle._F_mass
        self._mol_vlle = vle._mol_vle

        # Set light and heavy keys
        self._F_mol_light = vle._F_mol_light
        self._F_mol_heavy = vle._F_mol_heavy
        self._F_mol_vlle = vle._F_mol_vle
        self._F_mol = vle._F_mol
        self._z = vle._z
        self._z_light = vle._z_light
        self._z_heavy = vle._z_heavy
        self._z_norm = vle._z_norm
        self._N = vle._N
        
        # Set equilibrium objects
        self._bubble_point = vle._bubble_point
        self._dew_point = vle._dew_point
        self._pcf = vle._pcf
        self._gamma = vle._gamma
        self._phi = vle._phi


class VLLECache(Cache): load = VLLE
del Cache, Equilibrium

