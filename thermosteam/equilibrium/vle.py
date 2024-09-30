# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
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
from ..utils import Cache
import thermosteam as tmo
import numpy as np

__all__ = ('VLE', 'VLECache')

@njit(cache=True)
def xy(x, Ks):
    x[x < 0] = 1e-16
    x /= x.sum()
    y = x * Ks
    y /= y.sum()
    return x, y

def xVlogK_iter_2n(xVlogK, pcf_Psat_over_P, T, P, z, f_gamma, gamma_args, f_phi, n,
                  gas_conversion, liquid_conversion):
    xVlogK = xVlogK.copy()
    x = xVlogK[:n]
    Ks = np.exp(xVlogK[n+1:])
    x, y = xy(x, Ks)
    Ks[:] = pcf_Psat_over_P * f_gamma(x, T, *gamma_args) / f_phi(y, T, P)
    if gas_conversion or liquid_conversion:
        V = xVlogK[n]
        if V < 0.: V = 0.
        elif V > 1.: V = 1.
        V = binary.compute_phase_fraction_2N(z, Ks)
        if gas_conversion: 
            z = z + gas_conversion(material=y * V, T=T, P=P, phase='g')
            z[z <= 0] = 1e-16
            z /= z.sum()
        if liquid_conversion:
            z = z + liquid_conversion(material=x * (1 - V), T=T, P=P, phase='l')
            z[z <= 0] = 1e-16
            z /= z.sum()
    Ks[Ks < 1e-16] = 1e-16
    xVlogK[n] = V = binary.compute_phase_fraction_2N(z, Ks)
    xVlogK[:n] = z/(1. + V * (Ks - 1.))
    xVlogK[n+1:] = np.log(Ks)
    return xVlogK

def xVlogK_iter(
        xVlogK, pcf_Psat_over_P, T, P,
        z, z_light, z_heavy, f_gamma, gamma_args,
        f_phi, n, gas_conversion, liquid_conversion
    ):
    xVlogK = xVlogK.copy()
    x = xVlogK[:n]
    Ks = np.exp(xVlogK[n+1:])
    x, y = xy(x, Ks)
    Ks[:] = pcf_Psat_over_P * f_gamma(x, T, *gamma_args) / f_phi(y, T, P)
    V = xVlogK[n]
    if V < 0.: V = 0.
    elif V > 1.: V = 1.
    if gas_conversion or liquid_conversion:
        if gas_conversion: 
            z = z + gas_conversion(material=y * V, T=T, P=P, phase='g')
            z[z <= 0] = 1e-16
            z /= z.sum()
        if liquid_conversion:
            z = z + liquid_conversion(material=x * (1 - V), T=T, P=P, phase='l')
            z[z <= 0] = 1e-16
            z /= z.sum()
    Ks[Ks < 1e-16] = 1e-16
    xVlogK[n] = V = binary.solve_phase_fraction_Rashford_Rice(z, Ks, V, z_light, z_heavy)
    xVlogK[:n] = z / (1. + V * (Ks - 1.))
    xVlogK[n+1:] = np.log(Ks)
    return xVlogK

def set_flows(vapor_mol, liquid_mol, index, vapor_data, total_data):
    vapor_mol[index] = vapor_data
    liquid_mol[index] = total_data - vapor_data

class VLE(Equilibrium, phases='lg'):
    """
    Create a VLE object that performs vapor-liquid equilibrium when called.
        
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
    First create a VLE object:
    
    >>> from thermosteam import indexer, equilibrium, settings
    >>> settings.set_thermo(['Water', 'Ethanol', 'Methanol', 'Propanol'], cache=True)
    >>> imol = indexer.MolarFlowIndexer(
    ...             l=[('Water', 304), ('Ethanol', 30)],
    ...             g=[('Methanol', 40), ('Propanol', 1)])
    >>> vle = equilibrium.VLE(imol)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Methanol', 40), ('Propanol', 1)],
            l=[('Water', 304), ('Ethanol', 30)]),
        thermal_condition=ThermalCondition(T=298.15, P=101325))
    
    Equilibrium given vapor fraction and pressure:
    
    >>> vle(V=0.5, P=101325)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 126.7), ('Ethanol', 26.38), ('Methanol', 33.49), ('Propanol', 0.8958)],
            l=[('Water', 177.3), ('Ethanol', 3.622), ('Methanol', 6.509), ('Propanol', 0.1042)]),
        thermal_condition=ThermalCondition(T=363.85, P=101325))
    
    Equilibrium given temperature and pressure:
    
    >>> vle(T=363.88, P=101325)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 127.4), ('Ethanol', 26.41), ('Methanol', 33.54), ('Propanol', 0.8968)],
            l=[('Water', 176.6), ('Ethanol', 3.59), ('Methanol', 6.456), ('Propanol', 0.1032)]),
        thermal_condition=ThermalCondition(T=363.88, P=101325))
    
    Equilibrium given enthalpy and pressure:
    
    >>> H = vle.thermo.mixture.xH(vle.imol, T=363.88, P=101325)
    >>> vle(H=H, P=101325)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 127.4), ('Ethanol', 26.41), ('Methanol', 33.54), ('Propanol', 0.8968)],
            l=[('Water', 176.6), ('Ethanol', 3.59), ('Methanol', 6.456), ('Propanol', 0.1032)]),
        thermal_condition=ThermalCondition(T=363.88, P=101325))
    
    Equilibrium given entropy and pressure:
    
    >>> S = vle.thermo.mixture.xS(vle.imol, T=363.88, P=101325)
    >>> vle(S=S, P=101325)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 127.4), ('Ethanol', 26.41), ('Methanol', 33.54), ('Propanol', 0.8968)],
            l=[('Water', 176.6), ('Ethanol', 3.59), ('Methanol', 6.456), ('Propanol', 0.1032)]),
        thermal_condition=ThermalCondition(T=363.88, P=101325))
    
    Equilibrium given vapor fraction and temperature:
    
    >>> vle(V=0.5, T=363.88)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 126.7), ('Ethanol', 26.38), ('Methanol', 33.49), ('Propanol', 0.8958)],
            l=[('Water', 177.3), ('Ethanol', 3.622), ('Methanol', 6.509), ('Propanol', 0.1042)]),
        thermal_condition=ThermalCondition(T=363.88, P=101431))
    
    Equilibrium given enthalpy and temperature:
    
    >>> vle(H=H, T=363.88)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 127.4), ('Ethanol', 26.41), ('Methanol', 33.54), ('Propanol', 0.8968)],
            l=[('Water', 176.6), ('Ethanol', 3.59), ('Methanol', 6.456), ('Propanol', 0.1032)]),
        thermal_condition=ThermalCondition(T=363.88, P=101325))
    
    Non-partitioning heavy and gaseous chemicals also affect VLE. Calculation 
    are repeated with non-partitioning chemicals:
        
    >>> from thermosteam import indexer, equilibrium, settings, Chemical
    >>> O2 = Chemical('O2', phase='g')
    >>> Glucose = Chemical('Glucose', phase='l', default=True)
    >>> settings.set_thermo(['Water', 'Ethanol', 'Methanol', 'Propanol', O2, Glucose], cache=True)
    >>> imol = indexer.MolarFlowIndexer(
    ...             l=[('Water', 304), ('Ethanol', 30), ('Glucose', 5)],
    ...             g=[('Methanol', 40), ('Propanol', 1), ('O2', 10)])
    >>> vle = equilibrium.VLE(imol)
    >>> vle(T=363.88, P=101325)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 159.5), ('Ethanol', 27.65), ('Methanol', 35.63), ('Propanol', 0.9337), ('O2', 10)],
            l=[('Water', 144.5), ('Ethanol', 2.352), ('Methanol', 4.369), ('Propanol', 0.0663), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=363.88, P=101325))
    
    >>> vle(V=0.5, P=101325)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 126.7), ('Ethanol', 26.38), ('Methanol', 33.52), ('Propanol', 0.8957), ('O2', 10)],
            l=[('Water', 177.3), ('Ethanol', 3.618), ('Methanol', 6.478), ('Propanol', 0.1043), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=362.47, P=101325))
    
    >>> H = vle.thermo.mixture.xH(vle.imol, T=363.88, P=101325)
    >>> vle(H=H, P=101325)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 127.4), ('Ethanol', 26.42), ('Methanol', 33.58), ('Propanol', 0.8968), ('O2', 10)],
            l=[('Water', 176.6), ('Ethanol', 3.583), ('Methanol', 6.421), ('Propanol', 0.1032), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=362.51, P=101325))
    
    >>> S = vle.thermo.mixture.xS(vle.imol, T=363.88, P=101325)
    >>> vle(S=S, P=101325)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 128.2), ('Ethanol', 26.45), ('Methanol', 33.63), ('Propanol', 0.8979), ('O2', 10)],
            l=[('Water', 175.8), ('Ethanol', 3.548), ('Methanol', 6.365), ('Propanol', 0.1021), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=362.54, P=101325))
    
    >>> vle(V=0.5, T=363.88)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 126.7), ('Ethanol', 26.38), ('Methanol', 33.49), ('Propanol', 0.8958), ('O2', 10)],
            l=[('Water', 177.3), ('Ethanol', 3.622), ('Methanol', 6.509), ('Propanol', 0.1042), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=363.88, P=106841))
    
    >>> vle(H=H, T=363.88)
    >>> vle 
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 126.7), ('Ethanol', 26.38), ('Methanol', 33.49), ('Propanol', 0.8958), ('O2', 10)],
            l=[('Water', 177.3), ('Ethanol', 3.622), ('Methanol', 6.51), ('Propanol', 0.1042), ('Glucose', 5)]),
        thermal_condition=ThermalCondition(T=363.88, P=106842))

    >>> vle(S=S, T=363.88)
    >>> vle 
    VLE(imol=MolarFlowIndexer(
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
    >>> vle = equilibrium.VLE(imol)
    >>> vle(T=300., P=101325)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 0.3617), ('O2', 10)],
            l=[('Water', 29.64)]),
        thermal_condition=ThermalCondition(T=300.00, P=101325))

    """
    __slots__ = (
        'method', # [str] Method for solving equilibrium.
        '_T', # [float] Temperature [K].
        '_P', # [float] Pressure [Pa].
        '_H_hat', # [float] Specific enthalpy [kJ/kg].
        '_S_hat', # [float] Specific entropy [kJ/K/kg].
        '_V', # [float] Molar vapor fraction.
        '_dew_point', # [DewPoint] Solves for dew point.
        '_bubble_point', # [BubblePoint] Solves for bubble point.
        '_y', # [1d array] Vapor composition.
        '_z_last', # tuple[1d array] Last bulk composition.
        '_phi', # [FugacityCoefficients] Estimates fugacity coefficients of gas.
        '_pcf', # [PoyintingCorrectionFactors] Estimates the PCF of a liquid.
        '_gamma', # [ActivityCoefficients] Estimates activity coefficients of a liquid.
        '_liquid_mol', # [1d array] Liquid molar data.
        '_vapor_mol', # [1d array] Vapor molar data.
        '_phase_data', # tuple[str, 1d array] Phase-data pairs.
        '_K', # [1d array] Partition coefficients.
        '_v', # [1d array] Molar vapor data in equilibrium.
        '_index', # [1d array] Index of chemicals in equilibrium.
        '_F_mass', # [float] Total mass data.
        '_chemical', # [Chemical] Single chemical in equilibrium.
        '_mol_vle', # [1d array] Moles of chemicals in VLE calculations.
        '_N', # [int] Number of chemicals in equilibrium.
        '_z', # [1d array] Molar composition of chemicals in equilibrium
        '_z_norm', # [1d array] Normalized molar composition of chemicals in equilibrium
        '_z_light', # [1d array] Molar composition of light chemicals not included in equilibrium calculation.
        '_z_heavy', # [1d array] Molar composition of heavy chemicals not included in equilibrium calculation.
        '_nonzero', # [1d array(bool)] Chemicals present in the mixture
        '_F_mol', # [float] Total moles of chemicals (accounting for dissociation).
        '_F_mol_vle', # [float] Total moles in equilibrium.
        '_F_mol_light', # [float] Total moles of gas chemicals not included in equilibrium calculation.
        '_F_mol_heavy', # [float] Total moles of heavy chemicals not included in equilibrium calculation.
        '_dmol_vle',
        '_dF_mol',
    )
    maxiter = 20
    T_tol = 5e-8
    P_tol = 1.
    H_hat_tol = 1e-6
    S_hat_tol = 1e-6
    V_tol = 1e-6
    K_tol = 1e-6
    y_tol = 1e-8
    default_method = 'fixed-point'
    
    def __init__(self, imol=None, thermal_condition=None,
                 thermo=None):
        self.method = self.default_method
        self._T = self._P = self._H_hat = self._V = 0
        super().__init__(imol, thermal_condition, thermo)
        self._z_last = None
        self._nonzero = None
        self._index = ()
    
    def __call__(self, *, T=None, P=None, V=None, H=None, S=None, x=None, y=None,
                 gas_conversion=None, liquid_conversion=None):
        """
        Perform vapor-liquid equilibrium.

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
        x : float, optional
            Molar composition of liquid (for binary mixtures).
        y : float, optional
            Molar composition of vapor (for binary mixtures).
        
        Notes
        -----
        You may only specify two of the following parameters: P, H, T, V, x, and y.
        Additionally, If x or y is specified, the other parameter must be either
        P or T (e.g., x and V is invalid).
        
        """
        if gas_conversion or liquid_conversion:
            if gas_conversion:
                if isinstance(gas_conversion, tmo.KineticReaction):
                    gas_conversion = gas_conversion.conversion_handle(tmo.Stream(thermo=self.thermo))
                elif isinstance(gas_conversion, (tmo.Reaction, tmo.ReactionItem)):
                    gas_conversion = gas_conversion.conversion_handle(None)
            if liquid_conversion:
                if isinstance(liquid_conversion, tmo.KineticReaction):
                    liquid_conversion = liquid_conversion.conversion_handle(tmo.Stream(thermo=self.thermo))
                elif isinstance(liquid_conversion, (tmo.Reaction, tmo.ReactionItem)):
                    liquid_conversion = liquid_conversion.conversion_handle(None)
            if x is not None or y is not None:
                raise ValueError(
                    "can not pass either 'x' or 'y' arguments with reactions preset"
                )
        
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
                    self.set_thermal_condition(T, P, gas_conversion, liquid_conversion)
                except NoEquilibrium:
                    thermal_condition = self._thermal_condition
                    thermal_condition.T = T
                    thermal_condition.P = P
            elif V_spec:
                try:
                    self.set_TV(T, V, gas_conversion, liquid_conversion)
                except NoEquilibrium:
                    thermal_condition = self._thermal_condition
                    thermal_condition.T = T
            elif H_spec:
                self.set_TH(T, H, gas_conversion, liquid_conversion)
            elif S_spec:
                self.set_TS(T, S, gas_conversion, liquid_conversion)
            elif x_spec:
                self.set_Tx(T, np.asarray(x))
            else: # y_spec
                self.set_Ty(T, np.asarray(y))
        elif P_spec:
            if V_spec:
                try:
                    self.set_PV(P, V, gas_conversion, liquid_conversion)
                except NoEquilibrium:
                    thermal_condition = self._thermal_condition
                    thermal_condition.P = P
            elif H_spec:
                try:
                    self.set_PH(P, H, gas_conversion, liquid_conversion)
                except NoEquilibrium:
                    thermal_condition = self._thermal_condition
                    thermal_condition.P = P
            elif S_spec:
                try:
                    self.set_PS(P, S, gas_conversion, liquid_conversion)
                except:
                    try:
                        self.set_PS(P, S, gas_conversion, liquid_conversion)
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
    
    ### Single component equilibrium case ###
        
    def _set_thermal_condition_chemical(self, T, P):
        chemical = self._chemical
        if T >= chemical.Tc: 
            self._liquid_mol[self._index] = 0
            self._vapor_mol[self._index] = self._mol_vle
        else:
            # Either liquid or gas
            Psat = chemical.Psat(T)
            tol = 1e-3
            if P < Psat - tol:
                self._liquid_mol[self._index] = 0
                self._vapor_mol[self._index] = self._mol_vle
            elif P > Psat + tol:
                self._liquid_mol[self._index] = self._mol_vle
                self._vapor_mol[self._index] = 0
    
    def _set_TV_chemical(self, T, V):
        # Set vapor fraction
        self._T = self._thermal_condition.T = self._chemical.Psat(T)
        self._vapor_mol[self._index] = V * self._mol_vle
        self._liquid_mol[self._index] = self._mol_vle - self._vapor_mol[self._index]
        
    def _set_PV_chemical(self, P, V):
        # Set vapor fraction
        self._T = self._thermal_condition.T = self._chemical.Tsat(P, check_validity=False)
        self._vapor_mol[self._index] = self._mol_vle * V
        self._liquid_mol[self._index] = self._mol_vle - self._vapor_mol[self._index]
        
    def _set_PH_chemical(self, P, H): 
        mol = self._mol_vle
        index = self._index
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        thermo = self._thermo
        phase_data = self._phase_data
        mixture = thermo.mixture
        chemical = self._chemical
        
        # Set temperature in equilibrium
        self._T = self._thermal_condition.T = T = chemical.Tsat(P, check_validity=False) 
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = mixture.xH(phase_data, T, P)
        if H >= H_dew:
            self._thermal_condition.T = mixture.xsolve_T_at_HP(phase_data, H, T, P)
            return

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = mixture.xH(phase_data, T, P)
        if H <= H_bubble:
            self._thermal_condition.T = mixture.xsolve_T_at_HP(phase_data, H, T, P)
            return
        
        # Adjust vapor fraction accordingly
        V = (H - H_bubble)/(H_dew - H_bubble)
        vapor_mol[index] = mol*V
        liquid_mol[index] = mol - vapor_mol[index]
        
    def _set_TH_chemical(self, T, H):
        index = self._index
        mol = self._mol_vle
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        mixture = self._thermo.mixture
        phase_data = self._phase_data
        
        # Set Pressure in equilibrium
        self._thermal_condition.P = P = self._chemical.Psat(T)
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = mixture.xH(phase_data, T, P)
        if H >= H_dew:
            raise NotImplementedError('cannot solve for pressure yet')
            self._thermal_condition.P = mixture.xsolve_P_at_HT(phase_data, H, T, P)
            return

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = mixture.xH(phase_data, T, P)
        if H <= H_bubble:
            raise NotImplementedError('cannot solve for pressure yet')
            self._thermal_condition.P = mixture.xsolve_P_at_HT(phase_data, H, T, P)
            return
        
        # Adjust vapor fraction accordingly
        V = (H - H_bubble)/(H_dew - H_bubble)
        vapor_mol[index] = mol*V
        liquid_mol[index] = mol - vapor_mol[index]
        
    def _set_PS_chemical(self, P, S): 
        mol = self._mol_vle
        index = self._index
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        thermo = self._thermo
        phase_data = self._phase_data
        mixture = thermo.mixture
        chemical = self._chemical
        
        # Set temperature in equilibrium
        self._T = self._thermal_condition.T = T = chemical.Tsat(P, check_validity=False)
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        S_dew = mixture.xS(phase_data, T, P)
        if S >= S_dew:
            self._thermal_condition.T = mixture.xsolve_T_at_SP(phase_data, S, T, P)
            return

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        S_bubble = mixture.xS(phase_data, T, P)
        if S <= S_bubble:
            self._thermal_condition.T = mixture.xsolve_T_at_SP(phase_data, S, T, P)
            return
        
        # Adjust vapor fraction accordingly
        V = (S - S_bubble)/(S_dew - S_bubble)
        vapor_mol[index] = mol*V
        liquid_mol[index] = mol - vapor_mol[index]
        
    def _set_TS_chemical(self, T, S):
        index = self._index
        mol = self._mol_vle
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        mixture = self._thermo.mixture
        phase_data = self._phase_data
        
        # Set Pressure in equilibrium
        self._thermal_condition.P = P = self._chemical.Psat(T)
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        S_dew = mixture.xS(phase_data, T, P)
        if S >= S_dew:
            raise NotImplementedError('cannot solve for pressure yet')
            self._thermal_condition.P = mixture.xsolve_P_at_ST(phase_data, S, T, P)
            return

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        S_bubble = mixture.xS(phase_data, T, P)
        if S <= S_bubble:
            raise NotImplementedError('cannot solve for pressure yet')
            self._thermal_condition.P = mixture.xsolve_P_at_ST(phase_data, S, T, P)
            return
        
        # Adjust vapor fraction accordingly
        V = (S - S_bubble)/(S_dew - S_bubble)
        vapor_mol[index] = mol*V
        liquid_mol[index] = mol - vapor_mol[index]
        
    def _lever_rule(self, x, y):
        split_frac = (self._z[0]-x[0])/(y[0]-x[0])
        if not -0.00001 < split_frac < 1.00001:
            raise InfeasibleRegion('phase composition')
        if split_frac > 1:
            split_frac = 1
        elif split_frac < 0:
            split_frac = 0
        self._vapor_mol[self._index] = v = self._F_mol * split_frac * y
        self._liquid_mol[self._index] = self._mol_vle - v
    
    def set_Tx(self, T, x):
        self._setup()
        assert self._N == 2, 'number of species in equilibrium must be 2 to specify x'
        self._thermal_condition.P, y = self._bubble_point.solve_Py(x, T)
        self._lever_rule(x, y)
    
    def set_Px(self, P, x):
        self._setup()
        assert self._N == 2, 'number of species in equilibrium must be 2 to specify x'
        self._thermal_condition.T, y = self._bubble_point.solve_Ty(x, P) 
        self._lever_rule(x, y)
        
    def set_Ty(self, T, y):
        self._setup()
        assert self._N == 2, 'number of species in equilibrium must be 2 to specify y'
        self._thermal_condition.P, x = self._dew_point.solve_Px(y, T)
        self._lever_rule(x, y)
    
    def set_Py(self, P, y):
        self._setup()
        assert self._N == 2, 'number of species in equilibrium must be 2 to specify y'
        self._thermal_condition.T, x = self._dew_point.solve_Tx(y, P) 
        self._lever_rule(x, y)
        
    def set_thermal_condition(self, T, P, gas_conversion=None, liquid_conversion=None):
        self._setup(gas_conversion, liquid_conversion)
        thermal_condition = self._thermal_condition
        self._T = thermal_condition.T = T
        self._P = thermal_condition.P = P
        if self._N == 0: return
        if self._N == 1: return self._set_thermal_condition_chemical(T, P)
        # Check if there is equilibrium
        if gas_conversion:
            P_dew, dz_dew, x_dew, _ = self._dew_point.solve_Px(self._z, T, gas_conversion)
            if P <= P_dew and not self._F_mol_heavy:
                self._vapor_mol[self._index] = self._mol_vle + dz_dew * self._F_mol_vle
                self._liquid_mol[self._index] = 0
                return
        else:
            P_dew, x_dew = self._dew_point.solve_Px(self._z, T)
            if P <= P_dew and not self._F_mol_heavy:
                self._vapor_mol[self._index] = self._mol_vle
                self._liquid_mol[self._index] = 0
                return
            dz_dew = None
        if liquid_conversion:
            P_bubble, dz_bubble, y_bubble, _ = self._bubble_point.solve_Py(self._z, T, liquid_conversion)
            if P >= P_bubble and not self._F_mol_light:
                self._vapor_mol[self._index] = 0
                self._liquid_mol[self._index] = self._mol_vle + dz_bubble * self._F_mol_vle
                return
        else:
            P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
            if P >= P_bubble and not self._F_mol_light:
                self._vapor_mol[self._index] = 0
                self._liquid_mol[self._index] = self._mol_vle
                return
            dz_bubble = None
        # Guess composition in the vapor is a
        # weighted average of bubble/dew points
        dP = (P_bubble - P_dew)
        V = (P - P_dew) / dP if dP > 1. else 0.5
        self._refresh_K(V, y_bubble, x_dew, dz_bubble, dz_dew)
        v = self._solve_v(T, P, gas_conversion, liquid_conversion)
        mol_vle = self._mol_vle + self._dmol_vle if (gas_conversion or liquid_conversion) else self._mol_vle 
        set_flows(self._vapor_mol, self._liquid_mol, self._index, v, mol_vle)
        
    def set_TV(self, T, V, gas_conversion=None, liquid_conversion=None):
        self._setup(gas_conversion, liquid_conversion)
        thermal_condition = self._thermal_condition
        thermal_condition.T = self._T = T
        if self._N == 0: raise RuntimeError('no chemicals present to perform VLE')
        if self._N == 1: return self._set_TV_chemical(T, V)
        if self._F_mol_heavy and V == 1.: V = 1. - 1e-3
        if self._F_mol_light and V == 0.: V = 1e-3
        if V == 1:
            if gas_conversion:
                P_dew, dz, y, x_dew = self._dew_point.solve_Px(self._z, T, gas_conversion)
                self._vapor_mol[self._index] = self._mol_vle + dz * self._F_mol_vle
            else:
                P_dew, x_dew = self._dew_point.solve_Px(self._z, T)
                self._vapor_mol[self._index] = self._mol_vle
            self._liquid_mol[self._index] = 0
            thermal_condition.P = P_dew
        elif V == 0:
            if liquid_conversion:
                P_bubble, dz, y_bubble, x = self._bubble_point.solve_Py(self._z, T)
                self._liquid_mol[self._index] = self._mol_vle + dz * self._F_mol_vle
            else:
                P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
                self._liquid_mol[self._index] = self._mol_vle
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
            self._refresh_K(V, y_bubble, x_dew, dz_bubble, dz_dew)
            if self._F_mol_light: P_bubble = 0.1 * self._bubble_point.Pmax + 0.9 * P_bubble
            if self._F_mol_heavy: P_dew = 0.1 * self._bubble_point.Pmin + 0.9 * P_dew
            
            V_bubble = self._V_err_at_P(P_bubble, 0., gas_conversion, liquid_conversion)
            if V_bubble > V:
                F_mol = self._F_mol
                mol = self._mol_vle
                if liquid_conversion:
                    mol = mol + self._dmol_vle
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
                    mol = self._mol_vle
                    if gas_conversion:
                        mol = mol + self._dmol_vle
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
                    mol = self._mol_vle
                    if gas_conversion or liquid_conversion:
                        mol = mol + self._dmol_vle
            
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
        if self._N == 0: raise RuntimeError('no chemicals present to perform VLE')
        if self._N == 1: return self._set_TH_chemical(T, H)
        self._T = T
        index = self._index
        mol = self._mol_vle
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
        self._refresh_K(V, y_bubble, x_dew)
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
        if self._N == 0: raise RuntimeError('no chemicals present to perform VLE')
        if self._N == 1: return self._set_TS_chemical(T, S)
        self._T = T
        index = self._index
        mol = self._mol_vle
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
        self._refresh_K(V, y_bubble, x_dew)
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
        if self._N == 0: raise RuntimeError('no chemicals present to perform VLE')
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
                self._vapor_mol[self._index] = self._mol_vle + dz * self._F_mol_vle
            else:
                T_dew, x_dew = self._dew_point.solve_Tx(self._z, P)
                self._vapor_mol[self._index] = self._mol_vle
            self._liquid_mol[self._index] = 0
            thermal_condition.T = T_dew
        elif V == 0 and not self._F_mol_light:
            if liquid_conversion:
                T_bubble, dz, y_bubble, x = self._bubble_point.solve_Ty(self._z, P, liquid_conversion)
                self._liquid_mol[self._index] = self._mol_vle + dz * self._F_mol_vle
            else:
                T_bubble, y_bubble = self._bubble_point.solve_Ty(self._z, P)
                self._liquid_mol[self._index] = self._mol_vle
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
            self._refresh_K(V, y_bubble, x_dew, dz_bubble, dz_dew)
            if self._F_mol_light: T_bubble = 0.1 * self._bubble_point.Tmin + 0.9 * T_bubble
            if self._F_mol_heavy: T_dew = 0.1 * self._bubble_point.Tmax + 0.9 * T_dew
            V_bubble = self._V_err_at_T(T_bubble, 0., gas_conversion, liquid_conversion)
            if V_bubble > V:
                F_mol = self._F_mol
                mol = self._mol_vle
                if liquid_conversion:
                    mol = mol + self._dmol_vle
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
                    mol = self._mol_vle
                    if gas_conversion:
                        mol = mol + self._dmol_vle
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
                    mol = self._mol_vle
                    if gas_conversion or liquid_conversion:
                        mol = mol + self._dmol_vle
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
        mol = self._mol_vle
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
        self._refresh_K(V, y_bubble, x_dew)
        
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
            mol = self._mol_vle + dz_bubble * self._F_mol_vle
        else:
            T_bubble, y_bubble = self._bubble_point.solve_Ty(self._z, P)
            mol = self._mol_vle
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
            mol = self._mol_vle + dz_dew
        else:
            T_dew, x_dew = self._dew_point.solve_Tx(self._z, P)
            mol = self._mol_vle
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
        self._refresh_K(V, y_bubble, x_dew, dz_bubble, dz_dew)
        
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
            H -= (self.chemicals.Hf * (self._mol_vle + self._dmol_vle)).sum()
            
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
    
    def _refresh_K(self, V, y_bubble, x_dew, dz_bubble=None, dz_dew=None): # TODO: use reaction data here for better estimate
        F_mol_vle = self._F_mol_vle
        z = self._z_norm
        L = (1 - V)
        if dz_bubble is not None:
            F_mol_vle = F_mol_vle - dz_bubble * L * F_mol_vle
        if dz_dew is not None:
            F_mol_vle = F_mol_vle - dz_dew * V * F_mol_vle
        v_bubble = (V * z + L * y_bubble) * V * F_mol_vle
        v_dew = (z - L * (L * z + V * x_dew)) * F_mol_vle
        v = L * v_bubble + V * v_dew
        l = self._mol_vle - v
        y = v / v.sum()
        l_net = l.sum()
        x = l / l_net if l_net > 0 else l * 0
        x[x < 1e-32] = 1e-32
        self._K = y / x
        self._V = V
    
    def _H_hat_err_at_T(self, T, H_hat, gas_conversion, liquid_conversion):
        v = self._solve_v(T, self._P, gas_conversion, liquid_conversion)
        if gas_conversion or liquid_conversion: 
            mol_vle = self._mol_vle + self._dmol_vle
            set_flows(self._vapor_mol, self._liquid_mol, self._index, v, mol_vle)
            H = self.mixture.xH(self._phase_data, T, self._P) + (self.chemicals.Hf * mol_vle).sum()
        else:
            mol_vle = self._mol_vle 
            set_flows(self._vapor_mol, self._liquid_mol, self._index, v, mol_vle)
            H = self.mixture.xH(self._phase_data, T, self._P)
        self._H_hat = H / self._F_mass
        return self._H_hat - H_hat
    
    def _H_hat_err_at_P(self, P, H_hat, gas_conversion=None, liquid_conversion=None):
        v = self._solve_v(self._T, P, gas_conversion, liquid_conversion)
        if gas_conversion or liquid_conversion: 
            mol_vle = self._mol_vle + self._dmol_vle
            set_flows(self._vapor_mol, self._liquid_mol, self._index, v, mol_vle)
            H = self.mixture.xH(self._phase_data, self._T, P) + (self.chemicals.Hf * mol_vle).sum()
        else:
            mol_vle = self._mol_vle 
            set_flows(self._vapor_mol, self._liquid_mol, self._index, v, mol_vle)
            H = self.mixture.xH(self._phase_data, self._T, P)
        self._H_hat = H / self._F_mass
        return self._H_hat - H_hat
    
    def _S_hat_err_at_T(self, T, S_hat):
        set_flows(self._vapor_mol, self._liquid_mol, self._index,
                  self._solve_v(T, self._P), self._mol_vle)
        self._S_hat = self.mixture.xS(self._phase_data, T, self._P)/self._F_mass
        return self._S_hat - S_hat
    
    def _S_hat_err_at_P(self, P, S_hat):
        set_flows(self._vapor_mol, self._liquid_mol, self._index,
                  self._solve_v(self._T, P), self._mol_vle)
        self._S_hat = self.mixture.xS(self._phase_data, self._T, P)/self._F_mass
        return self._S_hat - S_hat
    
    def _V_err_at_P(self, P, V, gas_conversion, liquid_conversion):
        v = self._solve_v(self._T , P, gas_conversion, liquid_conversion).sum()
        if gas_conversion or liquid_conversion:
            F_mol_vle = self._F_mol_vle + self._dF_mol
        else:
            F_mol_vle = self._F_mol_vle
        return v / F_mol_vle - V
        
    def _V_err_at_T(self, T, V, gas_conversion, liquid_conversion):
        v = self._solve_v(T, self._P, gas_conversion, liquid_conversion).sum()
        if gas_conversion or liquid_conversion:
            F_mol_vle = self._F_mol_vle + self._dF_mol
        else:
            F_mol_vle = self._F_mol_vle
        return v / F_mol_vle - V
    
    def _solve_v(self, T, P, gas_conversion=None, liquid_conversion=None):
        """Solve for vapor mol"""
        method = self.method
        if method == 'shgo':
            if gas_conversion or liquid_conversion:
                raise RuntimeError('shgo is not a valid method (yet) when reactions are present')
            gamma = self._gamma
            phi = self._phi
            Psats = np.array([i(T) for i in self._bubble_point.Psats]) 
            pcf = self._pcf(T, P, Psats)
            F_mol_vle = self._F_mol_vle
            mol_vle = self._mol_vle
            z = mol_vle / F_mol_vle
            self._v = v = F_mol_vle * solve_vle_vapor_mol_shgo(
                z, T, gamma.f, gamma.args, P, pcf * Psats, phi.f, phi.args, 
                dict(f_tol=self.y_tol, minimizer_kwargs=dict(f_tol=self.y_tol)),
            )
            self._z_last = z
        elif method == 'fixed-point':
            Psats = np.array([i(T) for i in
                              self._bubble_point.Psats])
            pcf_Psats_over_P = self._pcf(T, P, Psats) * Psats / P
            self._T = T
            self._v = v = self._solve_v_fixed_point(pcf_Psats_over_P, T, P, gas_conversion, liquid_conversion)
            if gas_conversion or liquid_conversion:
                mol_vle = self._mol_vle + self._dmol_vle
            else:
                mol_vle = self._mol_vle
            mask = v > mol_vle
            v[mask] = mol_vle[mask]
            v[v < 0.] = 0.
        else:
            raise RuntimeError(f"invalid method '{method}'")
        return v
    
    def _solve_v_fixed_point(self, pcf_Psat_over_P, T, P, gas_conversion, liquid_conversion):
        gamma = self._gamma
        z = self._z
        K = self._K # pcf_Psat_over_P
        n = z.size
        if n > 2 or self._z_light or self._z_heavy:
            f = xVlogK_iter
            args = (pcf_Psat_over_P, T, P, z, 
                    self._z_light, self._z_heavy, 
                    gamma.f, gamma.args, self._phi, n,
                    gas_conversion, liquid_conversion)
        else:
            f = xVlogK_iter_2n
            args = (pcf_Psat_over_P, T, P, z, 
                    gamma.f, gamma.args, self._phi, n,
                    gas_conversion, liquid_conversion)
        xVlogK = np.zeros(2 * n + 1)
        xVlogK[n] = V = self._V
        K[K < 1e-16] = 1e-16
        xVlogK[:n] = z/(1. + V * (K - 1.))
        xVlogK[n+1:] = np.log(K)
        xVlogK = flx.aitken(
            f, xVlogK, self.K_tol, args, checkiter=False, 
            checkconvergence=False, convergenceiter=5,
            maxiter=self.maxiter
        )
        self._V = V = xVlogK[n]
        self._K = K = np.exp(xVlogK[n+1:])
        x = xVlogK[:n]
        x, y = xy(x, K)
        self._z_last = z
        if gas_conversion or liquid_conversion:
            dmol_vle = 0
            if gas_conversion: 
                dmol_vle += gas_conversion(material=y * V, T=T, P=P, phase='g')
            if liquid_conversion:
                dmol_vle += liquid_conversion(material=x * (1 - V), T=T, P=P, phase='l')
            dmol_vle *= self._F_mol
            self._dmol_vle = dmol_vle
            self._dF_mol = dmol_vle.sum()
            v = (self._F_mol - self._dF_mol) * V * x * K
        else:
            v = self._F_mol * V * x * K
        return v
    
    def _setup(self, gas_conversion=None, liquid_conversion=None):
        imol = self._imol
        self._phase_data = tuple(imol)
        self._liquid_mol = liquid_mol = imol['l']
        self._vapor_mol = vapor_mol = imol['g']
        mol = liquid_mol + vapor_mol
        if not mol.any(): raise NoEquilibrium('no chemicals to perform equilibrium')
        nonzero = set(mol.nonzero_keys())
        if gas_conversion:
            nonzero.update(
                gas_conversion.reaction.stoichiometry.nonzero_keys()
            )
        if liquid_conversion:
            nonzero.update(
                liquid_conversion.reaction.stoichiometry.nonzero_keys()
            )
        chemicals = self.chemicals
        if self._nonzero == nonzero:
            index = self._index
            eq_chems = chemicals.tuple
            eq_chems = [eq_chems[i] for i in index]
            reset = False
        else:
            # Set up indices for both equilibrium and non-equilibrium species
            index = chemicals.get_vle_indices(nonzero)
            eq_chems = chemicals.tuple
            eq_chems = [eq_chems[i] for i in index]
            reset = True     
            self._nonzero = nonzero
            self._index = index
        if gas_conversion and [i.ID for i in eq_chems] != [i.ID for i in gas_conversion.reaction.chemicals]:
            gas_conversion.set_chemicals(eq_chems)
        if liquid_conversion and [i.ID for i in eq_chems] != [i.ID for i in liquid_conversion.reaction.chemicals]:
            liquid_conversion.set_chemicals(eq_chems)
        
        # Get overall composition
        self._F_mass = (chemicals.MW * mol).sum()
        self._mol_vle = mol_vle = mol[index]

        # Set light and heavy keys
        LNK_index = chemicals._light_indices
        HNK_index = chemicals._heavy_indices
        vapor_mol[HNK_index] = 0
        vapor_mol[LNK_index] = light_mol = mol[LNK_index]
        liquid_mol[LNK_index] = 0
        liquid_mol[HNK_index] = heavy_mol = mol[HNK_index]
        if not mol_vle.any(): raise NoEquilibrium('no chemicals to perform equilibrium')
        self._F_mol_light = F_mol_light = light_mol.sum()
        self._F_mol_heavy = F_mol_heavy = (heavy_mol * chemicals._heavy_solutes).sum()
        self._F_mol_vle = F_mol_vle = mol_vle.sum()
        self._F_mol = F_mol = F_mol_vle + F_mol_light + F_mol_heavy
        self._z = mol_vle / F_mol
        self._z_light = z_light = F_mol_light / F_mol
        self._z_heavy = z_heavy = F_mol_heavy / F_mol
        self._z_norm = mol_vle / F_mol_vle
        N = len(index)
        if N:
            N += z_light > 0.
            N += z_heavy > 0.
        self._N = N
        if reset:
            if N == 0:
                self._phi = self._gamma = self._pcf = self._dew_point = self._bubble_point = None
            elif N == 1:
                self._chemical, = eq_chems
            else:
                # Set equilibrium objects
                thermo = self._thermo
                self._bubble_point = bp = BubblePoint(eq_chems, thermo)
                self._dew_point = DewPoint(eq_chems, thermo)
                self._pcf = bp.pcf
                self._gamma = bp.gamma
                self._phi = bp.phi


class VLECache(Cache): load = VLE
del Cache, Equilibrium

@njit(cache=True)
def liquid_fugacity(mol_L, T, pcf_Psats, f_gamma, gamma_args):
    total_mol_L = mol_L.sum()
    if total_mol_L:
        x = mol_L / total_mol_L
        fugacity = x * f_gamma(x, T, *gamma_args) * pcf_Psats
    else:
        fugacity = np.ones_like(mol_L)
    return fugacity 

@njit(cache=True)
def vapor_fugacity(mol_v, T, P, f_phi, phi_args):
    total_mol_v = mol_v.sum()
    if total_mol_v:
        y = mol_v / total_mol_v
        fugacity = y * P * f_phi(y, T, P, *phi_args)
    else:
        fugacity = np.ones_like(mol_v)
    return fugacity

@njit(cache=True)
def gibbs_free_energy(mol, fugacity):
    fugacity[fugacity <= 0] = 1
    g_mix = (mol * np.log(fugacity)).sum()
    return g_mix

@njit(cache=True)
def vle_objective_function(mol_v, mol, T, f_gamma, gamma_args, P, pcf_Psats, f_phi, phi_args):
    mol_l = mol - mol_v
    g_mix_l = gibbs_free_energy(mol_l, liquid_fugacity(mol_l, T, pcf_Psats, f_gamma, gamma_args))
    g_mix_g = gibbs_free_energy(mol_v, vapor_fugacity(mol_v, T, P, f_phi, phi_args))
    g_mix = g_mix_l + g_mix_g
    return g_mix

def solve_vle_vapor_mol_shgo(
        mol, T, f_gamma, gamma_args, P, pcf_Psats, f_phi, phi_args, shgo_options,
    ):
    args = (mol, T, f_gamma, gamma_args, P, pcf_Psats, f_phi, phi_args)
    bounds = np.zeros([mol.size, 2])
    bounds[:, 1] = mol
    result = shgo(vle_objective_function, bounds, args, options=shgo_options)
    return result.x
