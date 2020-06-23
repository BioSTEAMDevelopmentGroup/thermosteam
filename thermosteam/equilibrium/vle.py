# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import flexsolve as flx
from ..exceptions import InfeasibleRegion
from ..utils.decorator_utils import thermo_user
from . import binary_phase_fraction as binary
from .dew_point import DewPointCache
from .bubble_point import BubblePointCache
from .fugacity_coefficients import IdealFugacityCoefficients
from .._thermal_condition import ThermalCondition
from .. import functional as fn
from ..utils import Cache
import numpy as np

__all__ = ('VLE', 'VLECache')

@thermo_user
class VLE:
    """
    Create a VLE object that performs vapor-liquid equilibrium when called.
        
    Parameters
    ----------
    imol : MaterialIndexer
        Chemical phase data is stored here.
    thermal_condition=None : ThermalCondition, optional
        Temperature and pressure results are stored here.
    thermo=None : Thermo, optional
        Themodynamic property package for equilibrium calculations.
        Defaults to `thermosteam.settings.get_thermo()`.
    bubble_point_cache=None : thermosteam.utils.Cache, optional
        Cache to retrieve bubble point object.
    dew_point_cache=None : thermosteam.utils.Cache, optional
        Cache to retrieve dew point object
    
    Examples
    --------
    >>> from thermosteam import indexer, equilibrium, settings
    >>> settings.set_thermo(['Water', 'Ethanol', 'Methanol', 'Propanol'])
    >>> imol = indexer.MolarFlowIndexer(
    ...             l=[('Water', 304), ('Ethanol', 30)],
    ...             g=[('Methanol', 40), ('Propanol', 1)])
    >>> vle = equilibrium.VLE(imol)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Methanol', 40), ('Propanol', 1)],
            l=[('Water', 304), ('Ethanol', 30)]),
        thermal_condition=ThermalCondition(T=298.15, P=101325))
    >>> vle(V=0.5, P=101325)
    >>> vle
    VLE(imol=MolarFlowIndexer(
            g=[('Water', 126.7), ('Ethanol', 26.4), ('Methanol', 33.49), ('Propanol', 0.896)],
            l=[('Water', 177.3), ('Ethanol', 3.598), ('Methanol', 6.509), ('Propanol', 0.104)]),
        thermal_condition=ThermalCondition(T=363.88, P=101325))
    
    """
    __slots__ = ('_T', # [float] Temperature [K].
                 '_P', # [float] Pressure [Pa].
                 '_H_hat', # [float] Specific enthalpy [kJ/kg].
                 '_V', # [float] Molar vapor fraction.
                 '_thermo', # [float] Thermo object for estimating mixture properties.
                 '_thermal_condition', # [ThermalCondition] T and P results are stored here.
                 '_y', # [1d array] Molar vapor composition in equilibrium.
                 '_dew_point', # [DewPoint] Solves for dew point.
                 '_bubble_point', # [BubblePoint] Solves for bubble point.
                 '_x', # [1d array] Liquid composition.
                 '_phi', # [FugacityCoefficients] Estimates fugacity coefficients of gas.
                 '_pcf', # [PoyintingCorrectionFactors] Estimates the PCF of a liquid.
                 '_gamma', # [ActivityCoefficients] Estimates activity coefficients of a liquid.
                 '_imol', # [MaterialIndexer] Stores vapor and liquid molar data.
                 '_liquid_mol', # [1d array] Liquid molar data.
                 '_vapor_mol', # [1d array] Vapor molar data.
                 '_phase_data', # tuple[str, 1d array] Phase-data pairs.
                 '_v',  # [1d array] Molar vapor data in equilibrium.
                 '_index', # [1d array] Index of chemicals in equilibrium.
                 '_F_mass', # [float] Total mass data.
                 '_chemical', # [Chemical] Single chemical in equilibrium.
                 '_mol', # [1d array] VLE chemicals in equilibrium.
                 '_N', # [int] Number of chemicals in equilibrium.
                 '_z', # [1d array] Molar composition of chemicals in equilibrium
                 '_Ks', # [1d array] Partition coefficients.
                 '_nonzero', # [1d array(bool)] Chemicals present in the mixture
                 '_F_mol_vle', # [float] Total moles in equilibrium.
                 '_dew_point_cache', # [Cache] Retrieves the DewPoint object if arguments are the same.
                 '_bubble_point_cache') # [Cache] Retrieves the BubblePoint object if arguments are the same.
    T_tol = 1e-9
    P_tol = 1e-3
    H_hat_tol = 1e-3
    V_tol = 1e-9
    
    def __init__(self, imol, thermal_condition=None,
                 thermo=None, bubble_point_cache=None, dew_point_cache=None):
        self._T = self._P = self._H_hat = self._V = 0
        self._dew_point_cache = dew_point_cache or DewPointCache()
        self._bubble_point_cache = bubble_point_cache or BubblePointCache()
        self._load_thermo(thermo)
        self._imol = imol
        self._thermal_condition = thermal_condition or ThermalCondition(298.15, 101325.)
        self._phase_data = tuple(imol)
        self._liquid_mol = liquid_mol = imol['l']
        self._vapor_mol = imol['g']
        self._nonzero = np.zeros(liquid_mol.shape, dtype=bool)
        self._index = ()
        self._y = None
    
    def __call__(self, P=None, H=None, T=None, V=None, x=None, y=None):
        """
        Perform vapor-liquid equilibrium.

        Parameters
        ----------
        P=None : float
            Operating pressure [Pa].
        H=None : float
            Enthalpy [kJ/hr].
        T=None : float
            Operating temperature [K].
        V=None : float
            Molar vapor fraction.
        x=None : float
            Molar composition of liquid (for binary mixtures).
        y=None : float
            Molar composition of vapor (for binary mixtures).
        
        Notes
        -----
        You may only specify two of the following parameters: P, H, T, V, x, and y.
        Additionally, If x or y is specified, the other parameter must be either
        P or T (e.g., x and V is invalid).
        
        """
        ### Decide what kind of equilibrium to run ###
        T_spec = T is not None
        P_spec = P is not None
        V_spec = V is not None
        H_spec = H is not None
        x_spec = x is not None
        y_spec = y is not None
        N_specs = (P_spec + H_spec + T_spec + V_spec + x_spec + y_spec)
        assert N_specs == 2, ("must pass two and only two of the following "
                              "specifications: T, P, V, H, x, y")
        
        # Run equilibrium
        if T_spec:
            if P_spec:
                return self.set_thermal_condition(T, P)
            elif V_spec:
                return self.set_TV(T, V)
            elif H_spec:
                return self.set_TH(T, H)
            elif x_spec:
                return self.set_Tx(T, np.asarray(x))
            else: # y_spec
                return self.set_Ty(T, np.asarray(y))
        elif P_spec:
            if V_spec:
                return self.set_PV(P, V)
            elif H_spec:
                return self.set_PH(P, H)
            elif x_spec:
                return self.set_Px(P, np.asarray(x))
            else: # y_spec
                return self.set_Py(P, np.asarray(y))
        elif H_spec:
            if y_spec:
                raise NotImplementedError('specification H and y is invalid')
            elif x_spec:
                raise NotImplementedError('specification H and x is invalid')
            else: # V_spec
                raise NotImplementedError('specification V and H not implemented')
        elif V_spec:
            if y_spec:
                raise ValueError("specification V and y is invalid")
            else: # x_spec
                raise ValueError("specification V and x is invalid")
        else: # x_spec and y_spec
            raise ValueError("can only pass either 'x' or 'y' arguments, not both")
    
    def _setup(self):
        # Get flow rates
        liquid_mol = self._liquid_mol
        vapor_mol = self._vapor_mol
        mol = liquid_mol + vapor_mol
        nonzero = mol > 0
        chemicals = self.chemicals
        if (self._nonzero == nonzero).all():
            index = self._index
        else:
            # Set up indices for both equilibrium and non-equilibrium species
            index = chemicals.get_vle_indices(nonzero)
            self._y = None            
            self._N = N = len(index)
            eq_chems = chemicals.tuple
            eq_chems = [eq_chems[i] for i in index]
            self._nonzero = nonzero
            self._index = index
            if N == 1:
                self._chemical, = eq_chems
            else:
                # Set equilibrium objects
                thermo = self._thermo
                self._bubble_point = bp = self._bubble_point_cache(eq_chems, thermo)
                self._dew_point = self._dew_point_cache(eq_chems, thermo)
                self._pcf = bp.pcf
                self._gamma = bp.gamma
                self._phi = bp.phi
        
        # Get overall composition
        self._F_mass = (chemicals.MW * mol).sum()
        F_mol = mol.sum()
        assert F_mol != 0, 'empty stream cannot perform equilibrium'
        self._mol = mol[index]

        # Set light and heavy keys
        LNK_index = chemicals._light_indices
        HNK_index = chemicals._heavy_indices
        vapor_mol[HNK_index] = 0
        vapor_mol[LNK_index] = light_mol = mol[LNK_index]
        liquid_mol[LNK_index] = 0
        liquid_mol[HNK_index] = heavy_mol = mol[HNK_index]
        F_mol_light = light_mol.sum()
        F_mol_heavy = heavy_mol.sum()
        self._F_mol_vle = F_mol_vle = F_mol - F_mol_light - F_mol_heavy
        self._z = self._mol / F_mol_vle

    @property
    def imol(self):
        return self._imol
    @property
    def thermal_condition(self):
        return self._thermal_condition

    ### Single component equilibrium case ###
        
    def _set_thermal_condition_chemical(self, T, P):
        # Either liquid or gas
        if P < self._chemical.Psat(T):
            self._liquid_mol[self._index] = 0
            self._vapor_mol[self._index] = self._mol
        else:
            self._liquid_mol[self._index] = self._mol
            self._vapor_mol[self._index] = 0
    
    def _set_TV_chemical(self, T, V):
        # Set vapor fraction
        self._T = self._thermal_condition.T = self._chemical.Psat(T)
        self._vapor_mol[self._index] = self._mol*V
        self._liquid_mol[self._index] = self._mol - self._vapor_mol[self._index]
        
    def _set_PV_chemical(self, P, V):
        # Set vapor fraction
        self._T = self._thermal_condition.T = self._chemical.Tsat(P)
        self._vapor_mol[self._index] = self._mol*V
        self._liquid_mol[self._index] = self._mol - self._vapor_mol[self._index]
        
    def _set_PH_chemical(self, P, H): 
        mol = self._mol
        index = self._index
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        thermo = self._thermo
        phase_data = self._phase_data
        mixture = thermo.mixture
        
        # Set temperature in equilibrium
        self._T = self._thermal_condition.T = T = self._chemical.Tsat(P)
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = mixture.xH(phase_data, T, P)
        if H >= H_dew:
            self._thermal_condition.T = mixture.xsolve_T(phase_data, H, T, P)
            return

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = mixture.xH(phase_data, T, P)
        if H <= H_bubble:
            self._thermal_condition.T = mixture.xsolve_T(phase_data, H, T, P)
            return
        
        # Adjust vapor fraction accordingly
        V = (H - H_bubble)/(H_dew - H_bubble)
        vapor_mol[index] = mol*V
        liquid_mol[index] = mol - vapor_mol[index]
        
    def _set_TH_chemical(self, T, H):
        index = self._index
        mol = self._mol
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        thermo = self._thermo
        phase_data = self._phase_data
        
        # Set Pressure in equilibrium
        self._thermal_condition.P = P = self._chemical.Psat(T)
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = thermo.mixture.xH(phase_data, T, P)
        if H >= H_dew:
            self._thermal_condition.T = thermo.xsolve_T(phase_data, H, T, P)
            return

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = thermo.mixture.xH(phase_data, T, P)
        if H <= H_bubble:
            self._thermal_condition.T = thermo.xsolve_T(phase_data, H, T, P)
            return
        
        # Adjust vapor fraction accordingly
        V = (H - H_bubble)/(H_dew - H_bubble)
        vapor_mol[index] = mol*V
        liquid_mol[index] = mol - vapor_mol[index]
        
    def _lever_rule(self, x, y):
        split_frac = (self._z[0]-x[0])/(y[0]-x[0])
        if not -0.00001 < split_frac < 1.00001:
            InfeasibleRegion('phase composition')
        if split_frac > 1:
            split_frac = 1
        elif split_frac < 0:
            split_frac = 0
        self._vapor_mol[self._index] = v = self._F_mol_vle * split_frac * y
        self._liquid_mol[self._index] = self._mol - v
    
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
        
    def set_thermal_condition(self, T, P):
        self._setup()
        thermal_condition = self._thermal_condition
        self._T = thermal_condition.T = T
        self._P = thermal_condition.P = P
        if self._N == 0: return
        if self._N == 1: return self._set_thermal_condition_chemical(T, P)
        # Check if there is equilibrium
        P_dew, x_dew = self._dew_point.solve_Px(self._z, T)
        if P <= P_dew:
            self._vapor_mol[self._index] = self._mol
            self._liquid_mol[self._index] = 0
            return
        P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
        if P >= P_bubble:
            self._vapor_mol[self._index] = 0
            self._liquid_mol[self._index] = self._mol
            return
        # Guess composition in the vapor is a
        # weighted average of bubble/dew points
        V = self._V or (T - P_dew)/(P_bubble - P_dew)
        self._refresh_v(V, y_bubble)
        # Solve
        try:
            v = self._solve_v(T, P)
        except:
            self._v = self._estimate_v(V, y_bubble)
            v = self._solve_v(T, P)
        self._vapor_mol[self._index] = v
        self._liquid_mol[self._index] = self._mol - v
        self._H_hat = self.mixture.xH(self._phase_data, T, P)/self._F_mass
        
    def set_TV(self, T, V):
        self._setup()
        mol = self._mol
        thermal_condition = self._thermal_condition
        thermal_condition.T = self._T = T
        if self._N == 0: raise RuntimeError('no chemicals present to perform VLE')
        if self._N == 1: return self._set_TV_chemical(T, V)
        if V == 1:
            P_dew, x_dew = self._dew_point.solve_Px(self._z, T)
            self._vapor_mol[self._index] = self._mol
            self._liquid_mol[self._index] = 0
            thermal_condition.P = P_dew
        elif V == 0:
            P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
            self._vapor_mol[self._index] = 0
            self._liquid_mol[self._index] = self._mol
            thermal_condition.P = P_bubble
        else:
            P_dew, x_dew = self._dew_point.solve_Px(self._z, T)
            P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
            self._V = V 
            self._refresh_v(V, y_bubble)
            P = flx.IQ_interpolation(self._V_err_at_P,
                                     P_bubble, P_dew, 0 - V, 1 - V,
                                     self._P, self.P_tol, self.V_tol,
                                     (self._V,), checkroot=False,
                                     checkbounds=False)
             
            # In case solver cannot reach the vapor fraction because
            # the composition does not converge at values that meets the
            # vapor fraction.
            V_offset = V - self._V
            if V_offset < -1e-2:
                F_mol_vapor = self._F_mol_vle * V
                v = y_bubble * F_mol_vapor
                mask = v > mol
                v[mask] = mol[mask]
                P = P_bubble
            elif V_offset > 1e2:
                v = x_dew * self._F_mol_vle * V
                mask = v < mol 
                v[mask] = mol[mask]
                P = P_dew
            else:
                v = self._v
            
            self._P = thermal_condition.P = P
            self._vapor_mol[self._index] = v
            self._liquid_mol[self._index] = mol - v
            self._H_hat = self.mixture.xH(self._phase_data, T, P) / self._F_mass

    def set_TH(self, T, H):
        self._setup()
        if self._N == 0: raise RuntimeError('no chemicals present to perform VLE')
        if self._N == 1: return self._set_TH_chemical(T, H)
        self._T = T
        index = self._index
        mol = self._mol
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        phase_data = self._phase_data
        
        # Check if super heated vapor
        P_dew, x_dew = self._dew_point.solve_Px(self._z, T)
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = self.mixture.xH(phase_data, T, P_dew)
        dH_dew = (H - H_dew)
        if dH_dew >= 0:
            self._T = self.mixture.xsolve_T(phase_data, H, T, P_dew)
            self._thermal_condition.P = P_dew
            return

        # Check if subcooled liquid
        P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = self.mixture.xH(phase_data, T, P_bubble)
        dH_bubble = (H - H_bubble)
        if dH_bubble <= 0:
            self._T = self.mixture.xsolve_T(phase_data, H, T, P_bubble)
            self._thermal_condition.P = P_bubble
            return

        # Guess overall vapor fraction, and vapor flow rates
        V = self._V or dH_bubble/(H_dew - H_bubble)
        
        # Guess composition in the vapor is a weighted average of boiling points
        self._refresh_v(V, y_bubble)
        F_mass = self._F_mass
        H_hat = H/F_mass
        P = flx.IQ_interpolation(self._H_hat_err_at_P,
                        P_bubble, P_dew,
                        H_bubble/F_mass - H_hat, H_dew/F_mass - H_hat,
                        self._P, self.P_tol, self.H_hat_tol,
                        (H_hat,), checkroot=False, checkbounds=False)
        self._P = self._thermal_condition.P = P   
        self._thermal_condition.T = T
    
    def set_PV(self, P, V):
        self._setup()
        self._thermal_condition.P = self._P = P
        if self._N == 0: raise RuntimeError('no chemicals present to perform VLE')
        if self._N == 1: return self._set_PV_chemical(P, V)
        
        # Setup bounderies
        thermal_condition = self._thermal_condition
        index = self._index
        mol = self._mol
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        if V == 1:
            T_dew, x_dew = self._dew_point.solve_Tx(self._z, P)
            vapor_mol[index] = mol
            liquid_mol[index] = 0
            thermal_condition.T = T_dew
        elif V == 0:
            T_bubble, y_bubble = self._bubble_point.solve_Ty(self._z, P)
            vapor_mol[index] = 0
            liquid_mol[index] = mol
            thermal_condition.T = T_bubble
        else:
            T_dew, x_dew = self._dew_point.solve_Tx(self._z, P)
            T_bubble, y_bubble = self._bubble_point.solve_Ty(self._z, P)
            self._refresh_v(V, y_bubble)
            self._V = V 
            T = flx.IQ_interpolation(self._V_err_at_T,
                                     T_bubble, T_dew, 0 - V, 1 - V,
                                     self._T, self.T_tol, self.V_tol,
                                     (V,), checkroot=False, checkbounds=False)
            
            # In case solver cannot reach the vapor fraction because
            # the composition does not converge at values that meets the
            # vapor fraction.
            V_offset = V - self._V
            if V_offset < -1e-2:
                F_mol_vapor = self._F_mol_vle * V
                v = y_bubble * F_mol_vapor
                mask = v > mol
                v[mask] = mol[mask]
                T = T_bubble
            elif V_offset > 1e2:
                v = x_dew * self._F_mol_vle * V
                mask = v < mol 
                v[mask] = mol[mask]
                T = T_dew
            else:
                v = self._v
            
            self._T = thermal_condition.T = T
            vapor_mol[index] = v
            liquid_mol[index] = mol - v
            self._H_hat = self.mixture.xH(self._phase_data, T, P)/self._F_mass
    
    def set_PH(self, P, H):
        self._setup()
        thermal_condition = self._thermal_condition
        thermal_condition.P = self._P = P
        if self._N == 0: 
            thermal_condition.T = self.mixture.xsolve_T(
                self._phase_data, H, thermal_condition.T, P
            )
            return
        if self._N == 1: return self._set_PH_chemical(P, H)
        
        # Setup bounderies
        index = self._index
        mol = self._mol
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        
        # Check if subcooled liquid
        T_bubble, y_bubble = self._bubble_point.solve_Ty(self._z, P)
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = self.mixture.xH(self._phase_data, T_bubble, P)
        dH_bubble = H - H_bubble
        if dH_bubble <= 0:
            thermal_condition.T = self.mixture.xsolve_T(self._phase_data, H, T_bubble, P)
            return
        
        # Check if super heated vapor
        T_dew, x_dew = self._dew_point.solve_Tx(self._z, P)
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = self.mixture.xH(self._phase_data, T_dew, P)
        dH_dew = H - H_dew
        if dH_dew >= 0:
            thermal_condition.T = self.mixture.xsolve_T(self._phase_data, H, T_dew, P)
            return
        
        # Guess T, overall vapor fraction, and vapor flow rates
        self._V = V = self._V or dH_bubble/(H_dew - H_bubble)
        self._refresh_v(V, y_bubble)
        
        F_mass = self._F_mass
        H_hat = H/F_mass
        H_hat_bubble = H_bubble/F_mass
        H_hat_dew = H_dew/F_mass
        T = flx.IQ_interpolation(self._H_hat_err_at_T,
                                 T_bubble, T_dew, 
                                 H_hat_bubble - H_hat, H_hat_dew - H_hat,
                                 self._T, self.T_tol, self.H_hat_tol,
                                 (H_hat,), checkroot=False, checkbounds=False)
        
        # Make sure enthalpy balance is correct
        self._T = thermal_condition.T = self.mixture.xsolve_T(
            self._phase_data, H, T, P
        )
        self._H_hat = H_hat
    
    def _estimate_v(self, V, y_bubble):
        return (V*self._z + (1-V)*y_bubble) * V * self._F_mol_vle
    
    def _refresh_v(self, V, y_bubble):
        y = self._y
        if y is None:
            self._v = self._estimate_v(V, y_bubble)
        else:
            self._v = y * self._F_mol_vle * V
    
    def _H_hat_err_at_T(self, T, H_hat):
        self._vapor_mol[self._index] = self._solve_v(T, self._P)
        self._liquid_mol[self._index] = self._mol - self._v
        self._H_hat = self.mixture.xH(self._phase_data, T, self._P)/self._F_mass
        return self._H_hat - H_hat
    
    def _H_hat_err_at_P(self, P, H_hat):
        self._vapor_mol[self._index] = self._solve_v(self._T , P)
        self._liquid_mol[self._index] = self._mol - self._v
        self._H_hat = self.mixture.xH(self._phase_data, self._T, P)/self._F_mass
        return self._H_hat - H_hat
    
    def _V_err_at_P(self, P, V):
        return self._solve_v(self._T , P).sum()/self._F_mol_vle - V
    
    def _V_err_at_T(self, T, V):
        return self._solve_v(T, self._P).sum()/self._F_mol_vle  - V
    
    def _x_iter(self, x, Psat_over_P_phi):
        x = x/x.sum()
        x[x < 1e-32] = 1e-32
        self._Ks = Psat_over_P_phi * self._gamma(x, self._T) * self._pcf(x, self._T)
        self._V = V = binary.phase_fraction(self._z, self._Ks, self._V)
        return self._z/(1. + V * (self._Ks - 1.))
    
    def _y_iter(self, y, Psats_over_P, T, P):
        phi = self._phi(y, T, P)
        Psat_over_P_phi = Psats_over_P / phi
        f = self._x_iter
        args = (Psat_over_P_phi,)
        x = flx.wegstein(f, self._x, 1e-12, args, checkroot=False)
        self._x = f(x, *args)
        if (np.abs(self._x - x) > 1e-6).any():
            x = flx.aitken(self._x_iter, self._x, 1e-6, args, checkroot=False)
        self._x = x
        v = self._F_mol_vle * self._V * x * self._Ks     
        return fn.normalize(v)
    
    def _solve_v(self, T, P):
        """Solve for vapor mol"""
        Psats_over_P = np.array([i(T) for i in
                                 self._bubble_point.Psats]) / P
        self._T = T
        v = self._v
        y = fn.normalize(v)
        l = self._mol - v
        self._x = fn.normalize(l)
        if isinstance(self._phi, IdealFugacityCoefficients):
            self._y = self._y_iter(y, Psats_over_P, T, P)
        else:
            self._y = flx.wegstein(self._y_iter, v/v.sum(), 1e-12,
                                   args=(Psats_over_P, T, P),
                                   checkroot=False)
        self._v = self._F_mol_vle * self._V * self._y
        return self._v

    def __format__(self, tabs=""):
        if not tabs: tabs = 1
        tabs = int(tabs)
        tab = tabs * 4 * " "
        imol = format(self.imol, str(2*tabs))
        if tabs:
            dlim = "\n" + tab
        else:
            dlim = ", "
        return (f"{type(self).__name__}(imol={imol},{dlim}"
                f"thermal_condition={self.thermal_condition})")
    
    def __repr__(self):
        return self.__format__("1")

class VLECache(Cache): load = VLE
del Cache    