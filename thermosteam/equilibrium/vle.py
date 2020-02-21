# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:40:05 2019

@author: yoelr
"""
from flexsolve import wegstein, IQ_interpolation, fixed_point
from ..utils.decorator_utils import thermo_user
from .dew_point import DewPoint
from .bubble_point import BubblePoint
from .fugacity_coefficients import IdealFugacityCoefficients
from .._thermal_condition import ThermalCondition
from ..utils import Cache
from numba import njit
import numpy as np

__all__ = ('VLE', 'V_2N', 'V_3N', 'V_error')

@njit
def V_error(V, zs, Ks):
    """Vapor fraction error."""
    return (zs*(Ks-1.)/(1.+V*(Ks-1.))).sum()
@njit
def V_2N(zs, Ks):
    """Solution for 2 component flash vessel."""
    z1, z2 = zs
    K1, K2 = Ks
    K1z1 = K1*z1
    K1z2 = K1*z2
    K2z1 = K2*z1
    K2z2 = K2*z2
    K1K2 = K1*K2
    K1K2z1 = K1K2*z1
    K1K2z2 = K1K2*z2
    z1_z2 = z1 + z2
    K1z1_K2z2 = K1z1 + K2z2
    return (-K1z1_K2z2 + z1_z2)/(K1K2z1 + K1K2z2 - K1z2 - K1z1_K2z2 - K2z1 + z1_z2)
@njit    
def V_3N(zs, Ks):
    """Solution for 3 component flash vessel."""
    z1, z2, z3 = zs
    K1, K2, K3 = Ks
    
    K1z1 = K1*z1
    K1z2 = K1*z2
    K1z3 = K1*z3
    
    K2z1 = K2*z1
    K2z2 = K2*z2
    K2z3 = K2*z3
    
    K3z1 = K3*z1
    K3z2 = K3*z2
    K3z3 = K3*z3
    
    K1K2 = K1*K2
    K1K3 = K1*K3
    K2K3 = K2*K3
    
    K1K2z1 = K1K2*z1
    K1K2z2 = K1K2*z2
    K1K3z1 = K1K3*z1
    K1K3z3 = K1K3*z3
    K2K3z2 = K2K3*z2
    K2K3z3 = K2K3*z3
    
    K12 = K1**2
    K22 = K2**2
    K32 = K3**2
    
    K12K2 = K12*K2
    K12K3 = K12*K3
    K12K22 = K12*K22
    K12K2K3 = K12*K2*K3
    K1K22 = K1*K22
    K1K22K3 = K1K22*K3
    
    K1K2K3 = K1K2*K3
    K22K3 = K22*K3
    K1K32 = K1*K32
    K2K32 = K2*K32
    K22K32 = K22*K32
    K12K32 = K12K3*K3
    
    
    z1_z2_z3 = z1 + z2 + z3
    z12 = z1**2
    z22 = z2**2
    z32 = z3**2
    z1z2 = z1*z2
    z1z3 = z1*z3
    z2z3 = z2*z3
    
    K1K2K32 = K1K2K3*K3
    K12K2K3 = K12K2*K3
    K12K22z12 = K12K22*z12
    K12K22z1z2 = K12K22*z1z2
    K12K22z22 = K12K22*z22
    K12K2K3z12 = K12K2K3*z12
    K12K2K3z1z2 = K12K2K3*z1z2
    K12K2K3z1z3 = K12K2K3*z1z3
    K12K2K3z2z3 = K12K2K3*z2z3
    K12K2z1z2 = K12K2*z1z2
    K12K2z1z3 = K12K2*z1z3
    K12K2z22 = K12K2*z22
    K12K2z2z3 = K12K2*z2z3
    K12K32z12 = K12K32*z12
    K12K32z1z3 = K12K32*z1z3
    K12K32z32 = K12K32*z32
    K12K3z1z2 = K12K3*z1z2
    K12K3z1z3 = K12K3*z1z3
    K12K3z2z3 = K12K3*z2z3
    K12K3z32 = K12K3*z32
    K12z2z3 = K12*z2z3
    K12z22 = K12*z22
    K12z32 = K12*z32
    K1K22K3z1z2 = K1K22K3*z1z2
    K1K22K3z1z3 = K1K22K3*z1z3
    K1K22K3z22 = K1K22K3*z22
    K1K22K3z2z3 = K1K22K3*z2z3
    K1K22z12 = K1K22*z12
    K1K22z1z2 = K1K22*z1z2
    K1K22z1z3 = K1K22*z1z3
    K1K22z2z3 = K1K22*z2z3
    K1K2K32z1z2 = K1K2K32*z1z2
    K1K2K32z1z3 = K1K2K32*z1z3
    K1K2K32z2z3 = K1K2K32*z2z3
    K1K2K32z32 = K1K2K32*z32
    K1K2K3z12 = K1K2K3*z12
    K1K2K3z1z2 = K1K2K3*z1z2
    K1K2K3z1z3 = K1K2K3*z1z3
    K1K2K3z22 = K1K2K3*z22
    K1K2K3z2z3 = K1K2K3*z2z3
    K1K2K3z32 = K1K2K3*z32
    K1K2z1z2 = K1K2*z1z2
    K1K2z1z3 = K1K2*z1z3
    K1K2z2z3 = K1K2*z2z3
    K1K2z32 = K1K2*z32
    K1K32z12 = K1K32*z12
    K1K2z1z3 = K1K2*z1z3
    K1K32z1z2 = K1K32*z1z2
    K1K32z1z3 = K1K32*z1z3
    K1K32z2z3 = K1K32*z2z3
    K1K3z1z2 = K1K3*z1z2
    K1K3z1z3 = K1K3*z1z3
    K1K3z22 = K1K3*z22
    K1K3z2z3 = K1K3*z2z3
    K22K32z22 = K22K32*z22
    K22K3z1z2 = K22K3*z1z2
    K22K32z2z3 = K22K32*z2z3
    K22K32z32 = K22K32*z32
    return ((-K1K2z1/2 - K1K2z2/2 - K1K3z1/2 - K1K3z3/2 + K1z1
             + K1z2/2 + K1z3/2 - K2K3z2/2 - K2K3z3/2 + K2z1/2 + K2z2
             + K2z3/2 + K3z1/2 + K3z2/2 + K3z3 - z1_z2_z3
             - (K12K22z12 + 2*K12K22z1z2 + K12K22z22
                - 2*K12K2K3z12 - 2*K12K2K3z1z2 - 2*K12K2K3z1z3
                + 2*K12K2K3z2z3 - 2*K12K2z1z2 + 2*K12K2z1z3
                - 2*K12K2z22 - 2*K12K2z2z3 + K12K32z12
                + 2*K12K32z1z3 + K12K32z32 + 2*K12K3z1z2
                - 2*K12K3z1z3 - 2*K12K3z2z3 - 2*K12K3z32
                + K12z22 + 2*K12z2z3 + K12z32
                - 2*K1K22K3z1z2 + 2*K1K22K3z1z3 - 2*K1K22K3z22
                - 2*K1K22K3z2z3 - 2*K1K22z12 - 2*K1K22z1z2
                - 2*K1K22z1z3 + 2*K1K22z2z3 + 2*K1K2K32z1z2
                - 2*K1K2K32z1z3 - 2*K1K2K32z2z3 - 2*K1K2K32z32
                + 4*K1K2K3z12 + 4*K1K2K3z1z2 + 4*K1K2K3z1z3
                + 4*K1K2K3z22 + 4*K1K2K3z2z3 + 4*K1K2K3z32
                + 2*K1K2z1z2 - 2*K1K2z1z3 - 2*K1K2z2z3 - 2*K1K2z32
                - 2*K1K32z12 - 2*K1K32z1z2 - 2*K1K32z1z3
                + 2*K1K32z2z3 - 2*K1K3z1z2 + 2*K1K3z1z3 - 2*K1K3z22
                - 2*K1K3z2z3 + K22K32z22 + 2*K22K32z2z3
                + K22K32z32 + 2*K22K3z1z2 - 2*K22K3*z1z3
                - 2*K22K3*z2z3 - 2*K22K3*z32 + K22*z12
                + 2*K22*z1z3 + K22*z32 - 2*K2K32*z1z2
                + 2*K2K32*z1z3 - 2*K2K32*z22 - 2*K2K32*z2z3
                - 2*K2K3*z12 - 2*K2K3*z1z2 - 2*K2K3*z1z3
                + 2*K2K3*z2z3 + K32*z12 + 2*K32*z1z2 + K32*z22)**0.5/2)
                / (K1K2K3*z1 + K1K2K3*z2 + K1K2K3*z3 - K1K2*z1 - K1K2*z2
                   - K1K2*z3 - K1K3*z1 - K1K3*z2 - K1K3*z3 + K1z1 + K1z2
                   + K1z3 - K2K3*z1 - K2K3*z2 - K2K3*z3 + K2z1 + K2z2 + K2z3
                   + K3z1 + K3z2 + K3z3 - z1_z2_z3))

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
                 '_TP', # [ThermalCondition] T and P results are stored here.
                 '_y', # [1d array] Molar vapor composition.
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
                 '_v',  # [1d array] Vapor molar data.
                 '_index', # [1d array] Index of chemicals in equilibrium.
                 '_F_mass', # [float] Total mass data.
                 '_chemical', # [Chemical] Single chemical in equilibrium.
                 '_mol', # [Chemical] Single chemical in equilibrium.
                 '_N', # [int] Number of chemicals in equilibrium.
                 '_solve_V', # [function] Solves for vapor fraction.
                 '_z', # [1d array] Molar composition of chemicals in equilibrium
                 '_Ks', # [1d array] Partition coefficients.
                 '_nonzero', # [1d array(bool)] Chemicals present in the mixture
                 '_F_mol', # [float] Total molar data.
                 '_F_mol_equilibrium', # [float] Total moles in equilibrium.
                 '_dew_point_cache', # [Cache] Retrieves the DewPoint object if arguments are the same.
                 '_bubble_point_cache') # [Cache] Retrieves the BubblePoint object if arguments are the same.
    
    solver = staticmethod(IQ_interpolation)
    itersolver = staticmethod(wegstein)
    T_tol = 0.00001
    P_tol = 0.1
    H_hat_tol = 0.1
    V_tol = 0.00001
    
    def __init__(self, imol, thermal_condition=None,
                 thermo=None, bubble_point_cache=None, dew_point_cache=None):
        self._T = self._P = self._H_hat = self._V = 0
        self._dew_point_cache = dew_point_cache or Cache(DewPoint)
        self._bubble_point_cache = bubble_point_cache or Cache(BubblePoint)
        self._load_thermo(thermo)
        self._imol = imol
        self._TP = thermal_condition or ThermalCondition(298.15, 101325.)
        self._phase_data = tuple(imol.iter_data())
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
                return self.set_TP(T, P)
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
                raise NotImplementedError('specification H and y not implemented')
            elif x_spec:
                raise NotImplementedError('specification H and x not implemented')
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
        mol = self._liquid_mol + self._vapor_mol
        notzero = mol > 0
        chemicals = self.chemicals
        if (self._nonzero == notzero).all():
            index = self._index
        else:
            # Set up indices for both equilibrium and non-equilibrium species
            index = chemicals.get_equilibrium_indices(notzero)
            self._y = None            
            self._N = N = len(index)
            eq_chems = chemicals.tuple
            eq_chems = [eq_chems[i] for i in index]
            self._nonzero = notzero
            self._index = index
            if N == 1:
                self._chemical, = eq_chems
            elif N == 2:
                self._solve_V = self._solve_V_2
            elif N == 3:
                self._solve_V = self._solve_V_3
            else:
                self._solve_V = self._solve_V_N
            
            if N != 1:
                # Set equilibrium objects
                thermo = self._thermo
                self._bubble_point = bp = self._bubble_point_cache.reload(eq_chems, thermo)
                self._dew_point = self._dew_point_cache.reload(eq_chems, thermo)
                self._pcf = bp.pcf
                self._gamma = bp.gamma
                self._phi = bp.phi
        
        # Get overall composition
        data = self._imol._data
        self._F_mass = (chemicals.MW * data).sum()
        F_mol = data.sum()
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
        self._F_mol = F_mol
        self._F_mol_equilibrium = F_mol_equilibrium = F_mol - F_mol_light - F_mol_heavy
        self._z = self._mol / F_mol_equilibrium

    @property
    def imol(self):
        return self._imol
    @property
    def thermal_condition(self):
        return self._TP

    ### Single component equilibrium case ###
        
    def _set_TP_chemical(self, T, P):
        # Either liquid or gas
        if P < self._chemical.Psat(T):
            self._liquid_mol[self._index] = 0
            self._vapor_mol[self._index] = self._mol
        else:
            self._liquid_mol[self._index] = self._mol
            self._vapor_mol[self._index] = 0
    
    def _set_TV_chemical(self, T, V):
        # Set vapor fraction
        self._T = self._TP.T = self._chemical.Psat(T)
        self._vapor_mol[self._index] = self._mol*V
        self._liquid_mol[self._index] = self._mol - self._vapor_mol[self._index]
        
    def _set_PV_chemical(self, P, V):
        # Set vapor fraction
        self._T = self._TP.T = self._chemical.Tsat(P)
        self._vapor_mol[self._index] = self._mol*V
        self._liquid_mol[self._index] = self._mol - self._vapor_mol[self._index]
        
    def _set_PH_chemical(self, P, H): 
        mol = self._mol
        index = self._index
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        thermo = self._thermo
        phase_data = self._phase_data
        
        # Set temperature in equilibrium
        self._T = self._TP.T = T = self._chemical.Tsat(P)
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = thermo.mixture.xH(phase_data, T, P)
        if H >= H_dew:
            self._TP.T = thermo.xsolve_T(phase_data, H, T, P)

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = thermo.mixture.xH(phase_data, T, P)
        if H <= H_bubble:
            self._TP.T = thermo.xsolve_T(phase_data, H, T, P)
        
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
        self._TP.P = P = self._chemical.Psat(T)
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = thermo.mixture.xH(phase_data, T, P)
        if H >= H_dew:
            # TODO: Possibly make this an error
            self._TP.T = thermo.xsolve_T(phase_data, H, T, P)

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = thermo.mixture.xH(phase_data, T, P)
        if H <= H_bubble:
            self._TP.T = thermo.xsolve_T(phase_data, H, T, P)
        
        # Adjust vapor fraction accordingly
        V = (H - H_bubble)/(H_dew - H_bubble)
        vapor_mol[index] = mol*V
        liquid_mol[index] = mol - vapor_mol[index]
        
    def _lever_rule(self, x, y):
        split_frac = (self._z[0]-x[0])/(y[0]-x[0])
        assert -0.00001 < split_frac < 1.00001, 'desired composition is infeasible'
        if split_frac > 1:
            split_frac = 1
        elif split_frac < 0:
            split_frac = 0
        self._vapor_mol[self._index] = v = self._F_mol_equilibrium * split_frac * y
        self._liquid_mol[self._index] = self._mol - v
    
    def set_Tx(self, T, x):
        self._setup()
        assert self._N == 2, 'number of species in equilibrium must be 2 to specify x'
        self._TP.P, y = self._bubble_point.solve_Py(x, T)
        self._lever_rule(x, y)
    
    def set_Px(self, P, x):
        self._setup()
        assert self._N == 2, 'number of species in equilibrium must be 2 to specify x'
        self._TP.T, y = self._bubble_point.solve_Ty(x, P) 
        self._lever_rule(x, y)
        
    def set_Ty(self, T, y):
        self._setup()
        assert self._N == 2, 'number of species in equilibrium must be 2 to specify y'
        self._TP.P, x = self._dew_point.solve_Px(y, T)
        self._lever_rule(x, y)
    
    def set_Py(self, P, y):
        self._setup()
        assert self._N == 2, 'number of species in equilibrium must be 2 to specify y'
        self._TP.T, x = self._dew_point.solve_Tx(y, P) 
        self._lever_rule(x, y)
        
    def set_TP(self, T, P):
        self._setup()
        TP = self._TP
        self._T = TP.T = T
        self._P = TP.P = P
        if self._N == 1: return self._set_TP_chemical(T, P)
        # Setup bounderies
        P_dew, x_dew = self._dew_point.solve_Px(self._z, T)
        P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
        
        # Check if there is equilibrium
        if P <= P_dew:
            self._vapor_mol[self._index] = self._mol
            self._liquid_mol[self._index] = 0
        elif P >= P_bubble:
            self._vapor_mol[self._index] = 0
            self._liquid_mol[self._index] = self._mol
        else:
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
            self._H_hat = self.mixture.xH_at_TP(self._phase_data, TP)/self._F_mass
        
    def set_TV(self, T, V):
        self._setup()
        TP = self._TP
        TP.T = self._T = T
        if self._N == 1: return self._set_TV_chemical(T, V)
        P_dew, x_dew = self._dew_point.solve_Px(self._z, T)
        P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
        if V == 1:
            self._vapor_mol[self._index] = self._mol
            self._liquid_mol[self._index] = 0
            TP.P = P_dew
        elif V == 0:
            self._vapor_mol[self._index] = 0
            self._liquid_mol[self._index] = self._mol
            TP.P = P_bubble
        else:
            self._V = V 
            self._refresh_v(V, y_bubble)
            try:
                P = self.solver(self._V_at_P,
                                P_bubble, P_dew, 0, 1,
                                self._P, self._V,
                                self.P_tol, self.V_tol)
            except:
                self._V = V
                self._v = self._estimate_v(V, y_bubble)
                P = self.solver(self._V_at_P,
                                P_bubble, P_dew, 0, 1,
                                self._P, V,
                                self.P_tol, self.V_tol)
            self._P = TP.P = P
            self._vapor_mol[self._index] = self._v
            self._liquid_mol[self._index] = self._mol - self._v
            self._H_hat = self.mixture.xH_at_TP(self._phase_data, TP)/self._F_mass

    def set_TH(self, T, H):
        self._setup()
        if self._N == 1: return self._set_TH_chemical(T, H)
        self._T = T
        
        # Setup bounderies
        P_dew, x_dew = self._dew_point.solve_Px(self._z, T)
        P_bubble, y_bubble = self._bubble_point.solve_Py(self._z, T)
        index = self._index
        mol = self._mol
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        phase_data = self._phase_data
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = self.mixture.xH(phase_data, T, P_dew)
        dH_dew = (H - H_dew)
        if dH_dew >= 0:
            # TODO: Possibly make this an error
            self._T = self.mixture.xsolve_T(phase_data, H, T, P_dew)
            self._TP.P = P_dew

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = self.mixture.xH(phase_data, T, P_bubble)
        dH_bubble = (H - H_bubble)
        if dH_bubble <= 0:
            self._T = self.mixture.xsolve_T(phase_data, H, T, P_bubble)
            self._TP.P = P_bubble

        # Guess overall vapor fraction, and vapor flow rates
        V = self._V or dH_bubble/(H_dew - H_bubble)
        # Guess composition in the vapor is a weighted average of boiling points
        self._refresh_v(V, y_bubble)
        F_mass = self._F_mass
        self._H_hat = H/F_mass
        try:
            P = self.solver(self._H_hat_at_P,
                            P_bubble, P_dew,
                            H_bubble/F_mass, H_dew/F_mass,
                            self._P, self._H_hat,
                            self.P_tol, self.H_hat_tol) 
        except:
            self._v = self._estimate_v(V, y_bubble)
            P = self.solver(self._H_hat_at_P,
                            P_bubble, P_dew,
                            H_bubble/F_mass, H_dew/F_mass,
                            self._P, self._H_hat,
                            self.P_tol, self.H_hat_tol) 
        self._P = self._TP.P = P   
        self._TP.T = T
    
    def set_PV(self, P, V):
        self._setup()
        self._TP.P = self._P = P
        if self._N == 1: return self._set_PV_chemical(P, V)
        
        # Setup bounderies
        T_dew, x_dew = self._dew_point.solve_Tx(self._z, P)
        T_bubble, y_bubble = self._bubble_point.solve_Ty(self._z, P)
        
        index = self._index
        mol = self._mol
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        
        if V == 1:
            vapor_mol[index] = mol
            liquid_mol[index] = 0
            self._TP.T = T_dew
        elif V == 0:
            vapor_mol[index] = 0
            liquid_mol[index] = mol
            self._TP.T = T_bubble
        else:
            self._refresh_v(V, y_bubble)
            self._V = V 
            try:
                T = self.solver(self._V_at_T,
                                T_bubble, T_dew, 0, 1,
                                self._T , V,
                                self.T_tol, self.V_tol)
            except:
                self._v = self._estimate_v(V, y_bubble)
                T = self.solver(self._V_at_T,
                                T_bubble, T_dew, 0, 1,
                                self._T , V,
                                self.T_tol, self.V_tol)
            self._T = self._TP.T = T
            vapor_mol[index] = self._v
            liquid_mol[index] = mol - self._v
            self._H_hat = self.mixture.xH(self._phase_data, self._T, P)/self._F_mass
    
    def set_PH(self, P, H):
        self._setup()
        self._TP.P = self._P = P
        if self._N == 1: return self._set_PH_chemical(P, H)
        
        # Setup bounderies
        T_dew, x_dew = self._dew_point.solve_Tx(self._z, P)
        T_bubble, y_bubble = self._bubble_point.solve_Ty(self._z, P)
        
        index = self._index
        mol = self._mol
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = self.mixture.xH(self._phase_data, T_dew, P)
        dH_dew = H - H_dew
        if dH_dew >= 0:
            self._TP.T = self.mixture.xsolve_T(self._phase_data, H, T_dew, P)

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = self.mixture.xH(self._phase_data, T_dew, P)
        dH_bubble = H - H_bubble
        if dH_bubble <= 0:
            self._TP.T = self.mixture.xsolve_T(self._phase_data, H, T_bubble, P)
        
        # Guess T, overall vapor fraction, and vapor flow rates
        self._V = V = self._V or dH_bubble/(H_dew - H_bubble)
        self._refresh_v(V, y_bubble)
        
        F_mass = self._F_mass
        self._H_hat = H/F_mass
        self._T = self._TP.T = self.solver(self._H_hat_at_T,
                                           T_bubble, T_dew, 
                                           H_bubble/F_mass, H_dew/F_mass,
                                           self._T , self._H_hat,
                                           self.T_tol, self.H_hat_tol)
    
    def _estimate_v(self, V, y_bubble):
        return (V*self._z + (1-V)*y_bubble) * V * self._F_mol_equilibrium
    
    def _refresh_v(self, V, y_bubble):
        y = self._y
        if y is None:
            self._v = self._estimate_v(V, y_bubble)
        else:
            self._v = y * self._F_mol_equilibrium * V
    
    def _H_hat_at_T(self, T):
        self._vapor_mol[self._index] = self._solve_v(T, self._P)
        self._liquid_mol[self._index] = self._mol - self._v
        return self.mixture.xH(self._phase_data, T, self._P)/self._F_mass
    
    def _H_hat_at_P(self, P):
        self._vapor_mol[self._index] = self._solve_v(self._T , P)
        self._liquid_mol[self._index] = self._mol - self._v
        return self.mixture.xH(self._phase_data, self._T, P)/self._F_mass
    
    def _V_at_P(self, P):
        return self._solve_v(self._T , P).sum()/self._F_mol
    
    def _V_at_T(self, T):
        V = self._solve_v(T, self._P).sum()/self._F_mol 
        return V
    
    def _x_iter(self, x, Psat_over_P_phi):
        x[x < 0.] = 0.
        x = x/x.sum()
        self._Ks = Psat_over_P_phi * self._gamma(x, self._T) * self._pcf(x, self._T)
        return self._z/(1. + self._solve_V() * (self._Ks - 1.))
    
    def _y_iter(self, y, Psats_over_P, T, P):
        phi = self._phi(y, T, P)
        Psat_over_P_phi = Psats_over_P / phi
        try:
            self._x = x = self.itersolver(self._x_iter,
                                          self._x, 1e-4,
                                          args=(Psat_over_P_phi,))
        except:
            self._x = x = fixed_point(self._x_iter,
                                      self._x, 1e-4,
                                      args=(Psat_over_P_phi,))
        self._v = v = self._F_mol_equilibrium * self._V * x * self._Ks     
        return v / v.sum()
    
    def _solve_v(self, T, P):
        """Solve for vapor mol"""
        Psats_over_P = np.array([i(T) for i in
                                 self._bubble_point.Psats]) / P
        self._T = T
        v = self._v
        y = v / v.sum()
        l = self._mol - v
        self._x = l / l.sum()
        if isinstance(self._phi, IdealFugacityCoefficients):
            self._y = self._y_iter(y, Psats_over_P, T, P)
        else:
            self._y = self.itersolver(self._y_iter, v/v.sum(), 1e-4,
                                      args=(Psats_over_P, T, P))
        return self._v

    def _load_V(self, V):
        if V > 1.:
            V = 1.
        elif V < 0.:
            V = 0.
        self._V = V
        return V

    def _V_error(self, V):
        """Vapor fraction error."""
        return (self._z * (self._Ks-1.) / (1. + V * (self._Ks-1.))).sum()

    def _solve_V_N(self):
        """Update V for N components."""
        V = self.solver(self._V_error, 0, 1,
                        self._V_error(0), self._V_error(1),
                        self._V, 0, 1e-4, 1e-7)
        return self._load_V(V)

    def _solve_V_2(self):
        """Update V for 2 components."""
        V = V_2N(self._z, self._Ks)
        return self._load_V(V)
    
    def _solve_V_3(self):
        """Update V for 3 components."""
        V = V_3N(self._z, self._Ks)
        return self._load_V(V)

    def __format__(self, tabs=""):
        if not tabs: tabs = 1
        tabs = int(tabs)
        tab = tabs * 4 * " "
        imol = format(self.imol, str(2*tabs))
        if tabs:
            dlim = "\n" + tab
        else:
            dlim = ", "
        return (f"VLE(imol={imol},{dlim}"
                f"thermal_condition={self.thermal_condition})")
    
    def __repr__(self):
        return self.__format__("1")
