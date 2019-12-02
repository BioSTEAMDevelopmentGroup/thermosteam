# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:40:05 2019

@author: yoelr
"""
from flexsolve import bounded_wegstein, wegstein, aitken, \
                      bounded_aitken, IQ_interpolation
from ..settings import settings
from .dew_point import DewPoint
from .bubble_point import BubblePoint
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
    return (-K1*z1 - K2*z2 + z1 + z2)/(K1*K2*z1 + K1*K2 *
                                       z2 - K1*z1 - K1*z2
                                       - K2*z1 - K2*z2 + z1 + z2)
@njit    
def V_3N(zs, Ks):
    """Solution for 3 component flash vessel."""
    z1, z2, z3 = zs
    K1, K2, K3 = Ks
    return (-K1*K2*z1/2 - K1*K2*z2/2 - K1*K3*z1/2 - K1*K3*z3/2 + K1*z1 + K1*z2/2 + K1*z3/2 - K2*K3*z2/2 - K2*K3*z3/2 + K2*z1/2 + K2*z2 + K2*z3/2 + K3*z1/2 + K3*z2/2 + K3*z3 - z1 - z2 - z3 - (K1**2*K2**2*z1**2 + 2*K1**2*K2**2*z1*z2 + K1**2*K2**2*z2**2 - 2*K1**2*K2*K3*z1**2 - 2*K1**2*K2*K3*z1*z2 - 2*K1**2*K2*K3*z1*z3 + 2*K1**2*K2*K3*z2*z3 - 2*K1**2*K2*z1*z2 + 2*K1**2*K2*z1*z3 - 2*K1**2*K2*z2**2 - 2*K1**2*K2*z2*z3 + K1**2*K3**2*z1**2 + 2*K1**2*K3**2*z1*z3 + K1**2*K3**2*z3**2 + 2*K1**2*K3*z1*z2 - 2*K1**2*K3*z1*z3 - 2*K1**2*K3*z2*z3 - 2*K1**2*K3*z3**2 + K1**2*z2**2 + 2*K1**2*z2*z3 + K1**2*z3**2 - 2*K1*K2**2*K3*z1*z2 + 2*K1*K2**2*K3*z1*z3 - 2*K1*K2**2*K3*z2**2 - 2*K1*K2**2*K3*z2*z3 - 2*K1*K2**2*z1**2 - 2*K1*K2**2*z1*z2 - 2*K1*K2**2*z1*z3 + 2*K1*K2**2*z2*z3 + 2*K1*K2*K3**2*z1*z2 - 2*K1*K2*K3**2*z1*z3 - 2*K1*K2*K3**2*z2*z3 - 2*K1*K2*K3**2*z3**2 + 4*K1*K2*K3*z1**2 + 4*K1*K2*K3*z1*z2 + 4*K1*K2*K3*z1*z3 + 4*K1*K2*K3*z2**2 + 4*K1*K2*K3*z2*z3 + 4*K1*K2*K3*z3**2 + 2*K1*K2*z1*z2 - 2*K1*K2*z1*z3 - 2*K1*K2*z2*z3 - 2*K1*K2*z3**2 - 2*K1*K3**2*z1**2 - 2*K1*K3**2*z1*z2 - 2*K1*K3**2*z1*z3 + 2*K1*K3**2*z2*z3 - 2*K1*K3*z1*z2 + 2*K1*K3*z1*z3 - 2*K1*K3*z2**2 - 2*K1*K3*z2*z3 + K2**2*K3**2*z2**2 + 2*K2**2*K3**2*z2*z3 + K2**2*K3**2*z3**2 + 2*K2**2*K3*z1*z2 - 2*K2**2*K3*z1*z3 - 2*K2**2*K3*z2*z3 - 2*K2**2*K3*z3**2 + K2**2*z1**2 + 2*K2**2*z1*z3 + K2**2*z3**2 - 2*K2*K3**2*z1*z2 + 2*K2*K3**2*z1*z3 - 2*K2*K3**2*z2**2 - 2*K2*K3**2*z2*z3 - 2*K2*K3*z1**2 - 2*K2*K3*z1*z2 - 2*K2*K3*z1*z3 + 2*K2*K3*z2*z3 + K3**2*z1**2 + 2*K3**2*z1*z2 + K3**2*z2**2)**0.5/2)/(K1*K2*K3*z1 + K1*K2*K3*z2 + K1*K2*K3*z3 - K1*K2*z1 - K1*K2*z2 - K1*K2*z3 - K1*K3*z1 - K1*K3*z2 - K1*K3*z3 + K1*z1 + K1*z2 + K1*z3 - K2*K3*z1 - K2*K3*z2 - K2*K3*z3 + K2*z1 + K2*z2 + K2*z3 + K3*z1 + K3*z2 + K3*z3 - z1 - z2 - z3)

class VLE:
    """Create a VLE object for solving VLE."""
    __slots__ = ('_T', '_P', '_H_hat', '_V', 'thermo',
                 '_dew_point', '_bubble_point', '_mixture',
                 '_phi', '_pcf', '_gamma',
                 '_liquid_mol', '_vapor_mol', '_phase_data',
                 '_v',  '_index', '_massnet', '_chemical',
                 '_update_V', '_mol', '_molnet', '_N', '_solve_V',
                 '_zs', '_Ks', '_Psat_over_P_phi')
    
    solver = staticmethod(IQ_interpolation)
    itersolver = staticmethod(aitken)
    T_tol = 0.00001
    P_tol = 0.1
    H_hat_tol = 0.1
    V_tol = 0.00001
    
    def __init__(self, thermo=None):
        self._T = self._P = self._H_hat = self._V = 0
        self.thermo = thermo = settings.get_default_thermo(thermo)
        self._mixture = thermo.mixture
    
    def __call__(self, phases, material_data, IDs=None, LNK=None, HNK=None,
                 P=None, H=None, T=None, V=None, x=None, y=None):
        """Perform vapor-liquid equilibrium.

        Parameters
        ----------
        phases : Iterable[str]
                 Phases corresponding to material data.
        material_data : ndarray
                        Material data with phases by row and chemicals by column.
        Specify two:
            * **P:** Operating pressure (Pa)
            * **Q:** Energy input (kJ/hr)
            * **T:** Operating temperature (K)
            * **V:** Molar vapor fraction
            * **x:** Molar composition of liquid (for binary mixture)
            * **y:** Molar composition of vapor (for binary mixture)
        IDs = None : tuple, optional
                     IDs of chemicals in equilibrium.
        LNK = None : tuple[str], optional
              Light non-keys that remain as a vapor (disregards equilibrium).
        LNK = None : tuple[str], optional
              Heavy non-keys that remain as a liquid (disregards equilibrium).

        """
        ### Decide what kind of equilibrium to run ###
        T_spec = T is not None
        P_spec = P is not None
        V_spec = V is not None
        H_spec = H is not None
        x_spec = x is not None
        y_spec = y is not None
        N_specs = (P_spec + H_spec + T_spec + V_spec + x_spec + y_spec)
        assert N_specs == 2, ("must pass two and only two of the following specifications: "
                              "T, P, V, H, x, y")
        # Setup material flows
        self.setup(phases, material_data, IDs, LNK, HNK)
        
        # Run equilibrium
        if T_spec:
            if P_spec:
                return self.TP(T, P)
            elif V_spec:
                return self.TV(T, V)
            elif H_spec:
                return self.TH(T, H)
            elif x_spec:
                return self.Tx(T, np.asarray(x))
            else: # y_spec
                return self.Ty(T, np.asarray(y))
        elif P_spec:
            if V_spec:
                return self.PV(P, V)
            elif H_spec:
                return self.PH(P, H)
            elif x_spec:
                return self.Px(P, np.asarray(x))
            else: # y_spec
                return self.Py(P, np.asarray(y))
        elif H_spec:
            if y_spec:
                raise NotImplementedError('specification H and y not implemented')
            elif x_spec:
                raise NotImplementedError('specification H and x not implemented')
            else: # V_spec
                raise NotImplementedError('specification V and H not implemented yet')
        else: # x_spec and y_spec
            raise ValueError("can only pass either 'x' or 'y' arguments, not both")
    
    def setup(self, phases, material_data, IDs, LNK, HNK):
        self._phase_data = tuple(zip(phases, material_data))
        self._liquid_mol = liquid_mol = material_data[phases.index('l')]
        self._vapor_mol = vapor_mol = material_data[phases.index('g')]

        # Get flow rates
        mol = liquid_mol + vapor_mol
        notzero = mol > 0

        # Reused attributes
        chemicals = self.thermo.chemicals
        dct = chemicals.__dict__
        index = chemicals._index

        # Set up indices for both equilibrium and non-equilibrium species
        if IDs:
            eq_index = [index[specie] for specie in IDs]
            eq_chems = [dct[ID] for ID in IDs]
        else:
            eq_index = chemicals._equilibrium_indices(notzero)
            chems = chemicals._chemicals
            eq_chems = [chems[i] for i in eq_index]
        if LNK:
            LNK_index = [index[i] for i in LNK]
        else:
            LNK_index = chemicals._light_indices(notzero)
        if HNK:
            HNK_index = [index[i] for i in HNK]
        else:
            HNK_index = chemicals._heavy_indices(notzero)
        self._mol = mol[eq_index]
        self._index = eq_index

        # Set light and heavy keys
        vapor_mol[HNK_index] = 0
        vapor_mol[LNK_index] = mol[LNK_index]
        liquid_mol[LNK_index] = 0
        liquid_mol[HNK_index] = mol[HNK_index]
        
        self._N = N = len(eq_index)
        if N == 1:
            self._chemical, = eq_chems
            return 
        elif N == 2:
            self._solve_V = self._solve_V_2
        elif N == 3:
            self._solve_V = self._solve_V_3
        else:
            self._solve_V = self._solve_V_N
        self._massnet = (chemicals._MW * material_data).sum()
        
        # Get overall composition
        self._molnet = molnet = material_data.sum()
        assert molnet != 0, 'empty stream cannot perform equilibrium'
        self._zs = self._mol/molnet
        
        # Set equilibrium objects
        thermo = self.thermo
        self._bubble_point = bp = BubblePoint(eq_chems, thermo)
        self._dew_point = DewPoint(eq_chems, thermo, bp)
        self._pcf = bp.pcf
        self._gamma = bp.gamma
        self._phi = bp.phi

    ### Single component equilibrium case ###
        
    def _TP_chemical(self, T, P):
        # Either liquid or gas
        if P < self._chemical.Psat(T):
            self._liquid_mol[self._index] = 0
            self._vapor_mol[self._index] = self._mol
        else:
            self._liquid_mol[self._index] = self._mol
            self._vapor_mol[self._index] = 0
    
    def _TV_chemical(self, T, V):
        # Set vapor fraction
        self._T = T = self._chemical.Psat(T)
        self._vapor_mol[self._index] = self._mol*V
        self._liquid_mol[self._index] = self._mol - self._vapor_mol[self._index]
        return T
        
    def _PV_chemical(self, P, V):
        # Set vapor fraction
        self._T = T = self._chemical.Tsat(P)
        self._vapor_mol[self._index] = self._mol*V
        self._liquid_mol[self._index] = self._mol - self._vapor_mol[self._index]
        return T
        
    def _PH_chemical(self, P, H): 
        mol = self._mol
        index = self._index
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        thermo = self.thermo
        phase_data = self._phase_data
        
        # Set temperature in equilibrium
        self._T = T = self._chemical.Tsat(P)
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = thermo.mixture.xH(phase_data, T)
        if H >= H_dew:
            self._T = T = thermo.xsolve_T(phase_data, H, T)
            return T

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = thermo.mixture.xH(phase_data, T)
        if H <= H_bubble:
            self._T = T = thermo.xsolve_T(phase_data, H, T)
            return T
        
        # Adjust vapor fraction accordingly
        V = (H - H_bubble)/(H_dew - H_bubble)
        vapor_mol[index] = mol*V
        liquid_mol[index] = mol - vapor_mol[index]
        return T
        
    def _TH_chemical(self, T, H):
        index = self._index
        mol = self._mol
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        thermo = self.thermo
        phase_data = self._phase_data
        
        # Set Pressure in equilibrium
        P = self._chemical.Psat(T)
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = thermo.mixture.xH(phase_data, T)
        if H >= H_dew:
            # TODO: Possibly make this an error
            self._T = thermo.xsolve_T(phase_data, H, T)
            return P

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = thermo.mixture.xH(phase_data, T)
        if H <= H_bubble:
            self._T = thermo.xsolve_T(phase_data, H, T)
            return P
        
        # Adjust vapor fraction accordingly
        V = (H - H_bubble)/(H_dew - H_bubble)
        vapor_mol[index] = mol*V
        liquid_mol[index] = mol - vapor_mol[index]
        return P
        
    def _lever_rule(self, x, y):
        split_frac = (self._zs[0]-x[0])/(y[0]-x[0])
        assert -0.00001 < split_frac < 1.00001, 'desired composition is infeasible'
        if split_frac > 1:
            split_frac = 1
        elif split_frac < 0:
            split_frac = 0
        self._vapor_mol[self._index] = v = self._molnet * split_frac * y
        self._liquid_mol[self._index] = self._mol - v
    
    def Tx(self, T, x):
        assert self._N == 2, 'number of species in equilibrium must be 2 to specify x'
        P, y = self._bubble_point.solve_Px(x, T)
        self._lever_rule(x, y)
        return P
    
    def Px(self, P, x):
        assert self._N == 2, 'number of species in equilibrium must be 2 to specify x'
        T , y = self._bubble_point.solve_Ty(x, P) 
        self._lever_rule(x, y)
        return T
        
    def Ty(self, T, y):
        assert self._N == 2, 'number of species in equilibrium must be 2 to specify y'
        P, x = self._dew_point.solve_Px(y, T)
        self._lever_rule(x, y)
        return P
    
    def Py(self, P, y):
        assert self._N == 2, 'number of species in equilibrium must be 2 to specify y'
        T , x = self._dew_point.solve_Ty(y, P) 
        self._lever_rule(x, y)
        return T
        
    def TP(self, T, P):
        self._T = T
        self._P = P
        if self._N == 1: return self._TP_chemical(T, P)
        # Setup bounderies
        P_dew, x_dew = self._dew_point.solve_Px(self._zs, T)
        P_bubble, y_bubble = self._bubble_point.solve_Py(self._zs, T)
        
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
            self._V = V = self._V or (T - P_dew)/(P_bubble - P_dew)
            y = V*self._zs + (1-V)*y_bubble
            
            # Guess vapor flow rates
            self._v = y * V * self._mol

            # Solve
            self._vapor_mol[self._index] = self._solve_v(T, P)
            self._liquid_mol[self._index] = self._mol - self._v
            self._H_hat = self._mixture.xH(self._phase_data, T)/self._massnet
        
    def TV(self, T, V):
        self._T = T
        if self._N == 1: return self._TV_chemical(T, V)
        P_dew, x_dew = self._dew_point.solve_Px(self._zs, T)
        P_bubble, y_bubble = self._bubble_point.solve_Py(self._zs, T)
        if V == 1:
            self._vapor_mol[self._index] = self._mol
            self._liquid_mol[self._index] = 0
            return P_dew
        elif V == 0:
            self._vapor_mol[self._index] = 0
            self._liquid_mol[self._index] = self._mol
            return P_bubble
        else:
            self._V = V
            self._v = (V*self._zs + (1-V)*y_bubble)*V*self._molnet
            self._P = P = self.solver(self._V_at_P,
                                      P_bubble, P_dew, 0, 1,
                                      self._P, self._V,
                                      self.P_tol, self.V_tol)
            self._vapor_mol[self._index] = self._v
            self._liquid_mol[self._index] = self._mol - self._v
            self._H_hat = self._mixture.xH(self._phase_data, T)/self._massnet
            return P

    def TH(self, T, H):
        if self._N == 1: return self._TH_chemical(T, H)
        self._T = T
        
        # Setup bounderies
        P_dew, x_dew = self._dew_point.solve_Px(self._zs, T)
        P_bubble, y_bubble = self._bubble_point.solve_Py(self._zs, T)
        index = self._index
        mol = self._mol
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = self._mixture.xH(self._phase_data, T)
        dH_dew = (H - H_dew)
        if dH_dew >= 0:
            # TODO: Possibly make this an error
            self._T = self._mixture.xsolve_T(self._phase_data, H, T)
            return P_dew

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = self._mixture.xH(self._phase_data, T)
        dH_bubble = (H - H_bubble)
        if dH_bubble <= 0:
            self._T = self._mixture.xsolve_T(self._phase_data, H, T)
            return P_bubble

        # Guess overall vapor fraction, and vapor flow rates
        V = self._V or dH_bubble/(H_dew - H_bubble)
        # Guess composition in the vapor is a weighted average of boiling points
        self._v = V*self._zs + (1-V)*y_bubble*V*self._molnet
        massnet = self._massnet
        self._H_hat = H/massnet
        self._P = P = self.solver(self._H_hat_at_P,
                                  P_bubble, P_dew,
                                  H_bubble/massnet, H_dew/massnet,
                                  self._P, self._H_hat,
                                  self.P_tol, self.H_hat_tol) 
        return P
    
    def PV(self, P, V):
        self._P = P
        if self._N == 1: return self._PV_chemical(P, V)
        
        # Setup bounderies
        T_dew, x_dew = self._dew_point.solve_Tx(self._zs, P)
        T_bubble, y_bubble = self._bubble_point.solve_Ty(self._zs, P)
        
        index = self._index
        mol = self._mol
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        
        if V == 1:
            vapor_mol[index] = mol
            liquid_mol[index] = 0
            return T_dew
        elif V == 0:
            vapor_mol[index] = 0
            liquid_mol[index] = mol
            return T_bubble
        else:
            self._v = (V*self._zs + (1-V)*y_bubble) * V * self._molnet
            self._V = V
            self._T = T = self.solver(self._V_at_T,
                                      T_bubble, T_dew, 0, 1,
                                      self._T , V,
                                      self.T_tol, self.V_tol)
            vapor_mol[index] = self._v
            liquid_mol[index] = mol - self._v
            self._P = P
            self._H_hat = self._mixture.xH(self._phase_data, T)/self._massnet
        return T
    
    def PH(self, P, H):
        self._P = P
        if self._N == 1: return self._PH_chemical(P, H)
        
        # Setup bounderies
        T_dew, x_dew = self._dew_point.solve_Tx(self._zs, P)
        T_bubble, y_bubble = self._bubble_point.solve_Ty(self._zs, P)
        
        index = self._index
        mol = self._mol
        vapor_mol = self._vapor_mol
        liquid_mol = self._liquid_mol
        
        # Check if super heated vapor
        vapor_mol[index] = mol
        liquid_mol[index] = 0
        H_dew = self._mixture.xH(self._phase_data, T_dew)
        dH_dew = H - H_dew
        if dH_dew >= 0:
            self._T = T = self._mixture.xsolve_T(self._phase_data, H, T_dew)
            return T

        # Check if subcooled liquid
        vapor_mol[index] = 0
        liquid_mol[index] = mol
        H_bubble = self._mixture.xH(self._phase_data, T_dew)
        dH_bubble = H - H_bubble
        if dH_bubble <= 0:
            self._T = T = self._mixture.xsolve_T(self._phase_data, H, T_bubble)
            return T
        
        # Guess T, overall vapor fraction, and vapor flow rates
        self._V = V = self._V or dH_bubble/(H_dew - H_bubble)
        self._v = (V*self._zs + (1-V)*y_bubble) * V * self._molnet
        
        massnet = self._massnet
        self._H_hat = H/massnet
        self._T = T = self.solver(self._H_hat_at_T,
                                  T_bubble, T_dew, 
                                  H_bubble/massnet, H_dew/massnet,
                                  self._T , self._H_hat,
                                  self.T_tol, self.H_hat_tol)
        return T
    
    def _H_hat_at_T(self, T):
        self._vapor_mol[self._index] = self._solve_v(T, self._P)
        self._liquid_mol[self._index] = self._mol - self._v
        return self._mixture.xH(self._phase_data, T)/self._massnet
    
    def _H_hat_at_P(self, P):
        self._vapor_mol[self._index] = self._solve_v(self._T , P)
        self._liquid_mol[self._index] = self._mol - self._v
        return self._mixture.xH(self._phase_data, self._T )/self._massnet
    
    def _V_at_P(self, P):
        return self._solve_v(self._T , P).sum()/self._molnet
    
    def _V_at_T(self, T):
        return self._solve_v(T, self._P).sum()/self._molnet
    
    def _x_iter(self, x):
        x = x/x.sum()
        self._Ks = self._Psat_over_P_phi * self._gamma(x, self._T) * self._pcf(x, self._T)
        return self._zs/(1. + self._solve_V()*(self._Ks-1.))
     
    def _solve_v(self, T, P):
        """Solve for vapor mol"""
        bp = self._bubble_point
        phi = self._phi
        mol = self._mol
        v = self._v
        l = mol - v
        y = v/mol
        Psats_over_P = np.array([i(T) for i in bp.Psats]) / P
        self._Psat_over_P_phi = Psats_over_P / phi(y, T, P)
        x = self.itersolver(self._x_iter, l/l.sum(), 1e-4)
        v = self._molnet*self._V*x/x.sum()*self._Ks            
        Psat_over_P_phi = Psats_over_P / phi(y, T, P)
        if np.abs(self._Psat_over_P_phi - Psat_over_P_phi).sum() < 0.001:
            # TODO: Make y solve iteratively with Psat_over_P_phi
            self._v = v
        else:
            x = self.itersolver(self._x_iter, x, 1e-4)
            self._v = v = self._molnet*self._V*x/x.sum()*self._Ks            
        return v

    def _V_error(self, V):
        """Vapor fraction error."""
        return (self._zs*(self._Ks-1.)/(1.+V*(self._Ks-1.))).sum()

    def _solve_V_N(self):
        """Update V for N components."""
        self._V = self.solver(self._V_error, 0, 1,
                             self._V_error(0), self._V_error(1),
                             self._V, 0, 1e-4, 1e-7)
        return self._V

    def _solve_V_2(self):
        """Update V for 2 components."""
        z1, z2 = self._zs
        K1, K2 = self._Ks
        self._V = (-K1*z1 - K2*z2 + z1 + z2)/(K1*K2*z1 + K1*K2 *
                                             z2 - K1*z1 - K1*z2
                                             - K2*z1 - K2*z2 + z1 + z2)
        return self._V
    
    def _solve_V_3(self):
        """Update V for 3 components."""
        z1, z2, z3 = self._zs
        K1, K2, K3 = self._Ks
        self._V = ((-K1*K2*z1/2 - K1*K2*z2/2 - K1*K3*z1/2 - K1*K3*z3/2 + K1*z1
                   + K1*z2/2 + K1*z3/2 - K2*K3*z2/2 - K2*K3*z3/2 + K2*z1/2
                   + K2*z2 + K2*z3/2 + K3*z1/2 + K3*z2/2 + K3*z3 - z1 - z2 - z3
                   - (K1**2*K2**2*z1**2 + 2*K1**2*K2**2*z1*z2 + K1**2*K2**2*z2**2
                      - 2*K1**2*K2*K3*z1**2 - 2*K1**2*K2*K3*z1*z2 - 2*K1**2*K2*K3*z1*z3
                      + 2*K1**2*K2*K3*z2*z3 - 2*K1**2*K2*z1*z2 + 2*K1**2*K2*z1*z3
                      - 2*K1**2*K2*z2**2 - 2*K1**2*K2*z2*z3 + K1**2*K3**2*z1**2
                      + 2*K1**2*K3**2*z1*z3 + K1**2*K3**2*z3**2 + 2*K1**2*K3*z1*z2
                      - 2*K1**2*K3*z1*z3 - 2*K1**2*K3*z2*z3 - 2*K1**2*K3*z3**2
                      + K1**2*z2**2 + 2*K1**2*z2*z3 + K1**2*z3**2 - 2*K1*K2**2*K3*z1*z2
                      + 2*K1*K2**2*K3*z1*z3 - 2*K1*K2**2*K3*z2**2 - 2*K1*K2**2*K3*z2*z3
                      - 2*K1*K2**2*z1**2 - 2*K1*K2**2*z1*z2 - 2*K1*K2**2*z1*z3
                      + 2*K1*K2**2*z2*z3 + 2*K1*K2*K3**2*z1*z2 - 2*K1*K2*K3**2*z1*z3
                      - 2*K1*K2*K3**2*z2*z3 - 2*K1*K2*K3**2*z3**2 + 4*K1*K2*K3*z1**2
                      + 4*K1*K2*K3*z1*z2 + 4*K1*K2*K3*z1*z3 + 4*K1*K2*K3*z2**2
                      + 4*K1*K2*K3*z2*z3 + 4*K1*K2*K3*z3**2 + 2*K1*K2*z1*z2
                      - 2*K1*K2*z1*z3 - 2*K1*K2*z2*z3 - 2*K1*K2*z3**2 - 2*K1*K3**2*z1**2
                      - 2*K1*K3**2*z1*z2 - 2*K1*K3**2*z1*z3 + 2*K1*K3**2*z2*z3
                      - 2*K1*K3*z1*z2 + 2*K1*K3*z1*z3 - 2*K1*K3*z2**2 - 2*K1*K3*z2*z3
                      + K2**2*K3**2*z2**2 + 2*K2**2*K3**2*z2*z3 + K2**2*K3**2*z3**2
                      + 2*K2**2*K3*z1*z2 - 2*K2**2*K3*z1*z3 - 2*K2**2*K3*z2*z3
                      - 2*K2**2*K3*z3**2 + K2**2*z1**2 + 2*K2**2*z1*z3
                      + K2**2*z3**2 - 2*K2*K3**2*z1*z2 + 2*K2*K3**2*z1*z3
                      - 2*K2*K3**2*z2**2 - 2*K2*K3**2*z2*z3 - 2*K2*K3*z1**2
                      - 2*K2*K3*z1*z2 - 2*K2*K3*z1*z3 + 2*K2*K3*z2*z3 + K3**2*z1**2
                      + 2*K3**2*z1*z2 + K3**2*z2**2)**0.5/2)
                   / (K1*K2*K3*z1 + K1*K2*K3*z2 + K1*K2*K3*z3 - K1*K2*z1
                      - K1*K2*z2 - K1*K2*z3 - K1*K3*z1 - K1*K3*z2 - K1*K3*z3
                      + K1*z1 + K1*z2 + K1*z3 - K2*K3*z1 - K2*K3*z2 - K2*K3*z3
                      + K2*z1 + K2*z2 + K2*z3 + K3*z1 + K3*z2 + K3*z3 - z1 - z2 - z3))
        return self._V

    def __repr__(self):
        return f"VLE({str(self.thermo.chemicals)})"
