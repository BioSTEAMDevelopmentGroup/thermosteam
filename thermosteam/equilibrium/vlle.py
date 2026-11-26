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
from .fugacities import LiquidFugacities, GasFugacities
from .binary_phase_fraction import solve_phase_fraction_Rashford_Rice
from ..utils import Cache
import thermosteam as tmo
import numpy as np
from scipy.optimize import root, minimize
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

def multiphase_Rachford_Rice(z, Ks, guess):
    """
    Return phase fractions of non-reference phases given partition
    coefficients relative to the reference phase (`Ks`; rows are phases) by
    minimizing the convex function F = -sum(z * ln(t)), where
    t = 1 + sum(beta * (K - 1)), subject to nonnegative phase fractions
    [1]_. Phases with zero phase fractions are absent.

    References
    ----------
    .. [1] Okuno, R.; Johns, R. T.; Sepehrnoori, K. A New Algorithm for
       Rachford-Rice for Multiphase Compositional Simulation. SPE Journal
       2010, 15 (02), 313-325.

    """
    Km1 = Ks - 1.

    def objective(beta):
        t = 1. + beta @ Km1
        if (t <= 0.).any(): return np.inf, np.zeros_like(beta)
        return -(z * np.log(t)).sum(), -Km1 @ (z / t)

    m = guess.size
    guess = np.clip(guess, 0., 1.)
    if guess.sum() > 1.: guess /= guess.sum()
    result = minimize(
        objective, guess, jac=True, method='SLSQP', bounds=[(0., 1.)] * m,
        constraints=dict(type='ineq', fun=lambda b: 1. - b.sum(), jac=lambda b: -np.ones(m)),
        options=dict(ftol=1e-14, maxiter=200),
    )
    beta = np.clip(result.x, 0., 1.)
    total = beta.sum()
    if total > 1.: beta /= total
    return beta

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

    Examples
    --------
    Vapor-liquid-liquid equilibrium of water, ethanol, and octane at a
    given temperature and pressure:

    >>> from thermosteam import indexer, equilibrium, settings
    >>> settings.set_thermo(['Water', 'Ethanol', 'Octane'], cache=True)
    >>> imol = indexer.MolarFlowIndexer(
    ...     l=[('Water', 300), ('Ethanol', 20), ('Octane', 50)],
    ...     phases=('L', 'g', 'l'),
    ... )
    >>> vlle = equilibrium.VLLE(imol)
    >>> vlle(T=360, P=101325)
    >>> vlle
    VLLE(imol=MolarFlowIndexer(
            L=[('Water', 215), ('Ethanol', 3.554), ('Octane', 0.01391)],
            g=[('Water', 84.37), ('Ethanol', 16.32), ('Octane', 38.91)],
            l=[('Water', 0.6186), ('Ethanol', 0.1283), ('Octane', 11.07)]),
        thermal_condition=ThermalCondition(T=360.00, P=101325))

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
        '_liquid_fugacities',
        '_gas_fugacities',
    )
    #: Options for Anderson acceleration of the sequential (flash-based) 
    #: iteration at given T and P. It is terminated early (loose tolerance, 
    #: few iterations, and when errors stop decreasing) to provide an initial 
    #: guess for the simultaneous correction.
    solver_options = dict(xtol=1e-2, maxiter=10, convergenceiter=3, memory=3)
    
    #: Tolerance of the simultaneous correction (Newton-type solution of 
    #: equal fugacities and Rashford-Rice equations for all phases at once).
    simultaneous_correction_tol = 1e-12

    #: Maximum number of iterations of successive substitution of partition
    #: coefficients to estimate phases at given T and P.
    successive_substitution_maxiter = 50
    
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
        self._liquid_fugacities = LiquidFugacities(chemicals, thermo)
        self._gas_fugacities = GasFugacities(chemicals, thermo)
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
        L_mol[indices] = data[0]
        g_mol[indices] = data[1]
        l_mol[indices] = data[2]
        self.iter += 1
            
        # LLE
        self.lle(T=T, P=P, top_chemical=self.top_chemical, use_cache=True)
        xa_lle = L_mol[indices].copy()
        xb_lle = l_mol[indices].copy()
        
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
        y_a = y / V0 if V0 else None
        
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
        
        if (Ka is None or Kb is None) and xa_lle.any() and xb_lle.any():
            # Both liquids exist, but a flash removed a phase (e.g., a 
            # liquid fully vaporized on its own or neither liquid boiled on
            # its own). Let the three-phase Rashford-Rice equations decide 
            # instead by computing the missing partition coefficients from 
            # the liquid compositions after LLE.
            if V: 
                y_ref = y / V
            elif y_a is not None:
                y_ref = y_a
            else: # Ideal vapor estimate from the extract
                xa_ref = xa_lle / xa_lle.sum()
                y_ref = xa_ref * self._liquid_fugacities.unweighted(xa_ref, T, P)
                y_ref /= y_ref.sum()
            f_gas = self._gas_fugacities.unweighted(y_ref, T, P)
            liquid_fugacities = self._liquid_fugacities.unweighted
            if Ka is None:
                xa = xa_lle / xa_lle.sum()
                Ka = liquid_fugacities(xa, T, P) / f_gas
                La = xa_lle.sum()
            if Kb is None:
                xb = xb_lle / xb_lle.sum()
                Kb = liquid_fugacities(xb, T, P) / f_gas
                Lb = xb_lle.sum()
            if not V: V = V0 or 0.5 * (La + Lb) # Positive guess for vapor
        
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
        # Remove phases that only hold round-off residue (flows are normalized
        # to a total of 1, so this is relative to the total flow).
        values[values.sum(axis=1) < 1e-12] = 0
        values[values < 0.] = 0.
        values *= z / values.sum(axis=0)
        return values
    
    def _set_TP(self, T, P):
        self.iter = 0
        indices = self._indices
        thermal_condition = self._thermal_condition
        thermal_condition.T = T
        thermal_condition.P = P
        L_mol, g_mol, l_mol = self._data
        
        use_points = not (self._dew_point_values is None or self._bubble_point_values is None)
        if use_points:
            # Bubble and dew points are available (e.g., for V, H, or S
            # specifications); use them to check for single phase regions.
            T_dew = self._dew_point_values.T_dew
            P_dew = self._dew_point_values.P_dew
            # Dew point may have been computed at constant T (varying P) or at
            # constant P (varying T); compare along the varying variable.
            superheated = P <= P_dew if T_dew == T else T >= T_dew
            if superheated:
                thermal_condition.T = T
                thermal_condition.P = P
                g_mol[:] += L_mol + l_mol
                L_mol.clear()
                l_mol.clear()
                return
            T_bubble = self._bubble_point_values.T_bubble
            P_bubble = self._bubble_point_values.P_bubble
            subcooled = P >= P_bubble if T_bubble == T else T <= T_bubble
            if subcooled:
                l_mol += g_mol
                g_mol[:] = 0
                self.lle(T=T, P=P, top_chemical=self.top_chemical)
                return
            else:
                thermal_condition.T = T
                thermal_condition.P = P

        z = self._z
        converged = False
        if not use_points:
            # Check for single phase regions by stability analysis and
            # start from liquid-liquid equilibrium
            result = self._initial_phase_flows_at_TP(T, P)
            if result is None: return
            subdata, converged = result
        elif(L_mol[indices].any() and g_mol[indices].any() and l_mol[indices].any()):
            subdata = np.array([
                L_mol[indices], g_mol[indices], l_mol[indices],
            ])
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
        new_subdata = self._iter_phase_flows_at_TP(subdata, converged, T, P)
        if (use_points and len(indices) == 2
            and (new_subdata.sum(axis=1) > 0.).all()):
            # By the Gibbs phase rule, three phases of two chemicals only
            # coexist at the bubble point (0 degrees of freedom), so the
            # hot start led to a spurious split. Restart by stability analysis.
            result = self._initial_phase_flows_at_TP(T, P)
            if result is None: return
            new_subdata = self._iter_phase_flows_at_TP(*result, T, P)
        if (new_subdata.sum(axis=1) > 0.).all(): # Three phases are required
            new_subdata = self._simultaneous_correction(new_subdata, T, P, z)
        self._data[:, indices] = self._total * new_subdata

    def _iter_phase_flows_at_TP(self, subdata, converged, T, P):
        z = self._z
        subdata *= z / subdata.sum(axis=0)
        if converged:
            # Successive substitution of partition coefficients converged
            return subdata
        else:
            return flx.anderson(
                self._iter_flows_at_TP, subdata,
                args=(T, P, self._indices, z),
                checkconvergence=False,
                checkiter=False,
                lb=0.,
                **self.solver_options
            )

    def _initial_phase_flows_at_TP(self, T, P):
        """
        Return initial phase flows (rows: L, g, l; normalized to a total of 1)
        for the vapor-liquid-liquid equilibrium iteration at given T and P,
        and whether they are converged (by successive substitution), or None 
        if the mixture is subcooled (liquids only) or superheated (vapor 
        only), in which case material flows are already updated.

        Single phase regions are found by stability analysis (instead of
        bubble and dew points) and the initial guess is based on
        liquid-liquid equilibrium of the overall composition.

        """
        indices = self._indices
        z = self._z
        total = self._total
        L_mol, g_mol, l_mol = self._data
        vapor_allowed, liquid_allowed = self._phase_masks(T)
        positive = z > 0
        liquid_fugacities = self._liquid_fugacities.unweighted
        gas_fugacities = self._gas_fugacities.unweighted
        if not liquid_allowed[positive].any(): # Only vapor is possible
            g_mol[indices] = total * z
            L_mol[indices] = 0.
            l_mol[indices] = 0.
            return None
        
        # Liquid-liquid equilibrium of the overall composition
        g_mol[indices] = 0.
        L_mol[indices] = 0.
        l_mol[indices] = total * z
        self.lle(T=T, P=P, top_chemical=self.top_chemical)
        liquids = [L_mol[indices] / total, l_mol[indices] / total]
        liquid_flows = [i.sum() for i in liquids]
        major = 0 if liquid_flows[0] >= liquid_flows[1] else 1
        x = liquids[major] / liquid_flows[major]
        if not vapor_allowed[positive].any(): return None # Only liquids are possible

        # Stability of liquids against a vapor: the minimum tangent plane
        # distance of a vapor is -ln(sum(f_liquid / (P * phi_vapor))).
        f_liquid = liquid_fugacities(x, T, P) * x * vapor_allowed
        y = f_liquid / P
        for i in range(3): # Fugacity coefficients of vapor depend on composition
            y = f_liquid / np.broadcast_to(gas_fugacities(y / y.sum(), T, P), y.shape)
        if y.sum() <= 0.99 and liquid_allowed[positive].all():
            return None # Subcooled; liquid-liquid equilibrium already set

        # Stability of a vapor (overall composition) against a liquid
        candidate = None
        if vapor_allowed[positive].all():
            lnf_vapor = np.log(np.maximum(z, 1e-300)) + np.log(np.maximum(
                np.broadcast_to(gas_fugacities(z, T, P), z.shape), 1e-300
            ))
            index = np.flatnonzero(liquid_allowed & positive)

            def tpd(w_index): # Tangent plane distance and log fugacity difference
                w = np.zeros(z.size)
                w[index] = w_index
                g = (np.log(w_index)
                     + np.log(np.maximum(liquid_fugacities(w, T, P)[index], 1e-300))
                     - lnf_vapor[index])
                return w_index @ g, g

            def objective(v): # Softmax parameterization (gradient by Gibbs-Duhem)
                w_index = np.exp(v - v.max())
                w_index /= w_index.sum()
                distance, g = tpd(w_index)
                return distance, w_index * (g - distance)

            N = index.size
            samples = np.full([N, N], 1e-3)
            np.fill_diagonal(samples, 1.)
            samples = [i[index] for i in liquids if i.sum() > 0] + [z[index], *samples]
            samples = [np.maximum(i, 1e-12) / np.maximum(i, 1e-12).sum() for i in samples]
            stable = True
            for sample in samples: # Unstable if any sample has negative distance
                if tpd(sample)[0] < 0.:
                    stable = False
                    candidate = sample
                    break
            if stable:
                for sample in samples:
                    result = minimize(
                        objective, np.log(sample), method='L-BFGS-B', jac=True,
                        options=dict(maxiter=50),
                    )
                    if result.fun < -1e-10:
                        stable = False
                        candidate = np.exp(result.x - result.x.max())
                        candidate /= candidate.sum()
                        break
            if stable: # Superheated
                g_mol[indices] = total * z
                L_mol[indices] = 0.
                l_mol[indices] = 0.
                return None

        # Initial guess by successive substitution of partition coefficients
        # starting from liquid-liquid equilibrium
        
        def successive_substitution(xs, y, beta):
            error = np.inf
            for iteration in range(self.successive_substitution_maxiter):
                f_vapor = np.broadcast_to(gas_fugacities(y, T, P), y.shape) # P * phi
                Ks = np.array([f_vapor / liquid_fugacities(i, T, P) for i in xs]) # Liquid over vapor
                Ks[:, ~vapor_allowed] = 1e16
                Ks[:, ~liquid_allowed] = 1e-16
                beta = multiphase_Rachford_Rice(z, Ks, beta)
                y_new = z / (1. + beta @ (Ks - 1.))
                xs = Ks * y_new
                xs /= xs.sum(axis=1, keepdims=True)
                y_new /= y_new.sum()
                error = np.abs(y_new - y).max()
                y = y_new
                if error < 1e-9: break
            return xs, y, beta, error < 1e-9
        
        xs = np.array([i / Li if Li > 0 else x for i, Li in zip(liquids, liquid_flows)])
        xs, y, beta, converged = successive_substitution(xs, y / y.sum(), 0.5 * np.array(liquid_flows))
        if beta.sum() <= 1e-12 and candidate is not None:
            # Converged to a vapor although a liquid can form (by stability 
            # analysis); restart from the most stable liquid.
            w = np.zeros(z.size)
            w[index] = candidate
            other = xs[np.argmax([np.abs(i - w).max() for i in xs])]
            xs, y, beta, converged = successive_substitution(
                np.array([w, other]), z.copy(), np.array([0.05, 0.])
            )
        V = 1. - beta.sum()
        data = np.array([xs[0] * beta[0], y * V, xs[1] * beta[1]])
        if np.abs(xs[0] - xs[1]).max() < 1e-6: # Identical liquids are one phase
            data[2] += data[0]
            data[0] = 0.
        elif not beta[1]: # A single liquid is in the 'l' phase
            data[2] = data[0]
            data[0] = 0.
        return data, converged

    def _phase_masks(self, T):
        """
        Return masks of components allowed in the vapor (not heavy, i.e.,
        locked in a condensed phase) and in liquids (not supercritical,
        consistent with VLE).

        """
        chemicals = self._liquid_fugacities.chemicals
        heavy = np.array([bool(i.locked_state) for i in chemicals])
        Tc = np.array([np.inf if i.Tc is None else i.Tc for i in chemicals])
        light = (T >= Tc) & ~heavy
        return ~heavy, ~light

    def _log_fugacity_coefficients(self, i, x, T, P):
        """
        Return the log of fugacities over mol fractions of phase `i`
        (0: L, 1: g, 2: l).

        """
        x = x / x.sum()
        if i == 1:
            f = self._gas_fugacities.unweighted(x, T, P)
        else:
            f = self._liquid_fugacities.unweighted(x, T, P)
        f = np.broadcast_to(f, x.shape) # Ideal gas returns a scalar
        return np.log(np.maximum(f, 1e-300))

    # def _tangent_plane_minima(self, data, T, P, z, phase):
    #     """
    #     Return local minima of the tangent plane distance (relative to the
    #     phases in `data`) of a trial phase of the given kind (1: vapor;
    #     0 or 2: liquid) as a list of (distance, composition) tuples sorted
    #     from lowest to highest distance.

    #     """
    #     flows = data.sum(axis=1)
    #     vapor_allowed, liquid_allowed = self._phase_masks(T)
    #     allowed = lambda i: vapor_allowed if i == 1 else liquid_allowed
    #     positive = z > 0
    #     # All existing phases share the same fugacities at equilibrium; use
    #     # the largest one that may hold every component as reference.
    #     existing = [i for i in range(3) if flows[i] > 0]
    #     candidates = [i for i in existing if allowed(i)[positive].all()] or existing
    #     ref = max(candidates, key=lambda i: flows[i])
    #     x_ref = data[ref] / flows[ref]
    #     with np.errstate(divide='ignore'):
    #         lnf_ref = np.log(x_ref) + self._log_fugacity_coefficients(ref, x_ref, T, P)
    #     index = np.flatnonzero(allowed(phase) & positive)
    #     N = index.size

    #     def composition(w_index):
    #         w = np.zeros(z.size)
    #         w[index] = w_index
    #         return w

    #     def tpd(v): # Tangent plane distance (softmax parameterization)
    #         w_index = np.exp(v - v.max())
    #         w_index /= w_index.sum()
    #         lnphi = self._log_fugacity_coefficients(phase, composition(w_index), T, P)
    #         return w_index @ (np.log(w_index) + lnphi[index] - lnf_ref[index])

    #     # Near-pure trial phases (a trace component may form the
    #     # incipient phase) and the overall composition
    #     samples = np.full([N, N], 1e-3)
    #     np.fill_diagonal(samples, 1.)
    #     samples = np.vstack([samples, z[index]])
    #     samples /= samples.sum(axis=1, keepdims=True)
    #     minima = []
    #     for sample in samples:
    #         result = minimize(
    #             tpd, np.log(sample), method='L-BFGS-B',
    #             options=dict(maxiter=100),
    #         )
    #         w_index = np.exp(result.x - result.x.max())
    #         w = composition(w_index / w_index.sum())
    #         if any([np.abs(w - i).max() < 1e-3 for _, i in minima]): continue
    #         minima.append((result.fun, w))
    #     minima.sort(key=lambda i: i[0])
    #     return minima

    # def _add_missing_phases(self, data, T, P, z):
    #     """
    #     Return phase flows with all missing phases formed by stability 
    #     analysis: each is seeded with a small amount at the composition that
    #     minimizes the tangent plane distance (the phase most likely to form).

    #     """
    #     flows = data.sum(axis=1)
    #     new_data = data.copy()
    #     liquids = [data[i] / flows[i] for i in (0, 2) if flows[i] > 0]
    #     for row in (1, 0, 2):
    #         if flows[row] > 0: continue
    #         minima = self._tangent_plane_minima(data, T, P, z, row)
    #         if row != 1: # Must be distinct from other liquids
    #             minima = [i for i in minima if all([np.abs(i[1] - x).max() >= 1e-3 for x in liquids])]
    #         if not minima: return data
    #         w = minima[0][1]
    #         if row != 1: liquids.append(w)
    #         # Seed the new phase with a small amount taken proportionally
    #         # from existing phases (preserving the mass balance).
    #         mask = w > 0
    #         amount = min(1e-3, 0.5 * (z[mask] / w[mask]).min())
    #         take = amount * w
    #         new_data *= 1. - take / np.where(z > 0, z, 1.)
    #         new_data[row] += take
    #     return new_data

    def _simultaneous_correction(self, data, T, P, z):
        """
        Return phase flows (rows: L, g, l; normalized to a total of 1) of
        three phases solved simultaneously for equal fugacities and mass
        balances, or `data` if the correction is not successful.

        Unknowns are the logarithm of partition coefficients of both liquids
        relative to the vapor and phase fractions of both liquids. The
        correction is only accepted if it yields a valid (non-trivial) split
        with lower Gibbs free energy than `data`.

        """
        vapor = 1
        liquids = (0, 2)
        n = z.size
        vapor_allowed, liquid_allowed = self._phase_masks(T)
        free = np.array([vapor_allowed & liquid_allowed] * 2)
        lnK_big = np.log(1e16)
        lnK_fixed = np.where(liquid_allowed, lnK_big, -lnK_big) * np.ones([2, n])
        N_free = free.sum()
        lnphi = self._log_fugacity_coefficients

        def unpack(u):
            lnK = lnK_fixed.copy()
            lnK[free] = u[:N_free]
            beta = u[N_free:]
            beta_vapor = 1. - beta.sum()
            # Invalid trial steps (e.g., overflow) are rejected by the residuals
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                K = np.exp(lnK)
                y = z / (beta_vapor + beta @ K)
                xs = K * y
            return lnK, beta, beta_vapor, y, xs

        def residuals(u):
            lnK, beta, beta_vapor, y, xs = unpack(u)
            if (y <= 0).any() or not np.isfinite(y).all():
                return np.full(u.size, 1e12)
            lnphi_vapor = lnphi(vapor, y, T, P)
            r = [
                (lnK[k] - lnphi_vapor + lnphi(j, xs[k], T, P))[free[k]]
                for k, j in enumerate(liquids)
            ]
            r.append(xs.sum(axis=1) - y.sum())
            return np.concatenate(r)

        def gibbs_energy(data): # Dimensionless [G/RT]
            G = 0.
            for i in range(3):
                n_i = data[i]
                mask = n_i > 0 # Zero flows do not contribute (n ln x -> 0)
                if not mask.any(): continue
                x = n_i / n_i.sum()
                lnf = np.log(x[mask]) + lnphi(i, x, T, P)[mask]
                G += (n_i[mask] * lnf).sum()
            return G

        flows = data.sum(axis=1) # Phase fractions
        x = data / flows[:, None]
        y = np.maximum(x[vapor], 1e-32)
        lnK0 = np.array([np.log(np.maximum(x[j], 1e-32) / y) for j in liquids])
        u0 = np.concatenate([lnK0[free], flows[list(liquids)]])
        sol = root(residuals, u0, method='hybr', tol=self.simultaneous_correction_tol)
        r = residuals(sol.x)
        if not np.isfinite(r).all() or np.abs(r).max() > 1e-8: return data
        lnK, beta, beta_vapor, y, xs = unpack(sol.x)
        if beta_vapor <= 1e-12 or (beta <= 1e-12).any(): return data
        xa, xb = xs
        if np.abs(xa / xa.sum() - xb / xb.sum()).max() < 1e-6: return data # Trivial solution
        new_data = np.array([beta[0] * xa, beta_vapor * y, beta[1] * xb])
        # Equilibrium minimizes the Gibbs free energy
        if gibbs_energy(new_data) > gibbs_energy(data) + 1e-9: return data
        return new_data

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
        P, x = dew_point.solve_Px(self._z, T, lle=True)
        data_copy = self._data.copy()
        L_mol, g_mol, l_mol = self._data
        L_mol[self._indices] = x
        l_mol.clear()
        g_mol.clear()
        self.lle(
            T=T, P=P, # Check liquid stability at the dew point
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
        T, x = dew_point.solve_Tx(self._z, P, lle=True)
        data_copy = self._data.copy()
        L_mol, g_mol, l_mol = self._data
        L_mol[self._indices] = x
        l_mol.clear()
        g_mol.clear()
        self.lle(
            T=T, P=P, # Check liquid stability at the dew point
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
    
    def _invariant_three_phase_region(self):
        """
        Return the compositions of the extract, vapor, and raffinate
        (normalized over VLLE chemicals) and the maximum vapor fraction
        (normalized to a total of 1) of the three-phase region at the bubble
        point, or None if there are no invariant three phases.

        Must be called right after the bubble point. With 2 chemicals and
        3 phases at a given pressure (or temperature), there are 0 degrees of
        freedom: temperature (or pressure) and phase compositions are fixed
        at the bubble point while the vapor fraction varies over a range.
        Thus, the vapor fraction (and the enthalpy) is discontinuous with
        respect to temperature (or pressure) and must be solved for directly.

        """
        indices = self._indices
        if len(indices) != 2: return None
        data = self._data
        xa = data[0, indices]
        xb = data[2, indices]
        La = xa.sum()
        Lb = xb.sum()
        if La <= 0. or Lb <= 0.: return None
        xa = xa / La
        xb = xb / Lb
        y = self._yx_bubble[0]
        y = y / y.sum()
        k = np.abs(xa - xb).argmax()
        dx = xa[k] - xb[k]
        if abs(dx) < 1e-6: return None # Identical liquids
        z = self._z[k]
        La0 = (z - xb[k]) / dx # Extract fraction without vapor
        if not (1e-12 < La0 < 1 - 1e-12): return None # Outside of miscibility gap
        # Vapor fractions where the extract (Va) or raffinate (Vb) vanishes
        Vs = []
        for x in (xb, xa):
            dy = y[k] - x[k]
            if dy: Vs.append((z - x[k]) / dy)
        Vs = [i for i in Vs if 0. < i <= 1.]
        if not Vs: return None
        return xa, y, xb, min(Vs)

    def _set_invariant_three_phase(self, region, V):
        """
        Set material flows at a vapor fraction (normalized to a total of 1)
        within the invariant three-phase region (by mass balance).

        """
        xa, y, xb, Vmax = region
        k = np.abs(xa - xb).argmax()
        La = (self._z[k] - V * y[k] - (1 - V) * xb[k]) / (xa[k] - xb[k])
        Lb = 1. - V - La
        values = np.array([La * xa, V * y, Lb * xb])
        values[values < 0.] = 0.
        values *= self._z / values.sum(axis=0)
        self._data[:, self._indices] = self._total * values

    def _solve_invariant_three_phase(self, region, f, target):
        """
        Solve for the vapor fraction within the invariant three-phase region
        that satisfies `f() = target`, where `f` is linear with respect
        to phase flows at constant phase compositions (e.g., vapor
        fraction and enthalpy). Return True if successful, or False if the
        target is above the three-phase region, in which case material flows
        are set at the upper bound of the three-phase region.

        """
        Vmax = region[-1]
        self._set_invariant_three_phase(region, 0.)
        f0 = f()
        self._set_invariant_three_phase(region, Vmax)
        f1 = f()
        if target > f1: return False
        V = Vmax * (target - f0) / (f1 - f0) if f1 != f0 else 0.
        self._set_invariant_three_phase(region, min(max(V, 0.), Vmax))

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

        # Check if within invariant three-phase region (0 degrees of freedom)
        region = self._invariant_three_phase_region()
        if region is not None:
            xH = lambda: self.mixture.xH(phase_data, T_bubble, P)
            self._solve_invariant_three_phase(region, xH, H)
            thermal_condition.T = T_bubble
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
            self._H_hat_err_at_T, max(2 * T0 - T1, T_bubble), T1,
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

        # Check if within invariant three-phase region (0 degrees of freedom)
        region = self._invariant_three_phase_region()
        if region is not None:
            xH = lambda: self.mixture.xH(phase_data, T, P_bubble)
            self._solve_invariant_three_phase(region, xH, H)
            thermal_condition.P = P_bubble
            return

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

        # Check if within invariant three-phase region (0 degrees of freedom)
        region = self._invariant_three_phase_region()
        if region is not None:
            xS = lambda: self.mixture.xS(phase_data, T_bubble, P)
            self._solve_invariant_three_phase(region, xS, S)
            thermal_condition.T = T_bubble
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
        S_dew = self.mixture.xS(phase_data, T, P_dew)
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

        # Check if within invariant three-phase region (0 degrees of freedom)
        region = self._invariant_three_phase_region()
        if region is not None:
            xS = lambda: self.mixture.xS(phase_data, T, P_bubble)
            self._solve_invariant_three_phase(region, xS, S)
            thermal_condition.P = P_bubble
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
            # Check if within invariant three-phase region (0 degrees of freedom)
            V_bubble = 0.
            region = self._invariant_three_phase_region()
            if region is not None:
                self._solve_invariant_three_phase(region, lambda: self.vapor_fraction, V)
                thermal_condition.T = T_bubble
                return
            T_dew = self.dew_point_at_P(P)
            flx.IQ_interpolation(
                self._V_hat_err_at_T, T_bubble, T_dew,
                V_bubble - V, 1. - V, xtol=1e-6, ytol=1e-6, args=(V,),
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
            # Check if within invariant three-phase region (0 degrees of freedom)
            region = self._invariant_three_phase_region()
            if region is not None:
                self._solve_invariant_three_phase(region, lambda: self.vapor_fraction, V)
                thermal_condition.P = P_bubble
                return
            P_dew = self.dew_point_at_T(T)
            flx.IQ_interpolation(
                self._V_hat_err_at_P, P_bubble, P_dew, 
                x=thermal_condition.P, xtol=1, ytol=1e-6, args=(V,),
                checkiter=False, checkbounds=False,
                maxiter=100,
            )
    
class VLLECache(Cache): load = VLLE
del Cache, Equilibrium

