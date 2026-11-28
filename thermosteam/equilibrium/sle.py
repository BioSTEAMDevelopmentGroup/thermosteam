# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
Solid-liquid equilibrium of simple eutectic systems (i.e., each crystallizing
chemical forms a pure, immiscible solid phase).

At equilibrium, the activity of each chemical *i* in the liquid satisfies
[1]_ [2]_:

.. math::

    x_i \\gamma_i \\leq \\psi_i(T) = \\exp\\left(-\\frac{\\Delta G_{fus,i}(T)}{RT}\\right)

with equality if a pure solid of *i* is present, where (assuming a constant
heat capacity of fusion, :math:`\\Delta C_{p,i}`, and neglecting solid-solid
transitions and the effect of pressure [1]_):

.. math::

    \\ln \\psi_i = -\\frac{\\Delta H_{fus,i}}{RT}\\left(1 - \\frac{T}{T_{m,i}}\\right)
    + \\frac{\\Delta C_{p,i}}{R}\\left(\\frac{T_{m,i}}{T} - 1 - \\ln\\frac{T_{m,i}}{T}\\right)

By default, :math:`\\Delta C_{p,i} = 0`, which is the most common
approximation in literature [1]_ [2]_ and generally agrees well with
experimental data (e.g., within 1 K for the liquidus of the
benzene-naphthalene system). A user-specified :math:`\\Delta C_{p,i}` may be
passed for each chemical (e.g., from measured liquid and solid heat
capacities near the melting point).

The set of solid phases present is found by an active-set method [3]_: pure
solids are added one at a time (most supersaturated first) based on their
stability (:math:`d_i = \\ln(x_i \\gamma_i / \\psi_i) > 0`) and removed once
fully dissolved. Given a set of solids, the liquid is solved by successive
substitution of activity coefficients with a closed-form mass balance.

References
----------
.. [1] Prausnitz, J. M.; Lichtenthaler, R. N.; de Azevedo, E. G. Molecular
   Thermodynamics of Fluid-Phase Equilibria, 3rd ed.; Prentice Hall, 1999;
   Chapter 11.
.. [2] Gmehling, J.; Kolbe, B.; Kleiber, M.; Rarey, J. Chemical
   Thermodynamics for Process Simulation; Wiley-VCH: Weinheim, 2012;
   Chapter 8.
.. [3] Michelsen, M. L.; Mollerup, J. M. Thermodynamic Models: Fundamentals
   & Computational Aspects, 2nd ed.; Tie-Line Publications, 2007.

"""
import flexsolve as flx
import numpy as np
from math import log
from ..utils import Cache
from ..constants import R
from .equilibrium import Equilibrium
from .activity_coefficients import IdealActivityCoefficients

__all__ = (
    'SLE', 'SLECache',
    'log_ideal_solubility',
    'liquid_mol_at_saturation',
)

def log_ideal_solubility(chemical, T, dCp=0.):
    """
    Return the natural logarithm of the ideal solubility (i.e., the activity
    of the pure subcooled liquid relative to the pure solid),
    :math:`\\ln\\psi = -\\Delta G_{fus}(T)/RT`.

    Parameters
    ----------
    chemical : :class:`~thermosteam.Chemical`
        Crystallizing chemical; must have a melting point, `Tm`, and a heat
        of fusion, `Hfus`.
    T : float
        Temperature [K].
    dCp : float, optional
        Heat capacity of fusion (liquid minus solid heat capacity) [J/mol/K].
        Defaults to 0.

    Examples
    --------
    Ideal solubility of naphthalene at 25 C (experimental solubility in
    benzene, a nearly ideal solvent, is 0.292 [1]_):

    >>> from thermosteam import Chemical
    >>> from thermosteam.equilibrium import log_ideal_solubility
    >>> from math import exp
    >>> naphthalene = Chemical('Naphthalene')
    >>> round(exp(log_ideal_solubility(naphthalene, 298.15)), 3)
    0.3

    References
    ----------
    .. [1] Chang, S. Ph.D. Thesis, 1969; as compiled in the IUPAC-NIST
       Solubility Database (NIST SRD 106).

    """
    Tm = chemical.Tm
    log_psi = chemical.Hfus * (T - Tm) / (R * T * Tm)
    if dCp:
        r = Tm / T
        log_psi += dCp * (r - 1. - log(r)) / R
    return log_psi

def liquid_mol_at_saturation(z, a, active):
    """
    Return the liquid molar flow rates given total molar flow rates, `z`,
    and target liquid mol fractions, `a`, of chemicals saturated with a pure
    solid phase (`active`). Chemicals not saturated remain in the liquid.
    A saturated chemical that would fully dissolve remains in the liquid.

    The liquid flow rate, L, is the largest root of the piecewise linear
    (concave) mass balance:

    .. math::

        L = \\sum_{j \\notin active} z_j + \\sum_{i \\in active} \\min(a_i L, z_i)

    A liquid flow rate of zero (i.e., complete solidification) is returned
    when all chemicals are saturated and the sum of `a` is less than one
    (e.g., below the eutectic temperature).

    """
    free = z.sum() - z[active].sum()
    a = a[active]
    z_active = z[active]
    slope = a.sum() # dL_dissolved / dL when no chemical is fully dissolved
    breakpoints = z_active / a
    order = np.argsort(breakpoints)
    if free <= 0. and slope <= 1.:
        L = 0.
    else:
        # Walk the linear segments in order of increasing L
        constant = free
        L = None
        for i in order:
            # Within segment: L = constant + slope * L
            if slope < 1.:
                L_segment = constant / (1. - slope)
                if L_segment <= breakpoints[i]:
                    L = L_segment
                    break
            # Chemical i fully dissolves past its breakpoint
            constant += z_active[i]
            slope -= a[i]
        if L is None: L = constant / (1. - slope) if slope < 1. else constant
    liquid_mol = z.copy()
    liquid_mol[active] = np.minimum(a * L, z_active)
    return liquid_mol


def solve_monotonic_bracket(f, x0, x1, y0, y1, x, xtol, ytol, maxiter, args=()):
    """
    Solve f(x) = 0 for a monotonically increasing function which may be
    discontinuous, given a bracket [x0, x1] where y0 = f(x0) < 0 < y1 = f(x1).
    Return the final bracket and whether the solution converged on the cold
    side (-1), hot side (1), or not at all due to a discontinuity (0).
    Inverse linear interpolation (Illinois algorithm) is used, falling back
    to bisection when the bracket does not shrink fast enough.

    """
    side = 0
    for _ in range(maxiter):
        if not x0 < x < x1: x = 0.5 * (x0 + x1)
        y = f(x, *args)
        if abs(y) < ytol: return (x, x, 1)
        width = x1 - x0
        if y < 0.:
            x0, y0 = x, y
            if side == -1: y1 *= 0.5
            side = -1
        else:
            x1, y1 = x, y
            if side == 1: y0 *= 0.5
            side = 1
        if x1 - x0 < xtol: break
        x = x0 - y0 * (x1 - x0) / (y1 - y0)
        if (x1 - x0) > 0.5 * width: x = 0.5 * (x0 + x1) # Bisect
    return (x0, x1, 0)


class SLE(Equilibrium, phases='ls'):
    """
    Create an SLE object that performs solid-liquid equilibrium when called.
    Each crystallizing chemical is assumed to form a pure solid phase (i.e.,
    simple eutectic behavior; solid solutions are not considered). Liquid
    nonidealities are accounted for with the activity coefficient model of
    the property package (`thermo.Gamma`).

    Parameters
    ----------
    imol=None : :class:`~thermosteam.indexer.MaterialIndexer`, optional
        Molar chemical phase data is stored here.
    thermal_condition=None : :class:`~thermosteam.ThermalCondition`, optional
        The temperature and pressure used in calculations are stored here.
    thermo=None : :class:`~thermosteam.Thermo`, optional
        Themodynamic property package for equilibrium calculations.
        Defaults to `thermosteam.settings.get_thermo()`.
    activity_coefficient=None : float, optional
        Activity coefficient of solutes in the liquid; only used if
        `thermo.Gamma` is :class:`~thermosteam.equilibrium.activity_coefficients.IdealActivityCoefficients`.
    dCp=None : float or dict[str, float], optional
        Heat capacity of fusion (liquid minus solid heat capacity) [J/mol/K]
        of all crystallizing chemicals (float) or by chemical ID (dict).
        Defaults to 0.

    Examples
    --------
    Solubility of naphthalene in toluene at 25 C (experimental mol fraction
    of 0.287 [1]_; 0.296 predicted):

    >>> from thermosteam import indexer, equilibrium, settings
    >>> settings.set_thermo(['Toluene', 'Naphthalene'], cache=True)
    >>> imol = indexer.MolarFlowIndexer(
    ...     l=[('Toluene', 10), ('Naphthalene', 10)],
    ...     phases=('s', 'l')
    ... )
    >>> sle = equilibrium.SLE(imol)
    >>> sle(T=298.15)
    >>> sle
    SLE(imol=MolarFlowIndexer(
            l=[('Toluene', 10), ('Naphthalene', 4.203)],
            s=[('Naphthalene', 5.797)]),
        thermal_condition=ThermalCondition(T=298.15, P=101325))
    >>> sle.solids
    ('Naphthalene',)

    Below the eutectic temperature, both chemicals crystallize:

    >>> H = sle.mixture.xH(imol, 298.15, 101325) # For later
    >>> sle(T=150)
    >>> sle
    SLE(imol=MolarFlowIndexer(phases=('l', 's'),
            s=[('Toluene', 10), ('Naphthalene', 10)]),
        thermal_condition=ThermalCondition(T=150.00, P=101325))
    >>> sle.solids
    ('Toluene', 'Naphthalene')

    The temperature may be solved given the enthalpy:

    >>> sle(H=H)
    >>> sle
    SLE(imol=MolarFlowIndexer(
            l=[('Toluene', 10), ('Naphthalene', 4.203)],
            s=[('Naphthalene', 5.797)]),
        thermal_condition=ThermalCondition(T=298.15, P=101325))

    The liquidus temperature (first crystals form upon cooling) and the
    solidus temperature (eutectic point, last liquid crystallizes upon
    cooling) are also available:

    >>> round(sle.T_liquidus(), 1)
    319.5
    >>> round(sle.T_solidus(), 1)
    179.1

    Solid-liquid equilibrium of a pure chemical:

    >>> settings.set_thermo(['Tetradecanol'], cache=True)
    >>> imol = indexer.MolarFlowIndexer(l=[('Tetradecanol', 30)], phases=('s', 'l'))
    >>> sle = equilibrium.SLE(imol)
    >>> sle(T=300) # Under melting point given T
    >>> sle
    SLE(imol=MolarFlowIndexer(phases=('l', 's'),
            s=[('Tetradecanol', 30)]),
        thermal_condition=ThermalCondition(T=300.00, P=101325))
    >>> sle(T=320) # Over melting point given T
    >>> sle
    SLE(imol=MolarFlowIndexer(phases=('l', 's'),
            l=[('Tetradecanol', 30)]),
        thermal_condition=ThermalCondition(T=320.00, P=101325))
    >>> H_solid = sle.mixture.H('s', imol['l'], 312.65, 101325)
    >>> H_liquid = sle.mixture.H('l', imol['l'], 312.65, 101325)
    >>> sle(H=0.5 * (H_solid + H_liquid)) # At melting point given H
    >>> sle
    SLE(imol=MolarFlowIndexer(
            l=[('Tetradecanol', 15)],
            s=[('Tetradecanol', 15)]),
        thermal_condition=ThermalCondition(T=312.65, P=101325))

    The solubility (mol fraction) of a solute may also be specified directly:

    >>> from thermosteam import Chemical
    >>> settings.set_thermo(['Water', 'Glucose'], cache=True)
    >>> imol = indexer.MolarFlowIndexer(l=[('Water', 10), ('Glucose', 1)], phases=('s', 'l'))
    >>> sle = equilibrium.SLE(imol)
    >>> sle(T=298.15, solute='Glucose', solubility=0.0833)
    >>> sle
    SLE(imol=MolarFlowIndexer(
            l=[('Water', 10), ('Glucose', 0.9087)],
            s=[('Glucose', 0.09131)]),
        thermal_condition=ThermalCondition(T=298.15, P=101325))

    References
    ----------
    .. [1] Cui, Y. et al. J. Chem. Eng. Data 2009, 54, 1065.

    """
    __slots__ = (
        'activity_coefficient',
        'dCp',
        'solids', # tuple[str] IDs of chemicals with a pure solid phase (results).
        '_index', # list[int] Indices of chemicals in equilibrium.
        '_sle_chemicals', # tuple[Chemical] Chemicals in equilibrium.
        '_candidates', # 1d array[bool] Chemicals which may crystallize.
        '_gamma', # [ActivityCoefficients] Activity coefficients of liquid.
        '_z', # 1d array Total molar flow rates of chemicals in equilibrium.
        '_active', # 1d array[bool] Chemicals with a pure solid phase.
        '_solubility', # 1d array Specified solubilities (nan if not specified).
    )
    #: Tolerance for the stability of pure solid phases (log of
    #: supersaturation ratio).
    stability_tol = 1e-9

    #: Maximum number of times solid phases can be added or removed.
    maxiter_active_set = 50

    #: Options for successive substitution of activity coefficients.
    solver_options = dict(
        xtol=1e-12, rtol=1e-12, maxiter=200, checkiter=False,
        checkconvergence=False, convergenceiter=5,
    )

    #: Lowest temperature bound as a fraction of the lowest melting point.
    T_min_factor = 0.5

    #: Temperature tolerance [K] for solving temperature given enthalpy.
    T_tol = 1e-9

    #: Relative enthalpy tolerance for solving temperature given enthalpy.
    H_tol = 1e-9

    #: Maximum number of iterations for solving temperature given enthalpy.
    maxiter_temperature = 200

    def __init__(self, imol=None, thermal_condition=None, thermo=None,
                 activity_coefficient=None, dCp=None):
        super().__init__(imol, thermal_condition, thermo)
        self.activity_coefficient = activity_coefficient
        self.dCp = dCp
        self.solids = ()
        self._sle_chemicals = None
        self._active = None

    def __call__(self, solute=None, *, T=None, P=None, H=None, Q=None,
                 solubility=None):
        """
        Perform solid-liquid equilibrium.

        Parameters
        ----------
        solute : str or Iterable[str], optional
            Chemicals that may crystallize. Defaults to all chemicals in
            equilibrium with a melting point and heat of fusion.
        T : float, optional
            Operating temperature [K].
        P : float, optional
            Operating pressure [Pa].
        H : float, optional
            Enthalpy [kJ/hr].
        Q : float, optional
            Duty [kJ/hr].
        solubility : float, optional
            Mol fraction of solute in the liquid at saturation. If given,
            the activity coefficient model is not used and only one solute
            may be passed.

        Notes
        -----
        You may only specify one of the following parameters: T, H, Q.

        """
        if Q is not None:
            if H is not None: raise ValueError('cannot specify both H and Q')
            H = Q + self.mixture.xH(self.imol, *self.thermal_condition)
        T_spec = T is not None
        H_spec = H is not None
        if T_spec == H_spec:
            raise ValueError("must pass one and only one of the following "
                             "specifications: T, H, Q")
        thermal_condition = self.thermal_condition
        if P is None:
            P = thermal_condition.P
        else:
            thermal_condition.P = P
        if not self._setup(solute, solubility):
            if T_spec: thermal_condition.T = T
            else: thermal_condition.T = self.mixture.xsolve_T_at_HP(self.imol, H, thermal_condition.T, P)
            return
        if T_spec:
            thermal_condition.T = T
            self._set_T(T)
        else:
            self._set_H(H, P)

    @property
    def solid_fraction(self):
        """[float] Molar fraction of chemicals in equilibrium that are solid."""
        imol = self.imol
        s = imol['s'].sum()
        return s / (s + imol['l'].sum())

    def _solid_fraction_at_T(self, T):
        liquid_mol = self._solve_TP(T)
        F_mol = self._z.sum()
        return (F_mol - liquid_mol.sum()) / F_mol

    def _temperature_bounds(self):
        Tms = [i.Tm for i, j in zip(self._sle_chemicals, self._candidates) if j]
        return self.T_min_factor * min(Tms), max(Tms) + 1.

    def _phase_boundary(self, f, T_min, T_max):
        # Find the highest temperature where f(T) is True (assuming f is
        # True below and False above the boundary). The cold bound is found
        # by stepping down from the hot bound with increasing steps to prevent
        # evaluating models far from their range of validity.
        if f(T_max): return T_max
        step = 1.
        T_hot = T_max
        T_cold = T_max - step
        while not f(T_cold):
            T_hot = T_cold
            if T_cold <= T_min: return T_min
            step *= 2.
            T_cold = max(T_cold - step, T_min)
        while T_hot - T_cold > self.T_tol:
            T = 0.5 * (T_cold + T_hot)
            if f(T): T_cold = T
            else: T_hot = T
        return 0.5 * (T_cold + T_hot)

    def T_liquidus(self, solute=None):
        """
        Return the temperature [K] at which the first solid crystallizes upon
        cooling (i.e., the highest temperature at which a solid is present).
        Material flows and thermal conditions are not changed.

        Examples
        --------
        Benzene-naphthalene system (experimental liquidus of 301.7 K at a
        naphthalene mol fraction of 0.332 [1]_):

        >>> from thermosteam import indexer, equilibrium, settings
        >>> settings.set_thermo(['Benzene', 'Naphthalene'], cache=True)
        >>> imol = indexer.MolarFlowIndexer(
        ...     l=[('Benzene', 0.668), ('Naphthalene', 0.332)],
        ...     phases=('s', 'l')
        ... )
        >>> sle = equilibrium.SLE(imol)
        >>> round(sle.T_liquidus(), 1)
        302.1

        References
        ----------
        .. [1] Kravchenko, V. M. Zh. Fiz. Khim. 1939, 13, 133; as compiled in
           the IUPAC-NIST Solubility Database (NIST SRD 106).

        """
        if not self._setup(solute, None): return None
        return self._phase_boundary(
            lambda T: self._solid_fraction_at_T(T) > 0., *self._temperature_bounds()
        )

    def T_solidus(self, solute=None):
        """
        Return the temperature [K] at which the last liquid crystallizes upon
        cooling (e.g., the eutectic temperature). If a liquid remains at all
        temperatures (e.g., due to chemicals that do not crystallize), the
        lower temperature bound is returned. Material flows and thermal
        conditions are not changed.

        Examples
        --------
        Eutectic temperature of the benzene-naphthalene system (experimental
        value of 269.6 K at a naphthalene mol fraction of 0.134 [1]_):

        >>> from thermosteam import indexer, equilibrium, settings
        >>> settings.set_thermo(['Benzene', 'Naphthalene'], cache=True)
        >>> imol = indexer.MolarFlowIndexer(
        ...     l=[('Benzene', 0.5), ('Naphthalene', 0.5)],
        ...     phases=('s', 'l')
        ... )
        >>> sle = equilibrium.SLE(imol)
        >>> round(sle.T_solidus(), 1)
        269.7

        References
        ----------
        .. [1] Kravchenko, V. M. Zh. Fiz. Khim. 1939, 13, 133; as compiled in
           the IUPAC-NIST Solubility Database (NIST SRD 106).

        """
        if not self._setup(solute, None): return None
        return self._T_solidus()

    def _T_solidus(self):
        T_min, T_max = self._temperature_bounds()
        T_liquidus = self._phase_boundary(
            lambda T: self._solid_fraction_at_T(T) > 0., T_min, T_max
        )
        return self._phase_boundary(
            lambda T: self._solid_fraction_at_T(T) >= 1., T_min, T_liquidus
        )

    def _setup(self, solute, solubility):
        imol = self.imol
        liquid_mol = imol['l']
        solid_mol = imol['s']
        chemicals = self.chemicals
        mol = liquid_mol + solid_mol
        index = chemicals.get_lle_indices(mol.nonzero_keys())
        if not index: return False
        z = mol[index]
        all_chemicals = chemicals.tuple
        sle_chemicals = tuple([all_chemicals[i] for i in index])
        if solute is None:
            candidates = np.array(
                [bool(i.Tm and i.Hfus) for i in sle_chemicals]
            )
        else:
            if isinstance(solute, str): solute = (solute,)
            solute = set([chemicals[i].ID for i in solute])
            candidates = np.array([i.ID in solute for i in sle_chemicals])
            for i, j in zip(sle_chemicals, candidates):
                if j and solubility is None and not (i.Tm and i.Hfus):
                    raise RuntimeError(
                        f"solute {i} requires a melting temperature, Tm, "
                        f"and a heat of fusion, Hfus"
                    )
        if solubility is None:
            self._solubility = None
        else:
            if candidates.sum() != 1:
                raise ValueError('solubility can only be specified for one solute')
            self._solubility = solubility
        if not candidates.any(): return False
        if self._sle_chemicals != sle_chemicals:
            self._sle_chemicals = sle_chemicals
            self._gamma = self.thermo.Gamma(sle_chemicals)
            self._active = None
        if self._active is None or (self._active & ~candidates).any():
            self._active = np.zeros(len(index), bool)
        self._index = index
        self._z = z
        self._candidates = candidates
        return True

    def _log_psi(self, T):
        candidates = self._candidates
        log_psi = np.full(candidates.size, np.inf)
        dCp = self.dCp
        if self._solubility is None:
            for i, (chemical, candidate) in enumerate(zip(self._sle_chemicals, candidates)):
                if not candidate: continue
                if dCp is None:
                    dCp_i = 0.
                elif isinstance(dCp, dict):
                    dCp_i = dCp.get(chemical.ID, 0.)
                else:
                    dCp_i = dCp
                log_psi[i] = log_ideal_solubility(chemical, T, dCp_i)
        else:
            log_psi[candidates] = log(self._solubility)
        return log_psi

    def _log_gamma(self, x, T):
        if self._solubility is not None:
            return np.zeros(x.size)
        gamma = self._gamma
        if isinstance(gamma, IdealActivityCoefficients):
            activity_coefficient = self.activity_coefficient
            if activity_coefficient is None:
                return np.zeros(x.size)
            log_gamma = np.zeros(x.size)
            log_gamma[self._candidates] = log(activity_coefficient)
            return log_gamma
        else:
            return np.log(gamma(x, T))

    def _liquid_mol_iter(self, log_gamma, z, log_psi, active, T):
        liquid_mol = liquid_mol_at_saturation(z, np.exp(log_psi - log_gamma), active)
        L = liquid_mol.sum()
        if L == 0.: return log_gamma
        return self._log_gamma(liquid_mol / L, T)

    def _solve_liquid_mol(self, z, log_psi, active, T):
        x = z / z.sum()
        log_gamma = self._log_gamma(x, T)
        if active.any():
            log_gamma = flx.aitken(
                self._liquid_mol_iter, log_gamma,
                args=(z, log_psi, active, T),
                **self.solver_options,
            )
            liquid_mol = liquid_mol_at_saturation(z, np.exp(log_psi - log_gamma), active)
            L = liquid_mol.sum()
            if L: log_gamma = self._log_gamma(liquid_mol / L, T)
        else:
            liquid_mol = z.copy()
            L = z.sum()
        return liquid_mol, L, log_gamma

    def _solve_TP(self, T):
        # Return liquid molar flow rates and active solids at equilibrium
        z = self._z
        log_psi = self._log_psi(T)
        candidates = self._candidates
        # Note that the active set is not warm started; starting from a
        # single liquid phase guarantees that complete solidification is
        # only reached when no liquid can coexist with all pure solids.
        active = np.zeros(candidates.size, bool)
        stability_tol = self.stability_tol
        for _ in range(self.maxiter_active_set):
            liquid_mol, L, log_gamma = self._solve_liquid_mol(z, log_psi, active, T)
            # Remove fully dissolved solids
            dissolved = active & (liquid_mol >= z * (1. - 1e-12))
            if dissolved.any():
                active[dissolved] = False
                continue
            if L == 0.: break # Completely solid
            # Add the most supersaturated solid
            x = liquid_mol / L
            with np.errstate(divide='ignore'):
                d = np.log(x) + log_gamma - log_psi
            d[~candidates | active] = -np.inf
            i = d.argmax()
            if d[i] > stability_tol:
                active[i] = True
            else:
                break
        self._active = active
        return liquid_mol

    def _set_T(self, T):
        liquid_mol = self._solve_TP(T)
        self._set_mol(liquid_mol)

    def _set_mol(self, liquid_mol):
        imol = self.imol
        index = self._index
        z = self._z
        solid_mol = z - liquid_mol
        solid_mol[solid_mol < 0.] = 0.
        imol['l'][index] = liquid_mol
        imol['s'][index] = solid_mol
        IDs = self.chemicals.IDs
        self.solids = tuple([IDs[i] for i, j in zip(index, self._active) if j])

    def _H_err_at_T(self, T, H, P):
        self._set_T(T)
        return self.mixture.xH(self.imol, T, P) - H

    def _set_H(self, H, P):
        thermal_condition = self.thermal_condition
        imol = self.imol
        mixture = self.mixture
        index = self._index
        z = self._z
        candidates = self._candidates
        if self._solubility is None:
            T_max = self._temperature_bounds()[1]
            T_min = max(self._T_solidus() - 1e-6, self.T_tol)
        else:
            T = thermal_condition.T
            T_max = T + 50.
            T_min = max(T - 50., 1.)

        # Check if completely liquid
        imol['l'][index] = z
        imol['s'][index] = 0.
        H_max = mixture.xH(imol, T_max, P)
        if H >= H_max:
            self._active[:] = False
            self.solids = ()
            thermal_condition.T = mixture.xsolve_T_at_HP(imol, H, T_max, P)
            return

        # Check if completely solid (or at the lowest temperature bound if
        # chemicals that do not crystallize remain in the liquid)
        self._set_T(T_min)
        H_min = mixture.xH(imol, T_min, P)
        if H <= H_min:
            # Assume phases do not change below the lowest temperature bound
            thermal_condition.T = mixture.xsolve_T_at_HP(imol, H, T_min, P)
            return

        # Solve temperature. Enthalpy is monotonic in temperature with
        # discontinuities at invariant points (e.g., eutectic and melting
        # points), where the solid fraction varies at constant temperature.
        T_guess = thermal_condition.T
        if not T_min < T_guess < T_max: T_guess = 0.5 * (T_min + T_max)
        ytol = self.H_tol * (H_max - H_min)
        T_cold, T_hot, converged = solve_monotonic_bracket(
            self._H_err_at_T, T_min, T_max, H_min - H, H_max - H, T_guess,
            self.T_tol, ytol, self.maxiter_temperature, (H, P),
        )
        if converged:
            thermal_condition.T = T_hot
            self._set_T(T_hot)
        else:
            # Invariant point (e.g., eutectic or melting point), where the
            # solid fraction varies at constant temperature; apply lever rule
            # between both sides.
            T = 0.5 * (T_cold + T_hot)
            self._set_T(T_cold)
            data_cold = imol.data.copy()
            H_cold = mixture.xH(imol, T, P)
            active_cold = self._active.copy()
            self._set_T(T_hot)
            data_hot = imol.data.copy()
            H_hot = mixture.xH(imol, T, P)
            if H_hot != H_cold:
                f = (H - H_cold) / (H_hot - H_cold)
                f = min(max(f, 0.), 1.)
                imol.data[:] = data_cold + f * (data_hot - data_cold)
                self._active = active_cold | self._active
                IDs = self.chemicals.IDs
                self.solids = tuple([IDs[i] for i, j in zip(index, self._active) if j])
            thermal_condition.T = T


class SLECache(Cache): load = SLE
del Cache, Equilibrium
