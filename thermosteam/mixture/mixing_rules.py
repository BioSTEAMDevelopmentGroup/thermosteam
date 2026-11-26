# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2025, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
Mixing rules for volumetric, transport, and interfacial properties of
non-ideal mixtures.

All models are homogeneous of degree one in the molar data (i.e., they
return the mixture property of the normalized composition multiplied by
the total moles). This is consistent with the molar weighted models of
the ideal mixture, so that multi-phase properties remain phase-fraction
weighted sums when the molar data of each phase is normalized by the
total moles of all phases.

References
----------
.. [1] Poling, B. E.; Prausnitz, J. M.; O'Connell, J. P. The Properties
   of Gases and Liquids, 5th ed.; McGraw-Hill: New York, 2001.
.. [2] Wilke, C. R. A Viscosity Equation for Gas Mixtures. J. Chem. Phys.
   1950, 18 (4), 517-519.
.. [3] Mason, E. A.; Saxena, S. C. Approximate Formula for the Thermal
   Conductivity of Gas Mixtures. Phys. Fluids 1958, 1 (5), 361-369.
.. [4] Winterfeld, P. H.; Scriven, L. E.; Davis, H. T. An Approximate
   Theory of Interfacial Tensions of Multicomponent Systems: Applications
   to Binary Liquid-Vapor Tensions. AIChE J. 1978, 24 (6), 1010-1014.
.. [5] Li, C. C. Thermal Conductivity of Liquid Mixtures. AIChE J.
   1976, 22 (5), 927-930.
.. [6] Tsonopoulos, C. An Empirical Correlation of Second Virial
   Coefficients. AIChE J. 1974, 20 (2), 263-272.
.. [7] Tarakad, R. R.; Danner, R. P. An Improved Corresponding States
   Method for Polar Fluids: Correlation of Second Virial Coefficients.
   AIChE J. 1977, 23 (5), 685-695.

"""
import numpy as np
from numba import njit
from fluids.constants import R
from ..base import display_asfunctor, SparseVector, MockPhaseTHandle

__all__ = (
    'VolumeMixtureModel',
    'ViscosityMixtureModel',
    'ThermalConductivityMixtureModel',
    'SurfaceTensionMixtureModel',
    'liquid_viscosity_logarithmic',
    'gas_viscosity_Wilke',
    'liquid_thermal_conductivity_Li',
    'gas_thermal_conductivity_Mason_Saxena',
    'surface_tension_Winterfeld_Scriven_Davis',
    'second_virial_coefficients_Tsonopoulos',
    'gas_excess_volume_virial',
)

phase_names = {'l': 'liquid', 'L': 'liquid', 'g': 'gas', 's': 'solid', 'S': 'solid'}

# %% Utilities

def describe_chemical(IDs, i):
    if IDs is None:
        return f"chemical at index {i}"
    else:
        return repr(IDs[i])

def describe_conditions(phase, T, P):
    conditions = f"at T={T:.5g} K and P={P:.5g} Pa"
    if phase is None:
        return conditions
    else:
        phase_name = phase_names.get(phase, repr(phase))
        return f"in the {phase_name} phase {conditions}"

def raise_infeasible_value(name, IDs, index, phase, T, P, pure_values):
    """
    Raise a RuntimeError noting which chemical had an infeasible pure
    component value (by property description in `pure_values`). If all
    pure component values are feasible, the error notes the infeasible
    mixture property instead.
    """
    conditions = describe_conditions(phase, T, P)
    for description, values in pure_values.items():
        for i, value in zip(index, values):
            if 0. <= value < np.inf: continue
            chemical = describe_chemical(IDs, i)
            raise RuntimeError(
                f"infeasible {description} of {chemical} ({value!r}) {conditions}"
            )
    chemicals = ', '.join([describe_chemical(IDs, i) for i in index])
    raise RuntimeError(
        f"infeasible mixture {name} of {chemicals} {conditions}"
    )

# %% Kernel helpers

@njit(cache=True)
def Wilke_phi(mui, muj, MWi, MWj):
    # phi_ij = [1 + (mu_i/mu_j)^0.5 * (M_j/M_i)^0.25]^2 / [8 * (1 + M_i/M_j)]^0.5
    MW_ratio = MWi / MWj
    a = 1. + np.sqrt(mui / muj) * MW_ratio ** -0.25
    return a * a / np.sqrt(8. * (1. + MW_ratio))

@njit(cache=True)
def Wassiljewa(x, values, mus, MWs):
    # sum_i x_i * value_i / sum_j x_j * phi_ij
    N = x.size
    total = 0.
    for i in range(N):
        mui = mus[i]
        MWi = MWs[i]
        denominator = 0.
        for j in range(N):
            denominator += x[j] * Wilke_phi(mui, mus[j], MWi, MWs[j])
        total += x[i] * values[i] / denominator
    return total

@njit(cache=True)
def Tsonopoulos(T, Tc, Pc, omega):
    # Second virial coefficient [m^3/mol] without polar terms
    Tr_inv = Tc / T
    Tr_inv2 = Tr_inv * Tr_inv
    Tr_inv3 = Tr_inv2 * Tr_inv
    Tr_inv8 = Tr_inv3 * Tr_inv3 * Tr_inv2
    B0 = 0.1445 - 0.33 * Tr_inv - 0.1385 * Tr_inv2 - 0.0121 * Tr_inv3 - 0.000607 * Tr_inv8
    B1 = 0.0637 + 0.331 * Tr_inv2 - 0.423 * Tr_inv3 - 0.008 * Tr_inv8
    return (B0 + omega * B1) * R * Tc / Pc

@njit(cache=True)
def Tsonopoulos_cross(T, Tci, Tcj, factori, factorj, Vc_cbrti, Vc_cbrtj, omegai, omegaj):
    # Cross second virial coefficient [m^3/mol] by Tarakad-Danner combining rules
    Tcij = np.sqrt(Tci * Tcj)
    den = Vc_cbrti + Vc_cbrtj
    Pcij = 4. * Tcij * (factori + factorj) / (den * den * den)
    return Tsonopoulos(T, Tcij, Pcij, 0.5 * (omegai + omegaj))

# %% Mixing rules

@njit(cache=True)
def second_virial_coefficients_Tsonopoulos(T, Tcs, Pcs, Vcs, omegas):
    r"""
    Return the matrix of second virial coefficients [m^3/mol] by the
    Tsonopoulos correlation [6]_ (without polar terms), using the
    combining rules of Tarakad and Danner [7]_ for cross coefficients:

    .. math::

        B_{ij} = \frac{R T_{c,ij}}{P_{c,ij}}\left[B^{(0)}(T_{r,ij})
        + \omega_{ij} B^{(1)}(T_{r,ij})\right]

        T_{c,ij} = \sqrt{T_{c,i} T_{c,j}}

        P_{c,ij} = \frac{4 T_{c,ij} \left(P_{c,i} V_{c,i} / T_{c,i}
        + P_{c,j} V_{c,j} / T_{c,j}\right)}{\left(V_{c,i}^{1/3}
        + V_{c,j}^{1/3}\right)^3}

        \omega_{ij} = \frac{\omega_i + \omega_j}{2}

    Diagonal elements are the pure component second virial coefficients.

    """
    N = Tcs.size
    factors = Pcs * Vcs / Tcs
    Vc_cbrts = Vcs ** (1. / 3.)
    Bs = np.empty((N, N))
    for i in range(N):
        Bs[i, i] = Tsonopoulos(T, Tcs[i], Pcs[i], omegas[i])
        for j in range(i):
            Bs[i, j] = Bs[j, i] = Tsonopoulos_cross(
                T, Tcs[i], Tcs[j], factors[i], factors[j],
                Vc_cbrts[i], Vc_cbrts[j], omegas[i], omegas[j]
            )
    return Bs

@njit(cache=True)
def gas_excess_volume_virial(x, Bs):
    r"""
    Return the excess molar volume of a gas mixture [m^3/mol] truncated
    at the second virial coefficient, given the mole fractions and the
    matrix of second virial coefficients (pure and cross):

    .. math::

        V^E = \frac{1}{2} \sum_i \sum_j x_i x_j (2 B_{ij} - B_{ii} - B_{jj})

    Mole fractions need not sum to one (e.g., when chemicals without
    critical properties are excluded, which is equivalent to assuming
    :math:`2 B_{ij} = B_{ii} + B_{jj}` for their pairs).

    """
    N = x.size
    VE = 0.
    for i in range(N):
        for j in range(i):
            VE += x[i] * x[j] * (2. * Bs[i, j] - Bs[i, i] - Bs[j, j])
    return VE

@njit(cache=True)
def gas_excess_volume_Tsonopoulos(x, T, Tcs, Pcs, Vcs, omegas):
    # Same as gas_excess_volume_virial(x, second_virial_coefficients_Tsonopoulos(...))
    # without allocating the matrix of second virial coefficients.
    N = x.size
    factors = Pcs * Vcs / Tcs
    Vc_cbrts = Vcs ** (1. / 3.)
    Bii = np.empty(N)
    for i in range(N): Bii[i] = Tsonopoulos(T, Tcs[i], Pcs[i], omegas[i])
    VE = 0.
    for i in range(N):
        for j in range(i):
            Bij = Tsonopoulos_cross(
                T, Tcs[i], Tcs[j], factors[i], factors[j],
                Vc_cbrts[i], Vc_cbrts[j], omegas[i], omegas[j]
            )
            VE += x[i] * x[j] * (2. * Bij - Bii[i] - Bii[j])
    return VE

@njit(cache=True)
def liquid_viscosity_logarithmic(x, mus):
    r"""
    Return the viscosity of a liquid mixture [Pa*s] by logarithmic
    molar mixing (Grunberg-Nissan with no interaction parameters):

    .. math::

        \ln \mu = \sum_i x_i \ln \mu_i

    """
    ln_mu = 0.
    for i in range(x.size): ln_mu += x[i] * np.log(mus[i])
    return np.exp(ln_mu)

@njit(cache=True)
def gas_viscosity_Wilke(x, mus, MWs):
    r"""
    Return the viscosity of a low pressure gas mixture [Pa*s] by the
    method of Wilke [2]_:

    .. math::

        \mu = \sum_i \frac{x_i \mu_i}{\sum_j x_j \phi_{ij}}

        \phi_{ij} = \frac{\left[1 + (\mu_i / \mu_j)^{1/2}
        (M_j / M_i)^{1/4}\right]^2}{\left[8(1 + M_i / M_j)\right]^{1/2}}

    """
    return Wassiljewa(x, mus, mus, MWs)

@njit(cache=True)
def liquid_thermal_conductivity_Li(x, kappas, Vs):
    r"""
    Return the thermal conductivity of a liquid mixture [W/m/K] by the
    method of Li [1]_ [5]_ (DIPPR procedure 9I), where :math:`\varphi_i`
    are liquid volume fractions:

    .. math::

        \kappa = \sum_i \sum_j \varphi_i \varphi_j \kappa_{ij}

        \kappa_{ij} = \frac{2}{1/\kappa_i + 1/\kappa_j}

    Notes
    -----
    Unlike DIPPR procedure 9H, Li's method is not dominated by dilute
    components with small thermal conductivities (e.g., dissolved gases
    with extrapolated liquid properties above their critical temperature).

    """
    N = x.size
    phi = x * Vs
    phi /= phi.sum()
    inverse = 1. / kappas
    kappa = 0.
    for i in range(N):
        phii = phi[i]
        kappa += phii * phii * kappas[i]
        for j in range(i):
            kappa += 4. * phii * phi[j] / (inverse[i] + inverse[j])
    return kappa

@njit(cache=True)
def gas_thermal_conductivity_Mason_Saxena(x, kappas, mus, MWs):
    r"""
    Return the thermal conductivity of a low pressure gas mixture [W/m/K]
    by the Wassiljewa equation with the Mason-Saxena coefficients [1]_ [3]_
    (which are equal to the Wilke coefficients of the gas viscosity
    after approximating the ratio of translational thermal conductivities
    with viscosities and molecular weights):

    .. math::

        \kappa = \sum_i \frac{x_i \kappa_i}{\sum_j x_j A_{ij}}

        A_{ij} = \phi_{ij}

    """
    return Wassiljewa(x, kappas, mus, MWs)

@njit(cache=True)
def surface_tension_Winterfeld_Scriven_Davis(x, sigmas, Vs):
    r"""
    Return the surface tension of a liquid mixture [N/m] by the method of
    Winterfeld, Scriven, and Davis [4]_, where :math:`\varphi_i` are
    liquid volume fractions:

    .. math::

        \sigma = \sum_i \sum_j \varphi_i \varphi_j \sqrt{\sigma_i \sigma_j}
        = \left(\sum_i \varphi_i \sqrt{\sigma_i}\right)^2

    """
    phiV = 0.
    total = 0.
    for i in range(x.size):
        phi = x[i] * Vs[i]
        total += phi
        phiV += phi * np.sqrt(sigmas[i])
    sigma = phiV / total
    return sigma * sigma

@njit(cache=True)
def molar_weighted(x, values):
    total = 0.
    for i in range(x.size): total += x[i] * values[i]
    return total

# %% Mixture models

class MixtureModel:
    r"""
    Abstract class for mixture models of the form:

    .. math::

        y = \sum_i x_i y_i + y^E

    where :math:`y_i` are pure component values and :math:`y^E` is the
    excess value given by the `excess` method of subclasses. The excess is
    only computed for mixtures (i.e., at least two chemicals) in phases
    with mixing rules (i.e., `excess_phases`).

    Abstract attributes
    -------------------
    var : str
        Name of mixture property.
    description : str
        Description of mixture property (for error messages).
    excess_phases : frozenset[str]
        Phases with an excess value.

    Abstract methods
    ----------------
    excess(phase, index, x, values, T, P) : float
        Excess value given indices of chemicals present, molar fractions,
        and pure component values.
    auxiliary_values(phase, index, T, P) : dict[str, 1d array]
        Pure component values (other than of the mixture property) used by
        the mixing rule, by description (for error messages).

    """
    __slots__ = ()
    allow_zero = False

    def __call__(self, phase, mol, T, P):
        # Molar fractions and total moles of chemicals present
        if mol.__class__ is SparseVector:
            dct = mol.dct
            N = len(dct)
            index = np.fromiter(dct, int, N)
            mol = np.fromiter(dct.values(), float, N)
        else:
            mol = np.asarray(mol, float)
            index = np.flatnonzero(mol)
            mol = mol[index]
            N = index.size
        if N == 0: return 0.
        models = self.models
        total = mol.sum()
        if N == 1: return total * models[index[0]](phase, T, P)
        x = mol / total
        # Pure component values
        values = np.array([models[i](phase, T, P) for i in index], float)
        value = molar_weighted(x, values)
        if phase in self.excess_phases:
            try:
                value += self.excess(phase, index, x, values, T, P)
            except ArithmeticError: # Only when numba JIT is disabled
                value = np.nan
        if not 0. <= value < np.inf:
            self.raise_infeasible_value(phase, index, values, T, P)
        return total * value

    def auxiliary_values(self, phase, index, T, P):
        return {}

    def raise_infeasible_value(self, phase, index, values, T, P):
        pure_values = {self.description: values}
        if phase in self.excess_phases:
            pure_values.update(self.auxiliary_values(phase, index, T, P))
        raise_infeasible_value(
            self.description, self.IDs, index, phase, T, P,
            pure_values, 
        )

    def __repr__(self):
        return f"<{display_asfunctor(self)}>"


class VolumeMixtureModel(MixtureModel):
    r"""
    Create a VolumeMixtureModel object that calculates the molar volume
    of a mixture [m^3/mol]. Gas molar volumes are the molar weighted pure
    component volumes plus the excess volume given by the second virial
    coefficients of the mixture (Tsonopoulos correlation with Tarakad and
    Danner combining rules). Liquid and solid molar volumes are molar
    weighted (Amagat's law).

    Parameters
    ----------
    models : Iterable[``function(phase, T, P)``]
        Chemical molar volume functions of phase, temperature and pressure.
    Tcs : Iterable[float]
        Critical temperatures [K].
    Pcs : Iterable[float]
        Critical pressures [Pa].
    Vcs : Iterable[float]
        Critical molar volumes [m^3/mol].
    omegas : Iterable[float]
        Acentric factors [-].
    IDs : Iterable[str], optional
        Chemical identifiers (for error messages).

    Raises
    ------
    RuntimeError
        If the mixture molar volume is infeasible (e.g., due to a
        non-positive or non-finite pure component molar volume).

    Notes
    -----
    The virial excess volume vanishes for pure components, so pure
    component volumes are given by the pure component models. Chemicals
    missing critical properties do not contribute to the excess volume.

    Amagat's law is used for liquids because activity coefficient models
    have no pressure dependence (i.e., the excess volume of the liquid,
    :math:`V^E = (\partial G^E / \partial P)_{T,x}`, is zero) and
    corresponding states mixing rules for liquid volumes (e.g., COSTALD
    and Rackett) predict excess volumes less accurately than Amagat's law
    for both polar and non-polar mixtures.

    Examples
    --------
    >>> from thermosteam.mixture import VolumeMixtureModel
    >>> from thermosteam import Chemicals
    >>> chemicals = Chemicals(['Water', 'Ethanol'])
    >>> mixture_model = VolumeMixtureModel(
    ...     [i.V for i in chemicals],
    ...     *[[getattr(i, j) for i in chemicals] for j in ('Tc', 'Pc', 'Vc', 'omega')]
    ... )
    >>> mixture_model
    <VolumeMixtureModel(phase, mol, T, P) -> V [m^3/mol]>
    >>> mixture_model('g', [0.2, 0.8], 400, 101325)
    0.03...

    """
    __slots__ = ('models', 'Tcs', 'Pcs', 'Vcs', 'omegas', 'has_critical_data', 'IDs')
    var = 'V'
    description = 'molar volume'
    excess_phases = frozenset(['g'])

    def __init__(self, models, Tcs, Pcs, Vcs, omegas, IDs=None):
        self.models = tuple(models)
        self.Tcs = Tcs = np.asarray(Tcs, float) # Missing values (None) are NaN
        self.Pcs = Pcs = np.asarray(Pcs, float)
        self.Vcs = Vcs = np.asarray(Vcs, float)
        self.omegas = omegas = np.asarray(omegas, float)
        self.has_critical_data = (Tcs > 0.) & (Pcs > 0.) & (Vcs > 0.) & np.isfinite(omegas)
        self.IDs = None if IDs is None else tuple(IDs)

    def excess(self, phase, index, x, Vs, T, P):
        """Return the excess molar volume of a gas mixture [m^3/mol]."""
        has_critical_data = self.has_critical_data[index]
        if has_critical_data.sum() < 2: return 0.
        index = index[has_critical_data]
        return gas_excess_volume_Tsonopoulos(
            x[has_critical_data], T,
            self.Tcs[index], self.Pcs[index], self.Vcs[index], self.omegas[index]
        )


class ViscosityMixtureModel(MixtureModel):
    """
    Create a ViscosityMixtureModel object that calculates the dynamic
    viscosity of a mixture [Pa*s]. Liquid viscosities are mixed
    logarithmically by mole fraction, gas viscosities are mixed by Wilke's
    method, and solid viscosities are molar weighted.

    Parameters
    ----------
    models : Iterable[``function(phase, T, P)``]
        Chemical viscosity functions of phase, temperature and pressure.
    MWs : 1d array[float]
        Component molecular weights [g/mol].
    IDs : Iterable[str], optional
        Chemical identifiers (for error messages).

    Raises
    ------
    RuntimeError
        If the mixture viscosity is infeasible (e.g., due to a non-positive
        or non-finite pure component viscosity).

    Examples
    --------
    >>> from thermosteam.mixture import ViscosityMixtureModel
    >>> from thermosteam import Chemicals
    >>> chemicals = Chemicals(['Water', 'Ethanol'])
    >>> mixture_model = ViscosityMixtureModel([i.mu for i in chemicals], [i.MW for i in chemicals])
    >>> mixture_model
    <ViscosityMixtureModel(phase, mol, T, P) -> mu [Pa*s]>
    >>> mixture_model('l', [0.2, 0.8], 350, 101325)
    0.00044...

    """
    __slots__ = ('models', 'MWs', 'IDs')
    var = 'mu'
    description = 'viscosity'
    excess_phases = frozenset(['l', 'L', 'g'])

    def __init__(self, models, MWs, IDs=None):
        self.models = tuple(models)
        self.MWs = np.asarray(MWs, float)
        self.IDs = None if IDs is None else tuple(IDs)

    def excess(self, phase, index, x, mus, T, P):
        """Return the excess viscosity [Pa*s]."""
        if phase == 'g':
            mu = gas_viscosity_Wilke(x, mus, self.MWs[index])
        else:
            mu = liquid_viscosity_logarithmic(x, mus)
        return mu - molar_weighted(x, mus)


class ThermalConductivityMixtureModel(MixtureModel):
    """
    Create a ThermalConductivityMixtureModel object that calculates the
    thermal conductivity of a mixture [W/m/K]. Liquid thermal
    conductivities are mixed by Li's method, gas thermal
    conductivities are mixed by the Wassiljewa equation with
    Mason-Saxena coefficients, and solid thermal conductivities are
    molar weighted.

    Parameters
    ----------
    models : Iterable[``function(phase, T, P)``]
        Chemical thermal conductivity functions of phase, temperature and
        pressure.
    mu_models : Iterable[``function(phase, T, P)``]
        Chemical viscosity functions of phase, temperature and pressure
        (for gas mixing coefficients).
    V_models : Iterable[``function(phase, T, P)``]
        Chemical molar volume functions of phase, temperature and pressure
        (for liquid volume fractions).
    MWs : 1d array[float]
        Component molecular weights [g/mol].
    IDs : Iterable[str], optional
        Chemical identifiers (for error messages).

    Raises
    ------
    RuntimeError
        If the mixture thermal conductivity is infeasible (e.g., due to a
        non-positive or non-finite pure component thermal conductivity,
        liquid molar volume, or gas viscosity).

    Examples
    --------
    >>> from thermosteam.mixture import ThermalConductivityMixtureModel
    >>> from thermosteam import Chemicals
    >>> chemicals = Chemicals(['Water', 'Ethanol'])
    >>> mixture_model = ThermalConductivityMixtureModel(
    ...     [i.kappa for i in chemicals], [i.mu for i in chemicals],
    ...     [i.V for i in chemicals], [i.MW for i in chemicals]
    ... )
    >>> mixture_model
    <ThermalConductivityMixtureModel(phase, mol, T, P) -> kappa [W/m/K]>
    >>> mixture_model('l', [0.2, 0.8], 350, 101325)
    0.17...

    """
    __slots__ = ('models', 'mu_models', 'V_models', 'MWs', 'IDs')
    var = 'kappa'
    description = 'thermal conductivity'
    excess_phases = frozenset(['l', 'L', 'g'])

    def __init__(self, models, mu_models, V_models, MWs, IDs=None):
        self.models = tuple(models)
        self.mu_models = tuple(mu_models)
        self.V_models = tuple(V_models)
        self.MWs = np.asarray(MWs, float)
        self.IDs = None if IDs is None else tuple(IDs)

    def auxiliary_values(self, phase, index, T, P):
        if phase == 'g':
            mu_models = self.mu_models
            return {'viscosity': np.array([mu_models[i](phase, T, P) for i in index], float)}
        else:
            V_models = self.V_models
            return {'molar volume': np.array([V_models[i](phase, T, P) for i in index], float)}

    def excess(self, phase, index, x, kappas, T, P):
        """Return the excess thermal conductivity [W/m/K]."""
        if phase == 'g':
            mu_models = self.mu_models
            mus = np.array([mu_models[i](phase, T, P) for i in index], float)
            kappa = gas_thermal_conductivity_Mason_Saxena(x, kappas, mus, self.MWs[index])
        else:
            V_models = self.V_models
            Vs = np.array([V_models[i](phase, T, P) for i in index], float)
            kappa = liquid_thermal_conductivity_Li(x, kappas, Vs)
        return kappa - molar_weighted(x, kappas)


class SurfaceTensionMixtureModel(MixtureModel):
    """
    Create a SurfaceTensionMixtureModel object that calculates the surface
    tension of a liquid mixture [N/m] by the method of Winterfeld,
    Scriven, and Davis.

    Parameters
    ----------
    models : Iterable[``function(T)``]
        Chemical surface tension functions of temperature.
    V_models : Iterable[``function(phase, T, P)``]
        Chemical molar volume functions of phase, temperature and pressure.
    IDs : Iterable[str], optional
        Chemical identifiers (for error messages).

    Raises
    ------
    RuntimeError
        If the mixture surface tension is infeasible (e.g., due to a
        negative or non-finite pure component surface tension, or a
        non-positive or non-finite liquid molar volume).

    Examples
    --------
    >>> from thermosteam.mixture import SurfaceTensionMixtureModel
    >>> from thermosteam import Chemicals
    >>> chemicals = Chemicals(['Water', 'Ethanol'])
    >>> mixture_model = SurfaceTensionMixtureModel(
    ...     [i.sigma for i in chemicals], [i.V for i in chemicals]
    ... )
    >>> mixture_model
    <SurfaceTensionMixtureModel(mol, T, P) -> sigma [N/m]>
    >>> mixture_model([0.2, 0.8], 350, 101325)
    0.01...

    """
    __slots__ = ('models', 'V_models', 'IDs')
    var = 'sigma'
    description = 'surface tension'
    excess_phases = frozenset(['l'])

    def __init__(self, models, V_models, IDs=None):
        # Surface tension models are only functions of temperature
        self.models = tuple([
            i if isinstance(i, MockPhaseTHandle) else MockPhaseTHandle('sigma', i)
            for i in models
        ])
        self.V_models = tuple(V_models)
        self.IDs = None if IDs is None else tuple(IDs)

    def __call__(self, mol, T, P):
        return MixtureModel.__call__(self, 'l', mol, T, P)

    def auxiliary_values(self, phase, index, T, P):
        V_models = self.V_models
        return {'molar volume': np.array([V_models[i](phase, T, P) for i in index], float)}

    def excess(self, phase, index, x, sigmas, T, P):
        """Return the excess surface tension [N/m]."""
        V_models = self.V_models
        Vs = np.array([V_models[i](phase, T, P) for i in index], float)
        sigma = surface_tension_Winterfeld_Scriven_Davis(x, sigmas, Vs)
        return sigma - molar_weighted(x, sigmas)
