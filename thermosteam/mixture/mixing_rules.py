# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2025, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
Mixing rules for transport and interfacial properties of non-ideal
mixtures.

All models assume the molar data is normalized (i.e., mole fractions that
sum to one). Multi-phase properties are phase-fraction weighted sums of
the properties of each phase given its normalized composition (see
`Mixture.xmu` and `Mixture.xkappa`).

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

"""
import numpy as np
from numba import njit
from ..base import display_asfunctor, SparseVector, MockPhaseTHandle

__all__ = (
    'ViscosityMixtureModel',
    'ThermalConductivityMixtureModel',
    'SurfaceTensionMixtureModel',
    'Wassiljewa',
    'liquid_viscosity_logarithmic',
    'liquid_thermal_conductivity_Li',
    'surface_tension_Winterfeld_Scriven_Davis',
)

# %% Mixing rules

@njit(cache=True)
def Wassiljewa(x, values, mus, MWs):
    r"""
    Return the mixture property of a low pressure gas mixture by the
    Wassiljewa equation with the Wilke coefficients [1]_ [2]_ [3]_:

    .. math::

        y = \sum_i \frac{x_i y_i}{\sum_j x_j \phi_{ij}}

        \phi_{ij} = \frac{\left[1 + (\mu_i / \mu_j)^{1/2}
        (M_j / M_i)^{1/4}\right]^2}{\left[8(1 + M_i / M_j)\right]^{1/2}}

    Notes
    -----
    Used for the following gas mixture properties:

    * Viscosity by the method of Wilke [2]_ (`values` are the
      viscosities, `mus`).
    * Thermal conductivity by the Wassiljewa equation with the
      Mason-Saxena coefficients [3]_ (`values` are the thermal
      conductivities). The Mason-Saxena coefficients equal the Wilke
      coefficients after approximating the ratio of translational
      thermal conductivities with viscosities and molecular weights.

    """
    MW_ratio = np.outer(MWs, 1. / MWs) # M_i / M_j
    a = 1. + np.sqrt(np.outer(mus, 1. / mus)) * MW_ratio ** -0.25
    phi = a * a / np.sqrt(8. * (1. + MW_ratio))
    return (x * values / (phi * x).sum(axis=1)).sum()

@njit(cache=True)
def liquid_viscosity_logarithmic(x, mus):
    r"""
    Return the viscosity of a liquid mixture [Pa*s] by logarithmic
    molar mixing (Grunberg-Nissan with no interaction parameters):

    .. math::

        \ln \mu = \sum_i x_i \ln \mu_i

    """
    return np.exp((x * np.log(mus)).sum())

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
    phi = x * Vs
    phi /= phi.sum()
    inverse = 1. / kappas
    N = inverse.size
    kappas_ij = 2. / (inverse.reshape((N, 1)) + inverse.reshape((1, N)))
    return (np.outer(phi, phi) * kappas_ij).sum()

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
    phi = x * Vs
    sigma_sqrt = (phi * np.sqrt(sigmas)).sum() / phi.sum()
    return sigma_sqrt * sigma_sqrt

# %% Mixture models

class MixtureModel:
    r"""
    Abstract class for mixture models of the form:

    .. math::

        y = f(x)

    where :math:`f` is the mixing rule given by the `f` method of
    subclasses. The mixing rule is only used for mixtures (i.e., at least
    two chemicals) in phases with mixing rules (i.e., `excess_phases`);
    otherwise, pure component values are molar weighted.

    Abstract attributes
    -------------------
    var : str
        Name of mixture property.
    description : str
        Description of mixture property (for error messages).
    excess_phases : frozenset[str]
        Phases with a mixing rule.

    Abstract methods
    ----------------
    f(phase, mol, T, P) : float
        Mixture value given the mole fractions of at least two chemicals
        (SparseVector; normalized).

    """
    __slots__ = ()

    def __call__(self, phase, mol, T, P):
        if mol.__class__ is not SparseVector: mol = SparseVector(mol)
        dct = mol.dct
        N = len(dct)
        if N == 0:
            return 0.
        elif N == 1:
            index, total = next(iter(dct.items()))
            return total * self.models[index](phase, T, P)
        if phase in self.excess_phases:
            return self.f(phase, mol, T, P)
        else:
            # Pure component values
            models = self.models
            return sum([j * models[i](phase, T, P) for i, j in dct.items()])

    def __repr__(self):
        return f"<{display_asfunctor(self)}>"


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

    def f(self, phase, mol, T, P):
        """Return the viscosity of a mixture [Pa*s]."""
        dct = mol.dct
        N = len(dct)
        index = np.fromiter(dct, int, N)
        x = np.fromiter(dct.values(), float, N)
        models = self.models
        mus = np.array([models[i](phase, T, P) for i in index], float)
        if phase == 'g':
            return Wassiljewa(x, mus, mus, self.MWs[index])
        else:
            return liquid_viscosity_logarithmic(x, mus)


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

    def f(self, phase, mol, T, P):
        """Return the thermal conductivity of a mixture [W/m/K]."""
        dct = mol.dct
        N = len(dct)
        index = np.fromiter(dct, int, N)
        x = np.fromiter(dct.values(), float, N)
        models = self.models
        kappas = np.array([models[i](phase, T, P) for i in index], float)
        if phase == 'g':
            mu_models = self.mu_models
            mus = np.array([mu_models[i](phase, T, P) for i in index], float)
            return Wassiljewa(x, kappas, mus, self.MWs[index])
        else:
            V_models = self.V_models
            Vs = np.array([V_models[i](phase, T, P) for i in index], float)
            return liquid_thermal_conductivity_Li(x, kappas, Vs)


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

    def f(self, phase, mol, T, P):
        """Return the surface tension of a liquid mixture [N/m]."""
        dct = mol.dct
        N = len(dct)
        index = np.fromiter(dct, int, N)
        x = np.fromiter(dct.values(), float, N)
        models = self.models
        sigmas = np.array([models[i](phase, T, P) for i in index], float)
        V_models = self.V_models
        Vs = np.array([V_models[i](phase, T, P) for i in index], float)
        return surface_tension_Winterfeld_Scriven_Davis(x, sigmas, Vs)
