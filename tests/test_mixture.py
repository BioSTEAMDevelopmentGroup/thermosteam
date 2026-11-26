# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
Tests for mixture property values (including derivatives). Excess
properties of the GibbsExcessMixture are compared against thermo's UNIFAC
implementation and checked for thermodynamic consistency.
"""
import numpy as np
import thermosteam as tmo
from thermosteam import equilibrium as eq
from thermo import unifac as tu
from numpy.testing import assert_allclose

# %% Utilities

def get_unifac_models():
    return [
        # Activity coefficient class, group name, subgroups, interaction parameters, thermo version
        (eq.UNIFACActivityCoefficients, 'UNIFAC', tu.UFSG, 'UFIP', 0),
        (eq.ModifiedUNIFACActivityCoefficients, 'Dortmund', tu.DOUFSG, 'DOUFIP2016', 1),
    ]

def get_activity_models():
    return [
        eq.UNIFACActivityCoefficients,
        eq.ModifiedUNIFACActivityCoefficients,
        eq.PR78ActivityCoefficients,
        eq.SRKActivityCoefficients,
    ]

def get_cases():
    return [
        # Chemicals, molar fractions, temperature
        (['Water', 'Ethanol'], [0.5, 0.5], 350.),
        (['Water', 'Ethanol', 'Acetone'], [0.2, 0.5, 0.3], 320.),
        (['Hexane', 'Ethanol'], [0.7, 0.3], 300.),
    ]

def thermo_unifac(chemicals, x, T, group_name, subgroups, interactions, version):
    return tu.UNIFAC.from_subgroups(
        T=T, xs=x,
        chemgroups=[dict(getattr(i, group_name)) for i in chemicals],
        subgroups=subgroups,
        interaction_data=getattr(tu, interactions),
        version=version,
    )

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2. * h)

def water_ethanol_acetone_mixture(Gamma):
    chemicals = tmo.Chemicals(['Water', 'Ethanol', 'Acetone', 'O2'], cache=True)
    mixture = tmo.GibbsExcessMixture.from_chemicals(chemicals, Gamma)
    mol = np.array([20., 15., 10., 0.])
    return chemicals, mixture, mol

# %% Excess properties

def test_gibbs_excess_mixture_excess_properties():
    # Excess Gibbs free energy, enthalpy, entropy, and heat capacity
    # (molar basis) are equal to thermo's UNIFAC values
    P = 101325.
    for Gamma, group_name, subgroups, interactions, version in get_unifac_models():
        for IDs, x, T in get_cases():
            chemicals = tmo.Chemicals(IDs, cache=True)
            ref = thermo_unifac(chemicals, x, T, group_name, subgroups, interactions, version)
            mixture = tmo.GibbsExcessMixture.from_chemicals(chemicals, Gamma)
            msg = f"{Gamma.__name__}: {IDs}"
            assert_allclose(mixture.GE(x, T, P), ref.GE(), rtol=1e-12, err_msg=msg)
            assert_allclose(mixture.HE(x, T, P), ref.HE(), rtol=1e-12, err_msg=msg)
            assert_allclose(mixture.SE(x, T, P), ref.SE(), rtol=1e-12, err_msg=msg)
            assert_allclose(mixture.CnE(x, T, P), ref.CpE(), rtol=1e-12, err_msg=msg)
            # Excess properties are extensive
            mol = 10. * np.array(x)
            assert_allclose(mixture.HE(mol, T, P), 10. * ref.HE(), rtol=1e-12, err_msg=msg)

def test_gibbs_excess_mixture_excess_property_consistency():
    # g^E = h^E - T s^E and c_p^E = d h^E / dT
    T = 320.
    P = 101325.
    for Gamma in get_activity_models():
        chemicals, mixture, mol = water_ethanol_acetone_mixture(Gamma)
        msg = Gamma.__name__
        assert_allclose(
            mixture.GE(mol, T, P),
            mixture.HE(mol, T, P) - T * mixture.SE(mol, T, P),
            rtol=1e-12, err_msg=msg,
        )
        HE = lambda T: mixture.HE(mol, T, P)
        assert_allclose(mixture.CnE(mol, T, P), central_difference(HE, T, 1e-3), rtol=1e-6, err_msg=msg)
        # Excess enthalpy is consistent with the temperature derivative of
        # the excess Gibbs free energy (Gibbs-Helmholtz): h^E = -T^2 d(g^E/T)/dT
        GE_T = lambda T: mixture.GE(mol, T, P) / T
        assert_allclose(mixture.HE(mol, T, P), -T * T * central_difference(GE_T, T, 1e-3), rtol=1e-6, err_msg=msg)

def test_gibbs_excess_mixture_partial_molar_excess_enthalpies():
    T = 320.
    P = 101325.
    for Gamma in get_activity_models():
        chemicals, mixture, mol = water_ethanol_acetone_mixture(Gamma)
        msg = Gamma.__name__
        h = mixture.dh_dep_dzs('l', mol, T, P)
        # Partial molar excess enthalpies are derivatives of the total excess
        # enthalpy with respect to moles
        for i in range(3):
            dmol = np.zeros(4)
            dmol[i] = 1e-4
            HE = lambda m: mixture.HE(mol + m * dmol / 1e-4, T, P)
            assert_allclose(h[i], central_difference(HE, 0., 1e-4), rtol=1e-6, err_msg=msg)
        # Euler's theorem: h^E = sum_i(x_i * h_i^E)
        assert_allclose((mol * h).sum(), mixture.HE(mol, T, P), rtol=1e-10, err_msg=msg)
        # Absent chemicals and non-liquid phases have no excess enthalpy
        assert h[3] == 0., msg
        assert (mixture.dh_dep_dzs('g', mol, T, P) == 0.).all(), msg

# %% Mixture properties

def test_gibbs_excess_mixture_properties():
    T = 320.
    P = 101325.
    for Gamma in get_activity_models():
        chemicals, mixture, mol = water_ethanol_acetone_mixture(Gamma)
        ideal = tmo.IdealMixture.from_chemicals(chemicals)
        msg = Gamma.__name__
        # Liquid enthalpy, entropy, and heat capacity include excess properties
        for phase in ('l', 'L'):
            assert_allclose(mixture.H(phase, mol, T, P), ideal.H(phase, mol, T, P) + mixture.HE(mol, T, P), rtol=1e-12, err_msg=msg)
            assert_allclose(mixture.S(phase, mol, T, P), ideal.S(phase, mol, T, P) + mixture.SE(mol, T, P), rtol=1e-12, err_msg=msg)
            assert_allclose(mixture.Cn(phase, mol, T, P), ideal.Cn(phase, mol, T, P) + mixture.CnE(mol, T, P), rtol=1e-12, err_msg=msg)
        # Other phases and properties are ideal
        assert mixture.H('g', mol, T, P) == ideal.H('g', mol, T, P), msg
        assert mixture.S('g', mol, T, P) == ideal.S('g', mol, T, P), msg
        assert mixture.Cn('g', mol, T, P) == ideal.Cn('g', mol, T, P), msg
        assert_allclose(mixture.V('l', mol, T, P), ideal.V('l', mol, T, P), rtol=1e-14, err_msg=msg)
        assert mixture.Hvap(mol, T, P) == ideal.Hvap(mol, T, P), msg
        # Heat capacity is consistent with enthalpy
        H = lambda T: mixture.H('l', mol, T, P)
        assert_allclose(mixture.Cn('l', mol, T, P), central_difference(H, T, 1e-3), rtol=1e-8, err_msg=msg)

def test_gibbs_excess_mixture_pure_components():
    T = 320.
    P = 101325.
    for Gamma in get_activity_models():
        chemicals, mixture, mol = water_ethanol_acetone_mixture(Gamma)
        pure = np.array([0., 5., 0., 0.])
        assert mixture.GE(pure, T, P) == 0.
        assert mixture.HE(pure, T, P) == 0.
        assert mixture.SE(pure, T, P) == 0.
        assert mixture.CnE(pure, T, P) == 0.

def test_gibbs_excess_mixture_sparse_flows():
    # Results with sparse molar flows (as in streams) are the same as with arrays
    T = 320.
    P = 101325.
    chemicals, mixture, mol = water_ethanol_acetone_mixture(eq.ModifiedUNIFACActivityCoefficients)
    sparse = tmo.base.SparseVector(mol)
    for name in ('GE', 'HE', 'SE', 'CnE'):
        assert_allclose(getattr(mixture, name)(sparse, T, P), getattr(mixture, name)(mol, T, P), rtol=1e-14)
    assert_allclose(mixture.H('l', sparse, T, P), mixture.H('l', mol, T, P), rtol=1e-14)

def test_gibbs_excess_mixture_in_streams():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    mixture = tmo.GibbsExcessMixture.from_chemicals(chemicals, eq.ModifiedUNIFACActivityCoefficients)
    tmo.settings.set_thermo(tmo.Thermo(chemicals, mixture=mixture, pkg='Dortmund-UNIFAC'))
    assert tmo.settings.thermo.mixture is mixture
    # Mixing water and ethanol is exothermic near room temperature
    # (negative excess enthalpy), so adiabatic mixing warms the stream
    water = tmo.Stream(None, Water=50, T=298.15)
    ethanol = tmo.Stream(None, Ethanol=50, T=298.15)
    mixed = tmo.Stream(None)
    mixed.mix_from([water, ethanol])
    assert mixture.HE(mixed.mol, 298.15, 101325.) < 0.
    assert mixed.T > 298.15
    assert_allclose(mixed.H, water.H + ethanol.H, atol=1e-6) # Pure streams are at the reference state (H ~ 0)
    # Temperature from enthalpy
    stream = tmo.Stream(None, Water=30, Ethanol=20, T=300)
    H = stream.H
    stream.H = H + 1e5
    assert_allclose(stream.H, H + 1e5, rtol=1e-8)
    # Vapor-liquid equilibrium at given enthalpy and entropy
    stream = tmo.Stream(None, Water=30, Ethanol=20, T=300)
    H = stream.H
    stream.vle(H=H + 5e5, P=101325)
    assert_allclose(stream.H, H + 5e5, rtol=1e-8)
    assert 0. < stream.vapor_fraction < 1.

# %% Mixing rules

def get_nonideal_mixtures(chemicals):
    return [
        tmo.GibbsExcessMixture.from_chemicals(chemicals, eq.ModifiedUNIFACActivityCoefficients),
        tmo.PRMixture.from_chemicals(chemicals),
    ]

def test_mixing_rules():
    # Transport and interfacial properties match the mixing rules
    # implemented in the chemicals library
    from chemicals import viscosity as cv, thermal_conductivity as ck, interface as ci
    chemicals = tmo.Chemicals(['Water', 'Ethanol', 'Methanol', 'O2'], cache=True)
    x = np.array([0.3, 0.4, 0.2, 0.1])
    T = 330.
    P = 101325.
    MWs = [i.MW for i in chemicals]
    Vls = [i.V('l', T, P) for i in chemicals]
    for mixture in get_nonideal_mixtures(chemicals):
        msg = type(mixture).__name__
        mus = [i.mu('l', T, P) for i in chemicals]
        kappas = [i.kappa('l', T, P) for i in chemicals]
        assert_allclose(mixture.mu('l', x, T, P), np.exp((x * np.log(mus)).sum()), rtol=1e-12, err_msg=msg)
        assert_allclose(mixture.kappa('l', x, T, P), ck.DIPPR9I(list(x), Vls, kappas), rtol=1e-12, err_msg=msg)
        mus = [i.mu('g', T, P) for i in chemicals]
        kappas = [i.kappa('g', T, P) for i in chemicals]
        assert_allclose(mixture.mu('g', x, T, P), cv.Wilke(list(x), mus, MWs), rtol=1e-12, err_msg=msg)
        # Mason-Saxena coefficients are the Wilke coefficients
        phis = [[(1 + (mui / muj)**0.5 * (Mj / Mi)**0.25)**2 / (8 * (1 + Mi / Mj))**0.5
                 for muj, Mj in zip(mus, MWs)] for mui, Mi in zip(mus, MWs)]
        kappa = sum([x[i] * kappas[i] / (np.array(phis[i]) * x).sum() for i in range(4)])
        assert_allclose(mixture.kappa('g', x, T, P), kappa, rtol=1e-12, err_msg=msg)
        sigmas = [i.sigma(T) for i in chemicals]
        assert_allclose(
            mixture.sigma(x, T, P),
            ci.Winterfeld_Scriven_Davis(list(x), sigmas, [1 / i for i in Vls]),
            rtol=1e-12, err_msg=msg
        )

def test_mixing_rules_consistency():
    chemicals = tmo.Chemicals(['Water', 'Ethanol', 'O2'], cache=True)
    T = 330.
    P = 101325.
    x = np.array([0.5, 0.5, 0.])
    ideal = tmo.IdealMixture.from_chemicals(chemicals)
    for mixture in get_nonideal_mixtures(chemicals):
        msg = type(mixture).__name__
        for phase in ('l', 'g'):
            for name in ('mu', 'kappa', 'V'):
                f = getattr(mixture, name)
                # Properties are homogeneous of degree one in moles (as in the ideal mixture)
                assert_allclose(f(phase, 5 * x, T, P), 5 * f(phase, x, T, P), rtol=1e-10, err_msg=msg)
                # Sparse molar data gives the same result
                assert_allclose(f(phase, tmo.base.SparseVector(x), T, P), f(phase, x, T, P), rtol=1e-14, err_msg=msg)
            for name in ('mu', 'kappa'):
                # Pure components are not affected by mixing rules
                pure = np.array([0., 2., 0.])
                f = getattr(mixture, name)
                assert f(phase, pure, T, P) == getattr(ideal, name)(phase, pure, T, P), msg
        assert_allclose(mixture.sigma(5 * x, T, P), 5 * mixture.sigma(x, T, P), rtol=1e-10, err_msg=msg)
        # Trace dissolved gases (with extrapolated liquid properties above
        # their critical temperature) do not dominate liquid properties
        trace = np.array([0.5, 0.5 - 1e-5, 1e-5])
        for name in ('mu', 'kappa'):
            f = getattr(mixture, name)
            assert_allclose(f('l', trace, T, P), f('l', x, T, P), rtol=1e-3, err_msg=msg)
        assert_allclose(mixture.sigma(trace, T, P), mixture.sigma(x, T, P), rtol=1e-3, err_msg=msg)
        # Mixtures of water and ethanol have lower thermal conductivities and
        # surface tensions than predicted by molar weighting
        assert mixture.kappa('l', x, T, P) < ideal.kappa('l', x, T, P), msg
        assert mixture.sigma(x, T, P) < ideal.sigma(x, T, P), msg

def test_eos_mixture_gas_volume():
    from thermo import PRMIX
    # Gas molar volumes are given by the equation of state
    chemicals = tmo.Chemicals(['Water', 'Ethanol', 'Methane'], cache=True)
    mixture = tmo.PRMixture.from_chemicals(chemicals)
    x = np.array([0.1, 0.1, 0.8])
    T = 450.
    for P in (101325., 2e6):
        data = tmo.ChemicalData(chemicals)
        kijs = tmo.equilibrium.get_kijs('ChemSep PR', chemicals)
        ref = PRMIX(Tcs=data.Tcs, Pcs=data.Pcs, omegas=data.omegas, kijs=kijs, T=T, P=P, zs=list(x))
        assert_allclose(mixture.V('g', x, T, P), ref.V_g, rtol=1e-12)
        assert_allclose(mixture.V('g', 2 * x, T, P), 2 * ref.V_g, rtol=1e-12)
    # Liquid molar volumes follow Amagat's law
    ideal = tmo.IdealMixture.from_chemicals(chemicals)
    assert_allclose(mixture.V('l', x, 300., 101325.), ideal.V('l', x, 300., 101325.), rtol=1e-14)

def test_gibbs_excess_mixture_gas_volume():
    # Gas molar volumes include the excess volume given by second virial
    # coefficients (Tsonopoulos with Tarakad-Danner combining rules)
    from chemicals import virial as v
    chemicals = tmo.Chemicals(['Water', 'Ethanol', 'CO2', 'N2'], cache=True)
    mixture = tmo.GibbsExcessMixture.from_chemicals(chemicals, eq.ModifiedUNIFACActivityCoefficients)
    ideal = tmo.IdealMixture.from_chemicals(chemicals)
    x = np.array([0.3, 0.2, 0.3, 0.2])
    T = 400.
    N = len(x)
    Tcs, Pcs, Vcs, omegas = [[getattr(i, j) for i in chemicals] for j in ('Tc', 'Pc', 'Vc', 'omega')]
    Tcijs = v.Tarakad_Danner_virial_CSP_Tcijs(Tcs, np.zeros([N, N]))
    Pcijs = v.Tarakad_Danner_virial_CSP_Pcijs(Tcs, Pcs, Vcs, Tcijs)
    omegaijs = v.Tarakad_Danner_virial_CSP_omegaijs(omegas)
    Bs = [[v.BVirial_Tsonopoulos(T, Tcijs[i][j], Pcijs[i][j], omegaijs[i][j]) for j in range(N)] for i in range(N)]
    for P in (101325., 1e6):
        VE = v.BVirial_mixture(list(x), Bs) - sum([x[i] * Bs[i][i] for i in range(N)])
        assert_allclose(mixture.V('g', x, T, P), ideal.V('g', x, T, P) + VE, rtol=1e-12)
        assert_allclose(mixture.V('g', 2 * x, T, P), 2 * mixture.V('g', x, T, P), rtol=1e-12)
        # Pure component second virial coefficients are consistent with
        # pure component gas molar volumes
        for i, chemical in enumerate(chemicals):
            assert_allclose(chemical.V('g', T, P), 8.314462618 * T / P + Bs[i][i], rtol=1e-10)
    # Pure components follow their pure component models
    pure = np.array([0., 2., 0., 0.])
    assert mixture.V('g', pure, T, P) == ideal.V('g', pure, T, P)
    # Liquid molar volumes follow Amagat's law (no pressure dependence
    # of the activity coefficient model)
    assert_allclose(mixture.V('l', x, 300., 101325.), ideal.V('l', x, 300., 101325.), rtol=1e-14)
    # Chemicals without critical properties do not contribute to the excess volume
    from thermosteam.mixture import VolumeMixtureModel
    V = VolumeMixtureModel([i.V for i in chemicals], Tcs, Pcs, Vcs[:2] + [None, None], omegas)
    P = 101325.
    VE = sum([x[i] * x[j] * (Bs[i][j] - Bs[i][i]) for i in range(2) for j in range(2)])
    assert_allclose(V('g', x, T, P), ideal.V('g', x, T, P) + VE, rtol=1e-12)

def test_mixing_rules_infeasible_values():
    # Infeasible pure component values raise a RuntimeError noting the chemical
    import pytest
    from thermosteam.mixture import (
        ViscosityMixtureModel, ThermalConductivityMixtureModel, SurfaceTensionMixtureModel
    )
    chemicals = tmo.Chemicals(['Water', 'Ethanol', 'O2'], cache=True)
    mixture = tmo.PRMixture.from_chemicals(chemicals)
    x = np.array([0.4, 0.4, 0.2])
    T = 330.
    P = 101325.
    constant = lambda value: (lambda *args: value)
    models = list(mixture.mu.models)
    models[1] = constant(-1e-3)
    mu = ViscosityMixtureModel(models, mixture.mu.MWs, mixture.mu.IDs)
    for phase in ('l', 'g'):
        with pytest.raises(RuntimeError, match="viscosity .* 'Ethanol'"):
            mu(phase, x, T, P)
    models = list(mixture.kappa.V_models)
    models[2] = constant(float('nan'))
    kappa = ThermalConductivityMixtureModel(
        mixture.kappa.models, mixture.kappa.mu_models, models, 
        mixture.kappa.MWs, mixture.kappa.IDs,
    )
    with pytest.raises(RuntimeError, match="molar volume .* 'O2'"):
        kappa('l', x, T, P)
    models = list(mixture.sigma.models)
    models[0] = constant(-0.01)
    sigma = SurfaceTensionMixtureModel(models, mixture.sigma.V_models, mixture.sigma.IDs)
    with pytest.raises(RuntimeError, match="surface tension of 'Water'"):
        sigma(x, T, P)

if __name__ == '__main__':
    test_gibbs_excess_mixture_excess_properties()
    test_gibbs_excess_mixture_excess_property_consistency()
    test_gibbs_excess_mixture_partial_molar_excess_enthalpies()
    test_gibbs_excess_mixture_properties()
    test_gibbs_excess_mixture_pure_components()
    test_gibbs_excess_mixture_sparse_flows()
    test_gibbs_excess_mixture_in_streams()
    test_mixing_rules()
    test_mixing_rules_consistency()
    test_eos_mixture_gas_volume()
    test_gibbs_excess_mixture_gas_volume()
    test_mixing_rules_infeasible_values()
