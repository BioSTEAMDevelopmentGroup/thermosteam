# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
Tests for activity coefficient values and their analytical derivatives.
Group contribution models are compared against thermo's UNIFAC
implementation; EOS-based models are compared against thermo's CEOSLiquid
and against finite differences.
"""
import pytest
import numpy as np
import thermosteam as tmo
from thermosteam import equilibrium as eq
from thermo import unifac as tu
from thermo import CEOSLiquid, eos_mix
from thermo.interaction_parameters import IPDB
from numpy.testing import assert_allclose

# %% Utilities

def get_group_models():
    return [
        # Activity coefficient class, group name, subgroups, interaction parameters, thermo version
        (eq.UNIFACActivityCoefficients, 'UNIFAC', tu.UFSG, 'UFIP', 0),
        (eq.UNIFAC2ActivityCoefficients, 'UNIFAC', tu.UFSG, 'UF2IP', 0),
        (eq.PSRKActivityCoefficients, 'PSRK', tu.PSRKSG, 'PSRKIP', 2),
        (eq.ModifiedUNIFACActivityCoefficients, 'Dortmund', tu.DOUFSG, 'DOUFIP2016', 1),
        (eq.ModifiedUNIFAC2ActivityCoefficients, 'Dortmund', tu.DOUFSG, 'DOUF2IP', 1),
    ]

def get_group_cases():
    return [
        # Chemicals, molar fractions, temperature
        (['Water', 'Ethanol'], [0.5, 0.5], 350.),
        (['Water', 'Ethanol', 'Acetone'], [0.2, 0.5, 0.3], 320.),
        (['Hexane', 'Ethanol'], [0.7, 0.3], 300.),
    ]

def get_eos_models():
    return [
        # Activity coefficient class, thermo EOS class, ChemSep interaction parameters
        (eq.PRActivityCoefficients, eos_mix.PRMIX, 'ChemSep PR'),
        (eq.PR78ActivityCoefficients, eos_mix.PR78MIX, 'ChemSep PR'),
        (eq.SRKActivityCoefficients, eos_mix.SRKMIX, None),
    ]

def get_eos_cases():
    return [
        # Chemicals, molar fractions, temperature, pressure
        (['Hexane', 'Toluene'], [0.5, 0.5], 350., 101325.),
        (['Water', 'Ethanol'], [0.3, 0.7], 340., 101325.),
        (['Methane', 'Hexane'], [0.1, 0.9], 300., 5e6), # Methane is supercritical
        (['Water', 'Ethanol', 'Hexane'], [0.2, 0.3, 0.5], 330., 2e5),
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

# %% Group contribution models

def test_group_activity_coefficients():
    for Gamma, group_name, subgroups, interactions, version in get_group_models():
        for IDs, x, T in get_group_cases():
            chemicals = tmo.Chemicals(IDs, cache=True)
            x = np.array(x)
            gamma = Gamma(chemicals)
            ref = thermo_unifac(chemicals, x.tolist(), T, group_name, subgroups, interactions, version)
            gammas = np.array(ref.gammas())
            dlngammas_dT = np.array(ref.dgammas_dT()) / gammas
            d2lngammas_dT2 = np.array(ref.d2lngammas_dT2())
            args = gamma.args
            msg = f"{Gamma.__name__}: {IDs}"
            assert_allclose(gamma(x, T), gammas, rtol=1e-12, err_msg=msg)
            assert_allclose(gamma.f(x, T, *args), gammas, rtol=1e-12, err_msg=msg)
            assert_allclose(
                gamma.f_der_T(x, T, *args),
                gammas * dlngammas_dT,
                rtol=1e-10, err_msg=msg,
            )
            assert_allclose(
                gamma.f_der2_T(x, T, *args),
                gammas * (d2lngammas_dT2 + dlngammas_dT * dlngammas_dT),
                rtol=1e-10, err_msg=msg,
            )
            assert_allclose(gamma.f_der_P(x, T, *args), 0., err_msg=msg)

def test_group_activity_coefficients_with_missing_groups():
    chemicals = tmo.Chemicals(['Hexane', 'Ethanol', 'O2', 'Water'], cache=True)
    x = np.array([0.4, 0.3, 0.1, 0.2])
    T = 300.
    for Gamma, *_ in get_group_models():
        gamma = Gamma(chemicals)
        args = gamma.args
        msg = Gamma.__name__
        # Chemicals without groups have activity coefficients of 1
        assert gamma(x, T)[2] == 1., msg
        assert gamma.f_der_T(x, T, *args)[2] == 0., msg
        assert gamma.f_der2_T(x, T, *args)[2] == 0., msg
        # Derivatives are consistent with finite differences
        f = lambda T: gamma.f(x, T, *args)
        f_der_T = lambda T: gamma.f_der_T(x, T, *args)
        assert_allclose(gamma.f_der_T(x, T, *args), central_difference(f, T, 1e-3),
                        rtol=1e-6, atol=1e-12, err_msg=msg)
        assert_allclose(gamma.f_der2_T(x, T, *args), central_difference(f_der_T, T, 1e-3),
                        rtol=1e-6, atol=1e-12, err_msg=msg)

def test_NIST_activity_coefficients():
    Water, Ethanol = chemicals = tmo.Chemicals(['Water', 'Ethanol'])
    Ethanol.NIST.set_group_counts_by_name({'CH3':1, 'CH2':1, 'OH prim':1})
    Water.NIST.set_group_counts_by_name({'H2O':1})
    x = np.array([0.4, 0.6])
    T = 330.
    gamma = eq.NISTActivityCoefficients(chemicals)
    ref = thermo_unifac(chemicals, x.tolist(), T, 'NIST', tu.NISTUFSG, 'NISTUFIP', 1)
    gammas = np.array(ref.gammas())
    dlngammas_dT = np.array(ref.dgammas_dT()) / gammas
    d2lngammas_dT2 = np.array(ref.d2lngammas_dT2())
    args = gamma.args
    assert_allclose(gamma(x, T), gammas, rtol=1e-12)
    assert_allclose(gamma.f_der_T(x, T, *args), gammas * dlngammas_dT, rtol=1e-10)
    assert_allclose(
        gamma.f_der2_T(x, T, *args),
        gammas * (d2lngammas_dT2 + dlngammas_dT * dlngammas_dT),
        rtol=1e-10,
    )

def test_ideal_activity_coefficients():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    gamma = eq.IdealActivityCoefficients(chemicals)
    x = np.array([0.5, 0.5])
    args = gamma.args
    assert_allclose(gamma(x, 300.), 1.)
    assert gamma.f(x, 300., *args) == 1.
    assert gamma.f_der_T(x, 300., *args) == 0.
    assert gamma.f_der2_T(x, 300., *args) == 0.
    assert gamma.f_der_P(x, 300., *args) == 0.
    # Group models with less than two chemicals with groups are ideal
    chemicals = tmo.Chemicals(['Water', 'O2'], cache=True)
    assert isinstance(eq.ModifiedUNIFACActivityCoefficients(chemicals), eq.IdealActivityCoefficients)

# %% EOS-based models

def test_eos_activity_coefficients():
    for Gamma, EOS, chemsep_db in get_eos_models():
        for IDs, x, T, P in get_eos_cases():
            chemicals = tmo.Chemicals(IDs, cache=True)
            x = np.array(x)
            gamma = Gamma(chemicals)
            if chemsep_db is None:
                kijs = None
            else:
                kijs = IPDB.get_ip_asymmetric_matrix(chemsep_db, [i.CAS for i in chemicals], 'kij')
            liquid = CEOSLiquid(
                EOS, dict(Tcs=[i.Tc for i in chemicals], Pcs=[i.Pc for i in chemicals],
                          omegas=[i.omega for i in chemicals], kijs=kijs),
                T=T, P=P, zs=x.tolist(),
            )
            msg = f"{Gamma.__name__}: {IDs}"
            assert_allclose(gamma(x, T, P), liquid.gammas(), rtol=1e-8, err_msg=msg)
            assert gamma.f is gamma and gamma.args == (), msg

def test_eos_activity_coefficient_derivatives():
    for Gamma, *_ in get_eos_models():
        for IDs, x, T, P in get_eos_cases():
            gamma = Gamma(tmo.Chemicals(IDs, cache=True))
            x = np.array(x)
            f_T = lambda T: gamma(x, T, P)
            f_der_T = lambda T: gamma.f_der_T(x, T, P)
            f_P = lambda P: gamma(x, T, P)
            msg = f"{Gamma.__name__}: {IDs}"
            assert_allclose(gamma.f_der_T(x, T, P), central_difference(f_T, T, 1e-3), rtol=1e-6, err_msg=msg)
            assert_allclose(gamma.f_der2_T(x, T, P), central_difference(f_der_T, T, 1e-2), rtol=1e-6, err_msg=msg)
            assert_allclose(gamma.f_der_P(x, T, P), central_difference(f_P, P, 1e-3 * P), rtol=1e-4, err_msg=msg)

def test_eos_lnphis_second_temperature_derivative():
    from thermosteam.equilibrium.activity_coefficients import lnphis_der2_T
    chemicals = tmo.Chemicals(['Hexane', 'Toluene', 'Ethanol'], cache=True)
    T = 320.
    P = 5e5
    for Gamma, *_ in get_eos_models():
        gamma = Gamma(chemicals)
        eos = gamma.eos(np.array([0.4, 0.3, 0.3]), T, P)
        dlnphis_dT = lambda T: np.array(
            eos.to_TP_zs(T=T, P=P, zs=eos.zs, only_l=True).dlnphis_dT('l')
        )
        assert_allclose(lnphis_der2_T(eos, 'l'), central_difference(dlnphis_dT, T, 1e-3),
                        rtol=1e-8, err_msg=Gamma.__name__)

def test_eos_activity_coefficients_with_missing_critical_properties():
    X = tmo.Chemical('X', search_db=False, phase='l')
    chemicals = tmo.Chemicals(['Water', 'Ethanol', X])
    with pytest.warns(RuntimeWarning):
        gamma = eq.PRActivityCoefficients(chemicals)
    x = np.array([0.3, 0.3, 0.4])
    assert gamma(x, 340.)[2] == 1.
    assert gamma.f_der_T(x, 340.)[2] == 0.
    assert gamma.f_der2_T(x, 340.)[2] == 0.
    assert gamma.f_der_P(x, 340.)[2] == 0.
    # Remaining chemicals behave as a binary mixture
    binary = eq.PRActivityCoefficients(tmo.Chemicals(['Water', 'Ethanol'], cache=True))
    assert_allclose(gamma(x, 340.)[:2], binary(np.array([0.5, 0.5]), 340.), rtol=1e-12)

def test_eos_activity_coefficients_pickle():
    import pickle
    chemicals = tmo.Chemicals(['Hexane', 'Toluene'], cache=True)
    gamma = eq.SRKActivityCoefficients(chemicals)
    new = pickle.loads(pickle.dumps(gamma))
    x = np.array([0.5, 0.5])
    assert_allclose(new(x, 350.), gamma(x, 350.))

if __name__ == '__main__':
    test_group_activity_coefficients()
    test_group_activity_coefficients_with_missing_groups()
    test_NIST_activity_coefficients()
    test_ideal_activity_coefficients()
    test_eos_activity_coefficients()
    test_eos_activity_coefficient_derivatives()
    test_eos_lnphis_second_temperature_derivative()
    test_eos_activity_coefficients_with_missing_critical_properties()
    test_eos_activity_coefficients_pickle()
