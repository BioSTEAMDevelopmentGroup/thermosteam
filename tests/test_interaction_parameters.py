# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import thermosteam as tmo
from thermosteam import equilibrium as eq
from thermosteam.equilibrium.interaction_parameters import Lambda_PR, Lambda_SRK
from thermo.eos_mix import PRMIX, SRKMIX
from thermo.group_contribution.ppr78 import PPR78_kij
from numpy.testing import assert_allclose

def test_PR2SRK_kij():
    # PR and SRK should give the same excess Gibbs energy at infinite pressure
    Tcs = [126.2, 305.32]
    Pcs = [3394387., 4872000.]
    omegas = [0.0372, 0.0995]
    T = 200.
    zs = [0.3, 0.7]
    kij_PR = 0.0533
    kij_SRK = eq.PR2SRK_kij(kij_PR, T, Tcs[0], Pcs[0], omegas[0], Tcs[1], Pcs[1], omegas[1])
    gEs = []
    for EOS, Lambda, kij in [(PRMIX, Lambda_PR, kij_PR), (SRKMIX, Lambda_SRK, kij_SRK)]:
        eos = EOS(Tcs=Tcs, Pcs=Pcs, omegas=omegas, zs=zs, kijs=[[0, kij], [kij, 0]], T=T, P=1e5)
        gEs.append(-Lambda * (eos.a_alpha / eos.b - sum([z * a / b for z, a, b in zip(zs, eos.a_alphas, eos.bs)])))
    assert_allclose(*gEs, rtol=1e-12)

def test_SRK_kijs():
    chemicals = tmo.Chemicals(['N2', 'Ethane'], cache=True)
    gamma = eq.SRKActivityCoefficients(chemicals)
    gamma([0.3, 0.7], 200., 2e6)
    assert gamma._eos.kijs[0][1] == 0.0174 # From ChemSep SRK (DECHEMA)

def test_kijs_from_PPR78():
    # Hexane/toluene is not in ChemSep SRK or PR data, so E-PPR78 is used
    hexane, toluene = tmo.Chemicals(['Hexane', 'Toluene'], cache=True)
    assert eq.PPR78_groups(hexane) == {1: 2, 2: 4} # 2 CH3, 4 CH2
    assert eq.PPR78_groups(toluene) == {7: 5, 8: 1, 1: 1} # 5 ACH, 1 AC, 1 CH3
    T = 298.15
    kij = PPR78_kij(T, {1: 2, 2: 4}, {7: 5, 8: 1, 1: 1}, hexane.Tc, hexane.Pc, hexane.omega,
                    toluene.Tc, toluene.Pc, toluene.omega, version='extended', string=False)
    assert eq.PR_kijs([hexane, toluene])[0][1] == kij
    kijs = eq.SRK_kijs([hexane, toluene])
    assert_allclose(
        kijs[0][1],
        eq.PR2SRK_kij(kij, T, hexane.Tc, hexane.Pc, hexane.omega, toluene.Tc, toluene.Pc, toluene.omega)
    )

if __name__ == '__main__':
    test_PR2SRK_kij()
    test_SRK_kijs()
    test_kijs_from_PPR78()
