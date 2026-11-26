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
from numpy.testing import assert_allclose

def get_results(pkg, IDs, flows, P):
    tmo.settings.set_thermo(tmo.Thermo(IDs, cache=True, pkg=pkg))
    stream = tmo.Stream(None, **flows, P=P)
    T_bubble = stream.bubble_point_at_P().T
    T_dew = stream.dew_point_at_P().T
    liquid = tmo.Stream(None, **flows, T=T_bubble - 20, P=P, phase='l')
    gas = tmo.Stream(None, **flows, T=T_dew + 20, P=P, phase='g')
    return [T_bubble, T_dew, liquid.Cn, gas.Cn, liquid.H, gas.H]

def test_SRK_vs_PR_property_packages():
    # SRK and PR equations of state should give similar results
    for IDs, flows, P in [
            (['Hexane', 'Heptane'], dict(Hexane=50, Heptane=50), 101325),
            (['Benzene', 'Toluene'], dict(Benzene=40, Toluene=60), 101325),
            (['Propane', 'Butane'], dict(Propane=30, Butane=70), 10e5),
            (['Methane', 'Hexane'], dict(Methane=10, Hexane=90), 20e5),
        ]:
        assert_allclose(
            get_results('SRK', IDs, flows, P),
            get_results('PR', IDs, flows, P),
            rtol=0.04, err_msg=str(IDs),
        )

if __name__ == '__main__':
    test_SRK_vs_PR_property_packages()
