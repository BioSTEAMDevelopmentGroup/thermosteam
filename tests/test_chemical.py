# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import pytest
from numpy.testing import assert_allclose

def test_chemical():
    import thermosteam as tmo
    pubchem_db = tmo.chemicals.identifiers.pubchem_db
    pubchem_db.load(pubchem_db.unloaded_files.pop())
    for i in pubchem_db.CAS_index.values():
        # Simply make sure they can be created without errors
        tmo.Chemical(i.CASs)

if __name__ == '__main__':
    test_chemical()