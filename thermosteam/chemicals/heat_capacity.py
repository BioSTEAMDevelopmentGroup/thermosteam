# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the heat_capacity module from the chemicals's library:
# https://github.com/CalebBell/chemicals
# Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>
#
# This module is under a dual license:
# 1. The UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
# 
# 2. The MIT open-source license. See
# https://github.com/CalebBell/chemicals/blob/master/LICENSE.txt for details.
from chemicals import heat_capacity as hc
from ..base import PhaseTHandleBuilder

def heat_capacity_handle(CAS, sdata, ldata, gdata):
    return PhaseTHandleBuilder(
        hc.HeatCapacitySolid(CAS, *sdata),
        hc.HeatCapacityLiquid(CAS, *ldata),
        hc.HeatCapacityGas(CAS, *gdata),
    )