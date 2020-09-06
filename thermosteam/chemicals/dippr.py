# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the dippr module from the chemicals's library:
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
from chemicals import dippr
from ..base import functor

EQ100 = functor(dippr.EQ100)
EQ101 = functor(dippr.EQ101)
EQ102 = functor(dippr.EQ102)
EQ104 = functor(dippr.EQ104)
EQ105 = functor(dippr.EQ105)
EQ106 = functor(dippr.EQ106)
EQ107 = functor(dippr.EQ107)
EQ114 = functor(dippr.EQ114)
EQ115 = functor(dippr.EQ115)
EQ116 = functor(dippr.EQ116)
EQ127 = functor(dippr.EQ127)
