# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the t_dependent_property module from the thermo library:
# https://github.com/CalebBell/thermo
# Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>
#
# This module is under a dual license:
# 1. The UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
# 
# 2. The MIT open-source license. See
# https://github.com/CalebBell/chemicals/blob/master/LICENSE.txt for details.
from thermo import TDependentProperty
TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = True

# Remove cache from call
TDependentProperty.__call__ = TDependentProperty.T_dependent_property

# Backwards compatibility with past thermosteam versions
TDependentProperty.add_model = TDependentProperty.add_method