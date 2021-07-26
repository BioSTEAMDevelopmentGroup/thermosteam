# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the tp_dependent_property module from the thermo library:
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
from thermo import TPDependentProperty

# Remove cache from call
def __call__(self, T, P):
    if self._method_P:
        return self.TP_dependent_property(T, P)
    else:
        return self.T_dependent_property(T)

TPDependentProperty.__call__ = __call__

def has_method(self):
    return bool(self._method or self._method_P)

TPDependentProperty.__bool__ = has_method