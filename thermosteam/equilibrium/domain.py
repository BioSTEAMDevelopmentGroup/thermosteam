# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""

__all__ = (
    'vle_domain',
)

Tmin_limit = 200.
Tmax_limit = 1000.

def vle_domain(chemicals):
    """
    Return the minimum and maximum temperatures at which vapor-liquid equilibrium can
    be computed for given chemicals.

    Parameters
    ----------
    chemicals : Iterable[~thermosteam.Chemical]
        Chemicals in vapor-liquid equilibrium.

    Returns
    -------
    Tmin : float
        Minimum temperature [K].
    Tmax : float
        Maximum temperature [K].

    """
    Psats = [i.Psat for i in chemicals]
    Tmax_Psat = min([i.Tmax for i in Psats])
    Tmax = min(Tmax_Psat, Tmax_limit) - 1e-2
    
    Tmin_Psat = max([i.Tmin for i in Psats])
    Tmin = max(Tmin_Psat, Tmin_limit) + 1e-2
    return (Tmin, Tmax)