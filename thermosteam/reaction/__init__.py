# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
The thermosteam.reaction package features objects to manage stoichiometric reactions and conversions.
"""

from . import _reaction

__all__ = (*_reaction.__all__,)

from ._reaction import *

#: Whether or not to check reaction feasibility.
CHECK_FEASIBILITY = True