# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""

__all__ = ('all_same_chemicals',
           'all_same_phases',
           'same_chemicals',
           'same_phases',
)

def chemicals_match(chemical_users, chemicals):
    return all([chemicals is i.chemicals for i in chemical_users if i])

def phases_match(multiphase_users, phases):
    return all([phases == i.phases for i in multiphase_users if i])

def all_same_chemicals(obj, others):
    chemicals = obj.chemicals
    assert chemicals_match(others, chemicals), "chemicals do not match"
    
def all_same_phases(obj, others):
    phases = obj.phases
    assert phases_match(others, phases), "phases do not match"
    
def same_chemicals(obj, other):
    assert obj.chemicals is other.chemicals, "chemicals do not match"
    
def same_phases(obj, other):
    assert obj.phases == other.phases, "phases do not match"