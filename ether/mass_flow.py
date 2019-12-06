# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 01:01:57 2019

@author: yoelr
"""
from free_properties import PropertyFactory
import numpy as np

__all__ = ('MassFlow', 'mass_flow_1d')

# %% Flow properties

@PropertyFactory(slots=('name', 'molar_flow', 'index', 'MW'))
def MassFlow(self):
    """Mass flow (kg/hr)."""
    return self.molar_flow[self.index] * self.MW
    
@MassFlow.setter
def MassFlow(self, value):
    self.molar_flow[self.index] = value/self.MW

def mass_flow_1d(molar_flow, chemicals):
    mass_flow = np.zeros_like(molar_flow, dtype=object)
    for i, chem in enumerate(chemicals):
        mass_flow[i] = MassFlow(chem.ID, molar_flow, i, chem.MW)
    return mass_flow