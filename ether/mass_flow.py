# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 01:01:57 2019

@author: yoelr
"""
from free_properties import PropertyFactory
from .material_array import MassFlow, ChemicalMassFlow
import numpy as np

__all__ = ('MassFlowProperty', 'mass_flow_1d', 'mass_flow_2d')

# %% Mass flow properties

@PropertyFactory(slots=('name', 'molar_flow', 'index', 'MW'))
def MassFlowProperty(self):
    """Mass flow (kg/hr)."""
    return self.molar_flow[self.index] * self.MW
    
@MassFlowProperty.setter
def MassFlowProperty(self, value):
    self.molar_flow[self.index] = value/self.MW

def mass_flow_1d(molar_flow):
    chemicals = molar_flow.chemicals
    molar_flow = molar_flow.data
    mass_flow = np.zeros_like(molar_flow, dtype=object)
    for i, chem in enumerate(chemicals):
        mass_flow[i] = MassFlowProperty(chem.ID, molar_flow, i, chem.MW)
    return ChemicalMassFlow(mass_flow, chemicals)
	
def mass_flow_2d(molar_flow):
    phases = molar_flow.phases
    chemicals = molar_flow.chemicals
    molar_flow = molar_flow.data
    mass_flow = np.zeros_like(molar_flow, dtype=object)
    for i, phase in enumerate(phases):
        for j, chem in enumerate(chemicals):
            index = (i, j)
            mass_flow[index] = MassFlowProperty(chem.ID, molar_flow, index, chem.MW)
    return MassFlow.from_data(mass_flow, phases, chemicals)