# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 03:42:22 2019

@author: yoelr
"""
from free_properties import PropertyFactory, property_array
from .material_array import VolumetricFlow, ChemicalVolumetricFlow
import numpy as np

__all__ = ('VolumetricFlowProperty', 'volumetric_flow_1d', 'volumetric_flow_2d')

# %% Volumetric flow properties

@PropertyFactory(slots=('name', 'molar_flow', 'index', 'V', 'thermal_condition', 'phase'))
def VolumetricFlowProperty(self):
    """Volumetric flow (m^3/hr)."""
    phase, T, P = self.thermal_condition
    return self.molar_flow[self.index] * self.V(phase or self.phase, T, P)
    
@VolumetricFlowProperty.setter
def VolumetricFlowProperty(self, value):
    phase, T, P = self.thermal_condition
    self.molar_flow[self.index] = value / self.V(phase or self.phase, T, P)

def volumetric_flow_1d(molar_flow, thermal_condition):
    chemicals = molar_flow.chemicals
    molar_flow = molar_flow.data
    volumetric_flow = np.zeros_like(molar_flow, dtype=object)
    for i, chem in enumerate(chemicals):
        volumetric_flow[i] = VolumetricFlowProperty(chem.ID, molar_flow, i, chem.V, thermal_condition)
    return ChemicalVolumetricFlow(property_array(volumetric_flow), chemicals)
	
def volumetric_flow_2d(molar_flow, thermal_condition):
    phases = molar_flow.phases
    chemicals = molar_flow.chemicals
    molar_flow = molar_flow.data
    volumetric_flow = np.zeros_like(molar_flow, dtype=object)
    for i, phase in enumerate(phases):
        for j, chem in enumerate(chemicals):
            index = i, j
            volumetric_flow[index] = VolumetricFlowProperty(f"{chem.ID} [{phase}]", molar_flow, index, chem.V, thermal_condition, phase)
    return VolumetricFlow.from_data(property_array(volumetric_flow), phases, chemicals)