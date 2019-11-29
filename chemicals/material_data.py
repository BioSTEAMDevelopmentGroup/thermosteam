# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 02:34:56 2019

@author: yoelr
"""
import numpy as np

class SinglePhaseMaterialData:
    __slots__ = ('_chemicals', '_material_data', '_phase', '_mixture')
    def __init__(self, phase, material_data, chemicals, mixture):
        self._chemicals = chemicals
        self._material_data = material_data
        self._mixture = mixture
        self._phase = phase
        
    def mix(self, other):
        assert self._chemicals is other._chemicals, "chemicals must be the same to mix material data"
        if isinstance(other, SinglePhaseMaterialData):
            self._material_data += other._material_data
        else:
            self._material_data += other._material_data.sum(0)


class MixedPhaseMaterialData:
    __slots__ = ('_chemicals', '_material_data', '_phases', '_phase_data')
    
    def __init__(self, chemicals, phases, data):
        self._chemicals = chemicals
        self._material_data = data
        self._phases = phases
        self._phase_data = dict(zip(phases, data))
        
    def mix(self, other):
        assert self._chemicals is other._chemicals, "chemicals must be the same to mix material data"
        phases = self._phases
        data = self._material_data
        if isinstance(other, SinglePhaseMaterialData):
            other_phase = other._phase
            if other_phase in phases:
                self._phase_data[other_phase] += other._material_data
            else:
                phases += other_phase
                self._phases = phases
                self._material_data = data = np.vstack((data, other._material_data))     
                self._phase_data = dict(zip(phases, data))
        else:
            other_phases = other._phases
            other_data = other._material_data
            if other_phases == phases:
                data += other_data
            else:
                new_phases = ''
                phase_data = self._phase_data
                other_phase_data = other._phase_data
                for phase in other_phases:
                    if phase in phases:
                        phase_data[phase] += other_phase_data[phase]
                    else:
                        new_phases += phase
                if new_phases:
                    new_data = [other_phase_data[phase] for phase in new_phases]
                    phases += new_phases
                    self._phases = phases              
                    self._material_data = data = np.vstack((data, new_data))     
                    self._phase_data = dict(zip(phases, data))
    
    # def __repr__(self):
    #     nonzero = np.any(self.data, 0)
    #     IDs = self.chemicals.IDs
    #     IDs = [i for i,j in zip(IDs, nonzero) if j]
    #     return f"<{type(self).__name__}:>"