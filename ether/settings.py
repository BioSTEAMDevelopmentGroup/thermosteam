# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 04:12:27 2019

@author: yoelr
"""
import ether

__all__ = ('settings',)

class EtherSettings:
    __slots__ = ('_thermo', '_rigorous_energy_balance')
    
    def __init__(self):
        self._thermo = None
        self._rigorous_energy_balance = False
    
    def get_default_thermo(self, thermo):
        if not thermo:
            thermo = settings.default_thermo
            assert thermo, "no available 'Thermo' object"
        elif isinstance(thermo, ether.Chemicals):
            thermo.compile()
            thermo = ether.Thermo(thermo)
        return thermo
    
    @property
    def rigorous_energy_balance(self):
        return self._rigorous_energy_balance
    @rigorous_energy_balance.setter
    def rigorous_energy_balance(self, isrigorous):
        self._rigorous_energy_balance = bool(isrigorous)
    
    @property
    def default_thermo(self):
        return self._thermo
    
    @default_thermo.setter
    def default_thermo(self, thermo):
        if isinstance(thermo, ether.Thermo):
            self._thermo = thermo
        elif isinstance(thermo, ether.Chemicals):
            thermo.compile()
            self._thermo = ether.Thermo(thermo)
        else:
            raise ValueError("default must be a 'Thermo' object, "
                            f"not a '{type(thermo).__name__}'")
    
    def __repr__(self):
        return "ether.settings"
    
settings = EtherSettings()