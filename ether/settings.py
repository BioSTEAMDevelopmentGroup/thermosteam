# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 04:12:27 2019

@author: yoelr
"""
import ether

__all__ = ('settings',)

class EtherSettings:
    __slots__ = ('_thermo', '_lazy_energy_balance')
    
    def __init__(self):
        self._thermo = None
        self._lazy_energy_balance = True
    
    @property
    def lazy_energy_balance(self):
        return self._lazy_energy_balance
    @lazy_energy_balance.setter
    def lazy_energy_balance(self, islazy):
        self._lazy_energy_balance = bool(islazy)
    
    @property
    def default_thermo(self):
        return self._thermo
    
    @default_thermo.setter
    def default_thermo(self, thermo):
        if isinstance(thermo, ether.Chemicals):
            thermo.compile()
            self._thermo = ether.Thermo(thermo)
        elif isinstance(thermo, ether.Thermo):
            self._thermo = thermo
        else:
            raise ValueError("default must be a 'Thermo' object, "
                            f"not a '{type(thermo).__name__}'")
            
settings = EtherSettings()