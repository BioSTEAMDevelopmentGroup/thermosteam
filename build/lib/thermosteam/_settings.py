# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 04:12:27 2019

@author: yoelr
"""
import thermosteam as tmo

__all__ = ('settings',)

class Settings:
    __slots__ = ('_thermo',
                 '_phase_names',
                 '_debug',
    )
    
    def __init__(self):
        self._thermo = None
        self._debug = __debug__
        self._phase_names = {'s': 'Solid',
                             'l': 'Liquid',
                             'g': 'Gas',
                             'S': 'SOLID',
                             'L': 'LIQUID',
                             'G': 'GAS'}
    
    @property
    def debug(self):
        return self._debug
    @debug.setter
    def debug(self, debug):
        self._debug = bool(debug)
    
    @property
    def phase_names(self):
        return self._phase_names
    
    def get_default_thermo(self, thermo):
        if not thermo:
            thermo = self.get_thermo()
        return thermo
    
    def get_default_chemicals(self, chemicals):
        if isinstance(chemicals, tmo.Chemicals):
            chemicals.compile()
        elif not chemicals:
            thermo = settings._thermo
            assert thermo, "no available 'Chemicals' object"
            chemicals = thermo.chemicals
        else:
            raise ValueError("chemicals must be a 'Chemicals' object")
        return chemicals
    
    def get_default_mixture(self, mixture):
        if not mixture:
            thermo = settings.thermo
            assert thermo, ("no available 'Thermo' object; "
                            "set settings.thermo first")
            mixture = thermo.mixture
        return mixture
    
    def get_thermo(self):
        thermo = self._thermo
        assert thermo, ("no available 'Thermo' object; "
                        "use settings.set_thermo")
        return thermo
    
    def set_thermo(self, thermo):
        if isinstance(thermo, tmo.Chemicals):
            thermo = tmo.Thermo(thermo)
        elif not isinstance(thermo, tmo.Thermo):
            raise ValueError("thermo must be either a 'Thermo' "
                             "or a 'Chemicals' object, not a "
                            f"'{type(thermo).__name__}'")
        self._thermo = thermo
    
    def __repr__(self):
        return "<Settings>"
    
settings = Settings()