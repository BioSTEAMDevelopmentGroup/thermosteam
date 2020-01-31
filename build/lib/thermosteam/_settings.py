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
        """[bool] If True, preventive assertions are run at runtime."""
        return self._debug
    @debug.setter
    def debug(self, debug):
        self._debug = bool(debug)
    
    @property
    def phase_names(self):
        """[dict] All phase definitions."""
        return self._phase_names
    
    def get_default_thermo(self, thermo):
        """
        Return a default Thermo object.
        If `thermo` is a Thermo object, return the same object.
        """
        if not thermo:
            thermo = self.get_thermo()
        return thermo
    
    def get_default_chemicals(self, chemicals):
        """
        Return a default CompiledChemicals object.
        If `chemicals` is a Chemicals object, return the same object.
        """
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
        """
        Return a default Mixture object.
        If `mixture` is a Mixture object, return the same object.
        """
        if not mixture:
            thermo = settings.thermo
            assert thermo, ("no available 'Thermo' object; "
                            "set settings.thermo first")
            mixture = thermo.mixture
        return mixture
    
    def get_thermo(self):
        """Return a default Thermo object."""
        thermo = self._thermo
        assert thermo, ("no available 'Thermo' object; "
                        "use settings.set_thermo")
        return thermo
    
    def set_thermo(self, thermo):
        """Set the default Thermo object."""
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